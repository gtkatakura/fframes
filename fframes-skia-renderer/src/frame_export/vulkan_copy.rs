//! What the GPU contexts of the Vulkan backend that share their device with libav need to
//! submit copies of their own: the queue lock and a ring of command buffers.

use ash::vk;
use fframes::AvBuffer;
use fframes::ffmpeg_sys_fframes::{AVHWDeviceContext, AVVulkanDeviceContext};
use fframes::{FFramesRendererError, FFramesRendererResult};

pub(super) fn vulkan_error(what: &str, result: vk::Result) -> FFramesRendererError {
    FFramesRendererError::Skia(format!("{what}: {result}"))
}

/// The lock of the queue a GPU context submits to. `FFmpeg` uses the queues of the shared
/// device too and guards them with these locks, so everything Skia or we submit to the
/// queue has to hold it.
#[derive(Clone)]
pub(crate) struct SharedQueueLock {
    /// The lock other Skia contexts on the same queue take.
    pub(super) backend_lock: crate::QueueLock,
    pub(super) encoder_device: AvBuffer,
    pub(super) queue_family_index: u32,
    pub(super) queue_index: u32,
}

impl SharedQueueLock {
    /// Runs `submit` while no other thread submits to the queue.
    pub(crate) fn locked<T>(&self, submit: impl FnOnce() -> T) -> T {
        let _backend = crate::lock_queue(Some(&self.backend_lock));
        unsafe {
            let device = self.encoder_device.data::<AVHWDeviceContext>();
            let hwctx = &*(*device).hwctx.cast::<AVVulkanDeviceContext>();

            if let Some(lock) = hwctx.lock_queue {
                lock(device, self.queue_family_index, self.queue_index);
            }
            let result = submit();
            if let Some(unlock) = hwctx.unlock_queue {
                unlock(device, self.queue_family_index, self.queue_index);
            }
            result
        }
    }
}

/// A command buffer for one copy and the fence that tells when the GPU is done with it.
struct CopySlot {
    commands: vk::CommandBuffer,
    fence: vk::Fence,
    in_flight: bool,
}

/// A few command buffers used in turn for copies, each with the fence of its submission.
/// Taking a buffer that is still running waits for it.
pub(crate) struct CopyRing {
    device: ash::Device,
    command_pool: vk::CommandPool,
    slots: Vec<CopySlot>,
    next: usize,
}

/// A command buffer taken from a [`CopyRing`]: record into it, submit it with the fence, and
/// say so with [`CopyRing::submitted`].
pub(crate) struct CopyCommands {
    pub(crate) commands: vk::CommandBuffer,
    pub(crate) fence: vk::Fence,
    slot: usize,
}

impl CopyRing {
    pub(crate) fn new(
        device: &ash::Device,
        queue_family_index: u32,
        size: usize,
    ) -> FFramesRendererResult<Self> {
        unsafe {
            let command_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(queue_family_index)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|err| vulkan_error("can not create a command pool", err))?;

            let mut slots = Vec::with_capacity(size);
            let allocated = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(size as u32),
                )
                .and_then(|buffers| {
                    for commands in buffers {
                        slots.push(CopySlot {
                            commands,
                            fence: device.create_fence(&vk::FenceCreateInfo::default(), None)?,
                            in_flight: false,
                        });
                    }
                    Ok(())
                });

            if let Err(err) = allocated {
                for slot in &slots {
                    device.destroy_fence(slot.fence, None);
                }
                device.destroy_command_pool(command_pool, None);
                return Err(vulkan_error("can not allocate command buffers", err));
            }
            Ok(Self {
                device: device.clone(),
                command_pool,
                slots,
                next: 0,
            })
        }
    }

    /// The next command buffer, reset, once the GPU is done with its last submission.
    pub(crate) fn take(&mut self) -> FFramesRendererResult<CopyCommands> {
        let index = self.next;
        self.next = (self.next + 1) % self.slots.len();
        let slot = &mut self.slots[index];
        unsafe {
            if std::mem::take(&mut slot.in_flight) {
                self.device
                    .wait_for_fences(&[slot.fence], true, u64::MAX)
                    .map_err(|err| vulkan_error("waiting for a frame copy failed", err))?;
            }
            self.device
                .reset_fences(&[slot.fence])
                .and_then(|()| {
                    self.device
                        .reset_command_buffer(slot.commands, vk::CommandBufferResetFlags::empty())
                })
                .map_err(|err| vulkan_error("can not reuse a command buffer", err))?;
        }
        Ok(CopyCommands {
            commands: slot.commands,
            fence: slot.fence,
            slot: index,
        })
    }

    /// The buffer was submitted with its fence.
    pub(crate) fn submitted(&mut self, copy: &CopyCommands) {
        self.slots[copy.slot].in_flight = true;
    }

    /// Waits until the copies that are still running on the GPU are done.
    pub(crate) fn wait(&mut self) {
        for slot in &mut self.slots {
            if std::mem::take(&mut slot.in_flight) {
                unsafe {
                    let _ = self.device.wait_for_fences(&[slot.fence], true, u64::MAX);
                }
            }
        }
    }
}

impl Drop for CopyRing {
    fn drop(&mut self) {
        self.wait();
        unsafe {
            for slot in &self.slots {
                self.device.destroy_fence(slot.fence, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}
