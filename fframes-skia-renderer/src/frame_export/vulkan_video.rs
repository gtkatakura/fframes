//! Decoded video frames drawn from the GPU, on the Vulkan device shared with libav (feature
//! `vulkan-video`).
//!
//! libav keeps a decoded frame in a Vulkan image with both planes of NV12. Skia could sample
//! it through a YCbCr sampler, but Skia only enables that when it is told that the device has
//! the feature, which `skia-safe` has no way to say. So the planes are copied, on the GPU,
//! into two ordinary images (`R8` for luma, `R8G8` for chroma) that Skia draws as the planes
//! of a YUVA image and converts to RGB itself:
//!
//! 1. The copy waits for the decoder on the frame's timeline semaphore (`AVVkFrame::sem`),
//!    signals the next value, and leaves the frame in `TRANSFER_SRC_OPTIMAL`, as libav's own
//!    users of a frame do. The frame is not used after it, so the decoder can recycle it at
//!    once, and libav waits for the semaphore before it frees a frame: a decoder that is
//!    dropped while the copy runs does not pull the image away.
//! 2. The images of the copy are filled by that submission, which ends with a barrier to
//!    `SHADER_READ_ONLY_OPTIMAL` that every later submission to the queue waits for, Skia's
//!    draws among them. A set of images is reused for the next frame only once the frame it
//!    held is drawn, and the barrier before the next copy orders it after those draws.
//!
//! Any frame this can not take is left to the caller, which draws its pixels.

use super::vulkan_copy::{CopyRing, SharedQueueLock};
use crate::render::gpu_video::GpuVideoSource;
use ash::vk::{self, Handle};
use fframes::ffmpeg_sys_fframes::{
    AVColorRange, AVColorSpace, AVHWDeviceContext, AVHWFramesContext, AVVkFrame,
    AVVulkanDeviceContext, AVVulkanFramesContext,
};
use fframes::media::GpuVideoFrame;
use fframes::{AvBuffer, FFramesRendererResult};
use skia_safe::gpu::{
    self, BackendTexture, RecordingContext, SurfaceOrigin, YUVABackendTextures,
    ganesh::vk::backend_textures, images,
};
use skia_safe::yuva_info::{PlaneConfig, Siting, Subsampling};
use skia_safe::{Image, YUVAInfo, YUVColorSpace};
use std::sync::atomic::{AtomicU64, Ordering};

/// How many frames were drawn from the GPU by any context of the process.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// The number of decoded video frames drawn from the GPU so far (see the module docs). A frame
/// that is drawn from its pixels instead is not counted.
#[doc(hidden)]
pub fn gpu_video_frames_drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// Command buffers for copies that may run on the GPU at once.
const COPIES_IN_FLIGHT: usize = 8;

/// Sets of plane images a context keeps: the frames of one picture that are drawn before it
/// is submitted. Past that the caller draws pixels.
const MAX_SETS: usize = 4;

/// Filled by a transfer, sampled by Skia. Skia only takes images made with both transfer usages.
const PLANE_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::SAMPLED.as_raw()
        | vk::ImageUsageFlags::TRANSFER_SRC.as_raw()
        | vk::ImageUsageFlags::TRANSFER_DST.as_raw(),
);

/// One image of a plane and its memory.
struct PlaneImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
}

/// The two plane images one frame is copied into.
struct PlaneSet {
    size: (u32, u32),
    luma: PlaneImage,
    chroma: PlaneImage,
    /// Holds a frame of the picture being drawn.
    in_use: bool,
}

/// Whether the device can sample the plane formats and copy into them. Every Vulkan 1.1
/// device must for `R8` and `R8G8` (the formats are mandatory); the check is for one that
/// says otherwise.
pub(crate) fn plane_formats_supported(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> bool {
    let needed = vk::FormatFeatureFlags::SAMPLED_IMAGE
        | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
        | vk::FormatFeatureFlags::TRANSFER_SRC
        | vk::FormatFeatureFlags::TRANSFER_DST;
    [vk::Format::R8_UNORM, vk::Format::R8G8_UNORM]
        .into_iter()
        .all(|format| unsafe {
            instance
                .get_physical_device_format_properties(physical_device, format)
                .optimal_tiling_features
                .contains(needed)
        })
}

/// What decides whether a frame can be copied from, see [`FrameFacts::check`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrameFacts {
    /// The frame lives on the device of this context.
    same_device: bool,
    /// The `AVVkFrame` has a first and a second image.
    first_image: bool,
    second_image: bool,
    /// The image has the NV12 format of a Vulkan decoder (both planes in one image).
    nv12: bool,
    /// The image can be copied from.
    transfer_source: bool,
    queue_family: u32,
    /// The size of the picture, and of the image it is in.
    picture: (u32, u32),
    image_size: (i32, i32),
}

/// Why a frame is not copied from the GPU. The caller draws its pixels instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unusable {
    OtherDevice,
    NoImage,
    /// Planes in images of their own (a device without multiplanar formats).
    SeparateImages,
    NotNv12,
    NoTransferSource,
    /// The image belongs to one queue family. Copying from it on the queue of this context
    /// would need a queue family ownership transfer, which is not done: libav makes its
    /// images usable by every queue family (`VK_QUEUE_FAMILY_IGNORED`) whenever its device
    /// uses more than one, so a device that decodes on a family of its own gets those.
    OwnedByQueueFamily,
    PictureOutsideImage,
}

impl FrameFacts {
    fn check(&self) -> Result<(), Unusable> {
        if !self.same_device {
            return Err(Unusable::OtherDevice);
        }
        if !self.first_image {
            return Err(Unusable::NoImage);
        }
        if self.second_image {
            return Err(Unusable::SeparateImages);
        }
        if !self.nv12 {
            return Err(Unusable::NotNv12);
        }
        if !self.transfer_source {
            return Err(Unusable::NoTransferSource);
        }
        if self.queue_family != vk::QUEUE_FAMILY_IGNORED {
            return Err(Unusable::OwnedByQueueFamily);
        }
        let (width, height) = self.picture;
        if width == 0
            || height == 0
            || i64::from(width) > i64::from(self.image_size.0)
            || i64::from(height) > i64::from(self.image_size.1)
        {
            return Err(Unusable::PictureOutsideImage);
        }
        Ok(())
    }
}

/// See the module docs. One per GPU context, used by the thread that draws with it.
pub(crate) struct VulkanVideoFrames {
    device: ash::Device,
    /// `VkDevice` of libav's device, which has to be the one frames are decoded on.
    device_handle: u64,
    memory: vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    queue_lock: SharedQueueLock,
    ring: CopyRing,
    sets: Vec<PlaneSet>,
    /// libav's device, so that it outlives everything created on it.
    _encoder_device: AvBuffer,
}

impl VulkanVideoFrames {
    pub(crate) fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        queue: vk::Queue,
        queue_lock: SharedQueueLock,
    ) -> FFramesRendererResult<Self> {
        let encoder_device = queue_lock.encoder_device.clone();
        let device_handle = unsafe {
            let hwctx = &*(*encoder_device.data::<AVHWDeviceContext>())
                .hwctx
                .cast::<AVVulkanDeviceContext>();
            hwctx.act_dev as usize as u64
        };
        Ok(Self {
            device: device.clone(),
            device_handle,
            memory: unsafe { instance.get_physical_device_memory_properties(physical_device) },
            queue,
            ring: CopyRing::new(device, queue_lock.queue_family_index, COPIES_IN_FLIGHT)?,
            queue_lock,
            sets: Vec::new(),
            _encoder_device: encoder_device,
        })
    }

    /// Destroys the images. The GPU context that drew them has to be gone, or done with them.
    pub(crate) fn destroy(&mut self) {
        self.ring.wait();
        for set in self.sets.drain(..) {
            unsafe {
                for plane in [set.luma, set.chroma] {
                    self.device.destroy_image(plane.image, None);
                    self.device.free_memory(plane.memory, None);
                }
            }
        }
    }

    /// An image of `format` that is `size` large, filled by transfers and sampled by shaders.
    fn create_plane(
        &self,
        format: vk::Format,
        (width, height): (u32, u32),
    ) -> Result<PlaneImage, vk::Result> {
        unsafe {
            let image = self.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(PLANE_USAGE)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?;
            let requirements = self.device.get_image_memory_requirements(image);
            let memory_type = (0..self.memory.memory_type_count).find(|&index| {
                requirements.memory_type_bits & (1 << index) != 0
                    && self.memory.memory_types[index as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            });
            let allocated = memory_type
                .ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
                .and_then(|memory_type| {
                    self.device.allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(requirements.size)
                            .memory_type_index(memory_type),
                        None,
                    )
                });
            let memory = match allocated {
                Ok(memory) => memory,
                Err(err) => {
                    self.device.destroy_image(image, None);
                    return Err(err);
                }
            };
            if let Err(err) = self.device.bind_image_memory(image, memory, 0) {
                self.device.destroy_image(image, None);
                self.device.free_memory(memory, None);
                return Err(err);
            }
            Ok(PlaneImage { image, memory })
        }
    }

    /// A free set of plane images for a picture of `size`, made when there is none.
    fn take_set(&mut self, size: (u32, u32)) -> Option<usize> {
        if let Some(index) = self
            .sets
            .iter()
            .position(|set| !set.in_use && set.size == size)
        {
            self.sets[index].in_use = true;
            return Some(index);
        }
        if self.sets.len() >= MAX_SETS {
            // sets of other sizes (a video that ended) make room
            let stale = self.sets.iter().position(|set| !set.in_use)?;
            let set = self.sets.swap_remove(stale);
            self.ring.wait();
            unsafe {
                for plane in [set.luma, set.chroma] {
                    self.device.destroy_image(plane.image, None);
                    self.device.free_memory(plane.memory, None);
                }
            }
        }
        let luma = self.create_plane(vk::Format::R8_UNORM, size).ok()?;
        let Ok(chroma) = self.create_plane(
            vk::Format::R8G8_UNORM,
            (size.0.div_ceil(2), size.1.div_ceil(2)),
        ) else {
            unsafe {
                self.device.destroy_image(luma.image, None);
                self.device.free_memory(luma.memory, None);
            }
            return None;
        };
        self.sets.push(PlaneSet {
            size,
            luma,
            chroma,
            in_use: true,
        });
        Some(self.sets.len() - 1)
    }

    /// Whether `frame` can be copied from here. Returns the `AVVkFrame` and the frames context.
    // `data[0]` of a Vulkan frame is the `AVVkFrame` libav allocated.
    #[allow(clippy::cast_ptr_alignment)]
    unsafe fn usable(
        &self,
        frame: &GpuVideoFrame,
    ) -> Option<(*mut AVVkFrame, *mut AVHWFramesContext)> {
        unsafe {
            let raw = frame.as_ptr();
            let vk_frame = (*raw).data[0].cast::<AVVkFrame>();
            if vk_frame.is_null() || (*raw).hw_frames_ctx.is_null() {
                return None;
            }
            let frames = (*(*raw).hw_frames_ctx).data.cast::<AVHWFramesContext>();
            let frames_hwctx = &*(*frames).hwctx.cast::<AVVulkanFramesContext>();
            let device = &*(*(*frames).device_ctx)
                .hwctx
                .cast::<AVVulkanDeviceContext>();
            let facts = FrameFacts {
                same_device: device.act_dev as usize as u64 == self.device_handle,
                first_image: (*vk_frame).img[0] != 0,
                second_image: (*vk_frame).img[1] != 0,
                nv12: frames_hwctx.format[0] == vk::Format::G8_B8R8_2PLANE_420_UNORM.as_raw(),
                transfer_source: frames_hwctx.usage
                    & vk::ImageUsageFlags::TRANSFER_SRC.as_raw() as i32
                    != 0,
                queue_family: (*vk_frame).queue_family[0],
                picture: frame.picture_size(),
                image_size: ((*frames).width, (*frames).height),
            };
            facts.check().ok()?;
            Some((vk_frame, frames))
        }
    }

    /// Records and submits the copy of the planes of `vk_frame`. The frame is locked.
    unsafe fn copy(
        &mut self,
        vk_frame: &mut AVVkFrame,
        (width, height): (u32, u32),
        set: usize,
    ) -> Result<(), vk::Result> {
        unsafe {
            let copy = self.ring.take().map_err(|_| vk::Result::ERROR_UNKNOWN)?;
            let commands = copy.commands;
            let device = &self.device;
            let (luma, chroma) = (self.sets[set].luma.image, self.sets[set].chroma.image);
            let source = vk::Image::from_raw(vk_frame.img[0]);
            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let barrier = |image,
                           src: (vk::PipelineStageFlags2, vk::AccessFlags2),
                           dst: (vk::PipelineStageFlags2, vk::AccessFlags2),
                           old,
                           new| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(src.0)
                    .src_access_mask(src.1)
                    .dst_stage_mask(dst.0)
                    .dst_access_mask(dst.1)
                    .old_layout(old)
                    .new_layout(new)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(range)
            };
            let all = vk::PipelineStageFlags2::ALL_COMMANDS;
            let transfer = vk::PipelineStageFlags2::ALL_TRANSFER;
            let memory = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
            let layers = |aspect_mask| vk::ImageSubresourceLayers {
                aspect_mask,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };
            let copy_plane = |aspect, (width, height): (u32, u32)| vk::ImageCopy {
                src_subresource: layers(aspect),
                src_offset: vk::Offset3D::default(),
                dst_subresource: layers(vk::ImageAspectFlags::COLOR),
                dst_offset: vk::Offset3D::default(),
                extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
            };

            let record = || -> Result<(), vk::Result> {
                device.begin_command_buffer(
                    commands,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                // what the decoder left the frame in, to what the copy reads; the plane images
                // are written whole, whatever they held
                let before = [
                    barrier(
                        source,
                        (all, vk::AccessFlags2::from_raw(vk_frame.access[0])),
                        (transfer, vk::AccessFlags2::TRANSFER_READ),
                        vk::ImageLayout::from_raw(vk_frame.layout[0]),
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    ),
                    barrier(
                        luma,
                        (all, memory),
                        (transfer, vk::AccessFlags2::TRANSFER_WRITE),
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    ),
                    barrier(
                        chroma,
                        (all, memory),
                        (transfer, vk::AccessFlags2::TRANSFER_WRITE),
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    ),
                ];
                device.cmd_pipeline_barrier2(
                    commands,
                    &vk::DependencyInfo::default().image_memory_barriers(&before),
                );
                for (aspect, target, extent) in [
                    (vk::ImageAspectFlags::PLANE_0, luma, (width, height)),
                    (
                        vk::ImageAspectFlags::PLANE_1,
                        chroma,
                        (width.div_ceil(2), height.div_ceil(2)),
                    ),
                ] {
                    device.cmd_copy_image(
                        commands,
                        source,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        target,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[copy_plane(aspect, extent)],
                    );
                }
                // the planes are drawn by everything submitted after this
                let after = [
                    barrier(
                        luma,
                        (transfer, vk::AccessFlags2::TRANSFER_WRITE),
                        (all, vk::AccessFlags2::MEMORY_READ),
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    ),
                    barrier(
                        chroma,
                        (transfer, vk::AccessFlags2::TRANSFER_WRITE),
                        (all, vk::AccessFlags2::MEMORY_READ),
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    ),
                ];
                device.cmd_pipeline_barrier2(
                    commands,
                    &vk::DependencyInfo::default().image_memory_barriers(&after),
                );
                device.end_command_buffer(commands)
            };
            record()?;

            // wait for the decoder, signal the next value for whoever uses the frame after us
            let semaphore = |value: u64| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(vk::Semaphore::from_raw(vk_frame.sem[0]))
                    .value(value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            };
            let waits = [semaphore(vk_frame.sem_value[0])];
            let signals = [semaphore(vk_frame.sem_value[0] + 1)];
            let command_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(commands)];
            let submit = vk::SubmitInfo2::default()
                .wait_semaphore_infos(&waits)
                .command_buffer_infos(&command_infos)
                .signal_semaphore_infos(&signals);
            self.queue_lock
                .locked(|| device.queue_submit2(self.queue, &[submit], copy.fence))?;
            self.ring.submitted(&copy);

            vk_frame.sem_value[0] += 1;
            vk_frame.layout[0] = vk::ImageLayout::TRANSFER_SRC_OPTIMAL.as_raw();
            vk_frame.access[0] = vk::AccessFlags2::TRANSFER_READ.as_raw() as _;
            Ok(())
        }
    }
}

impl GpuVideoSource for VulkanVideoFrames {
    fn image(&mut self, context: &mut RecordingContext, frame: &GpuVideoFrame) -> Option<Image> {
        unsafe {
            let (vk_frame, frames) = self.usable(frame)?;
            let size = frame.picture_size();
            let set = self.take_set(size)?;

            let frames_hwctx = &*(*frames).hwctx.cast::<AVVulkanFramesContext>();
            if let Some(lock) = frames_hwctx.lock_frame {
                lock(frames, vk_frame);
            }
            let copied = self.copy(&mut *vk_frame, size, set);
            if let Some(unlock) = frames_hwctx.unlock_frame {
                unlock(frames, vk_frame);
            }
            if copied.is_err() {
                self.sets[set].in_use = false;
                return None;
            }

            let image = yuva_image(context, &self.sets[set], frame);
            if image.is_some() {
                DRAWN.fetch_add(1, Ordering::Relaxed);
            } else {
                self.sets[set].in_use = false;
            }
            image
        }
    }

    fn end_frame(&mut self) {
        for set in &mut self.sets {
            set.in_use = false;
        }
    }
}

/// The two planes of `set` as one image that Skia converts to RGB when it draws it.
unsafe fn yuva_image(
    context: &mut RecordingContext,
    set: &PlaneSet,
    frame: &GpuVideoFrame,
) -> Option<Image> {
    let (width, height) = set.size;
    let info = YUVAInfo::new(
        (width as i32, height as i32),
        PlaneConfig::Y_UV,
        Subsampling::S420,
        yuv_color_space(frame.color()),
        None,
        // Skia only knows chroma centered between the luma samples
        Some((Siting::Centered, Siting::Centered)),
    )?;
    let textures = [
        plane_texture(set.luma.image, vk::Format::R8_UNORM, (width, height)),
        plane_texture(
            set.chroma.image,
            vk::Format::R8G8_UNORM,
            (width.div_ceil(2), height.div_ceil(2)),
        ),
    ];
    let textures = YUVABackendTextures::new(&info, &textures, SurfaceOrigin::TopLeft)?;
    images::texture_from_yuva_textures(context, &textures, None)
}

/// A plane image, in the layout the copy left it in, as a Skia texture.
fn plane_texture(
    image: vk::Image,
    format: vk::Format,
    (width, height): (u32, u32),
) -> BackendTexture {
    unsafe {
        let mut info = gpu::vk::ImageInfo::new(
            image.as_raw() as _,
            gpu::vk::Alloc::default(),
            gpu::vk::ImageTiling::OPTIMAL,
            gpu::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            std::mem::transmute::<i32, gpu::vk::Format>(format.as_raw()),
            1,
            None,
            None,
            gpu::Protected::No,
            None,
        );
        info.image_usage_flags = PLANE_USAGE.as_raw();
        info.sample_count = 1;
        backend_textures::make_vk((width as i32, height as i32), &info, "fframes video plane")
    }
}

/// The matrix and range of the frame, as the software conversion would use them: streams that
/// do not say are BT.601, limited range.
fn yuv_color_space((space, range): (AVColorSpace, AVColorRange)) -> YUVColorSpace {
    let full = range == AVColorRange::AVCOL_RANGE_JPEG;
    match (space, full) {
        (AVColorSpace::AVCOL_SPC_BT709, true) => YUVColorSpace::Rec709_Full,
        (AVColorSpace::AVCOL_SPC_BT709, false) => YUVColorSpace::Rec709_Limited,
        (AVColorSpace::AVCOL_SPC_BT2020_NCL | AVColorSpace::AVCOL_SPC_BT2020_CL, true) => {
            YUVColorSpace::BT2020_8bit_Full
        }
        (AVColorSpace::AVCOL_SPC_BT2020_NCL | AVColorSpace::AVCOL_SPC_BT2020_CL, false) => {
            YUVColorSpace::BT2020_8bit_Limited
        }
        (_, true) => YUVColorSpace::JPEG,
        (_, false) => YUVColorSpace::Rec601_Limited,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usable() -> FrameFacts {
        FrameFacts {
            same_device: true,
            first_image: true,
            second_image: false,
            nv12: true,
            transfer_source: true,
            queue_family: vk::QUEUE_FAMILY_IGNORED,
            picture: (1920, 1080),
            image_size: (1920, 1088),
        }
    }

    #[test]
    fn a_decoded_nv12_frame_of_the_device_is_copied() {
        assert_eq!(usable().check(), Ok(()));
    }

    #[test]
    fn every_other_frame_falls_back_to_its_pixels() {
        let cases = [
            (
                FrameFacts {
                    same_device: false,
                    ..usable()
                },
                Unusable::OtherDevice,
            ),
            (
                FrameFacts {
                    first_image: false,
                    ..usable()
                },
                Unusable::NoImage,
            ),
            (
                FrameFacts {
                    second_image: true,
                    ..usable()
                },
                Unusable::SeparateImages,
            ),
            (
                FrameFacts {
                    nv12: false,
                    ..usable()
                },
                Unusable::NotNv12,
            ),
            (
                FrameFacts {
                    transfer_source: false,
                    ..usable()
                },
                Unusable::NoTransferSource,
            ),
            // owned by a decode queue family, say
            (
                FrameFacts {
                    queue_family: 3,
                    ..usable()
                },
                Unusable::OwnedByQueueFamily,
            ),
            (
                FrameFacts {
                    picture: (1920, 1090),
                    ..usable()
                },
                Unusable::PictureOutsideImage,
            ),
            (
                FrameFacts {
                    picture: (0, 1080),
                    ..usable()
                },
                Unusable::PictureOutsideImage,
            ),
        ];
        for (facts, reason) in cases {
            assert_eq!(facts.check(), Err(reason), "{facts:?}");
        }
    }

    #[test]
    fn colors_follow_the_stream_and_default_to_bt601_limited() {
        use AVColorRange::{AVCOL_RANGE_JPEG as FULL, AVCOL_RANGE_MPEG as LIMITED};
        use AVColorSpace::*;
        let unspecified = AVColorRange::AVCOL_RANGE_UNSPECIFIED;
        let cases = [
            ((AVCOL_SPC_BT709, LIMITED), YUVColorSpace::Rec709_Limited),
            ((AVCOL_SPC_BT709, FULL), YUVColorSpace::Rec709_Full),
            (
                (AVCOL_SPC_BT2020_NCL, LIMITED),
                YUVColorSpace::BT2020_8bit_Limited,
            ),
            (
                (AVCOL_SPC_BT2020_NCL, FULL),
                YUVColorSpace::BT2020_8bit_Full,
            ),
            (
                (AVCOL_SPC_UNSPECIFIED, unspecified),
                YUVColorSpace::Rec601_Limited,
            ),
            ((AVCOL_SPC_BT470BG, LIMITED), YUVColorSpace::Rec601_Limited),
            ((AVCOL_SPC_SMPTE170M, FULL), YUVColorSpace::JPEG),
        ];
        for (colors, expected) in cases {
            assert_eq!(yuv_color_space(colors), expected, "{colors:?}");
        }
    }
}
