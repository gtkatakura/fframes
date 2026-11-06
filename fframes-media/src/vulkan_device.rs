//! The one Vulkan device of a render (feature `vulkan`).
//!
//! Opening a Vulkan device per decoder (one per frame-tree thread) crashes the NVIDIA driver
//! inside `vkCreateDevice` once many threads do it at the same time, and every device costs
//! memory. The decoders of a render that draws their frames on the GPU, the Skia context and
//! the video encoder share this device instead: images of one device can be used by all of them
//! without a copy.
//!
//! Nothing else uses it: decoders in other [`DecodeMode`](crate::DecodeMode)s open their own
//! devices as before.

use ffmpeg_sys_fframes::*;
use std::ptr;
use std::sync::Mutex;

struct SharedDevice(*mut AVBufferRef);

// The reference is only used to make new references, which libav allows from any thread.
unsafe impl Send for SharedDevice {}

/// Kept for the life of the process: a device is cheap next to the renders that use it, and
/// destroying one while another thread starts to use it is the race this avoids.
static DEVICE: Mutex<Option<SharedDevice>> = Mutex::new(None);

/// A new reference to the shared `AV_HWDEVICE_TYPE_VULKAN` device, created on first use.
/// Returns a libav error code (negative) when libav can not open a Vulkan device; the
/// reference belongs to the caller on success.
///
/// # Safety
/// `out` must be valid for writes.
pub unsafe fn shared_vulkan_device(out: *mut *mut AVBufferRef) -> i32 {
    unsafe {
        let mut shared = DEVICE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if shared.is_none() {
            let mut device = ptr::null_mut();
            let status = av_hwdevice_ctx_create(
                &raw mut device,
                AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN,
                ptr::null(),
                ptr::null_mut(),
                0,
            );
            if status < 0 {
                return status;
            }
            *shared = Some(SharedDevice(device));
        }
        let reference = av_buffer_ref(shared.as_ref().expect("just created").0);
        *out = reference;
        if reference.is_null() {
            AVERROR(ENOMEM)
        } else {
            0
        }
    }
}
