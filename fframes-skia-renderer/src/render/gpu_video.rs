//! Decoded video frames drawn straight from the GPU (feature `vulkan-video`).
//!
//! The decoder (`fframes_media::set_gpu_video_frames`) leaves frames in the images of libav's
//! Vulkan frame pool. Skia samples those images in place, through a YCbCr sampler, on the
//! device it shares with libav. Libav orders the users of a frame with a timeline semaphore
//! (`AVVkFrame::sem`) and tracks its layout, so a frame is used like this:
//!
//! 1. [`image`] locks the frame (`lock_frame`) and wraps its image for Skia. The lock stays
//!    until [`release`], so the decoder can not change the frame's state in between.
//! 2. [`waits`]: before Skia's submission, an empty submission waits for the decoder.
//! 3. [`release`]: after Skia's submission, an empty submission signals the next value of the
//!    semaphore, which covers everything Skia submitted before. The next user of the frame
//!    (the decoder, when the pool recycles it) waits for that value.

use ash::vk::{self, Handle};
use fframes::ffmpeg_sys_fframes::{
    AVChromaLocation, AVColorRange, AVColorSpace, AVHWFramesContext, AVVkFrame,
    AVVulkanDeviceContext, AVVulkanFramesContext,
};
use fframes::media::GpuVideoFrame;
use skia_safe::gpu::{self, SurfaceOrigin, ganesh::vk::backend_textures};
use skia_safe::{AlphaType, Canvas, ColorType, Image};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

/// A frame in use by the frame being drawn on this thread.
struct Used {
    frame: Arc<GpuVideoFrame>,
    vk_frame: *mut AVVkFrame,
    frames: *mut AVHWFramesContext,
    texture: gpu::BackendTexture,
    image: Image,
}

thread_local! {
    static USED: RefCell<Vec<Used>> = const { RefCell::new(Vec::new()) };
}

/// The frame as a Skia image and the size of its picture (the image itself has the coded
/// size, e.g. 1920x1088 for 1080p H.264).
pub(super) fn image(frame: &Arc<GpuVideoFrame>, canvas: &Canvas) -> Option<(Image, (i32, i32))> {
    // `data[0]` of a Vulkan frame is the `AVVkFrame` libav allocated.
    #[allow(clippy::cast_ptr_alignment)]
    unsafe {
        let raw = frame.as_ptr();
        let picture = ((*raw).width, (*raw).height);
        let vk_frame = (*raw).data[0].cast::<AVVkFrame>();
        if vk_frame.is_null() || (*raw).hw_frames_ctx.is_null() {
            return None;
        }
        if let Some(used) = USED.with_borrow(|used| {
            used.iter()
                .find(|used| used.vk_frame == vk_frame)
                .map(|used| used.image.clone())
        }) {
            return Some((used, picture));
        }
        // one image per frame; planes in separate images are not wrapped
        if (*vk_frame).img[1] != 0 {
            return None;
        }

        let frames = (*(*raw).hw_frames_ctx).data.cast::<AVHWFramesContext>();
        let frames_hwctx = &*(*frames).hwctx.cast::<AVVulkanFramesContext>();
        let mut context = canvas.recording_context()?;
        let ycbcr = ycbcr_conversion(frames, frame)?;

        if let Some(lock) = frames_hwctx.lock_frame {
            lock(frames, vk_frame);
        }
        let texture = backend_texture(
            &*vk_frame,
            frames_hwctx,
            ycbcr,
            ((*frames).width, (*frames).height),
        );
        let image = gpu::images::borrow_texture_from(
            &mut context,
            &texture,
            SurfaceOrigin::TopLeft,
            ColorType::RGB888x,
            AlphaType::Opaque,
            None,
        );
        let Some(image) = image else {
            if let Some(unlock) = frames_hwctx.unlock_frame {
                unlock(frames, vk_frame);
            }
            return None;
        };
        USED.with_borrow_mut(|used| {
            used.push(Used {
                frame: Arc::clone(frame),
                vk_frame,
                frames,
                texture,
                image: image.clone(),
            });
        });
        Some((image, picture))
    }
}

/// The semaphores (and values) the decoding of the frames in use signals.
pub(crate) fn waits() -> Vec<(u64, u64)> {
    USED.with_borrow(|used| {
        used.iter()
            .map(|used| unsafe { ((*used.vk_frame).sem[0], (*used.vk_frame).sem_value[0]) })
            .collect()
    })
}

/// Ends the use of the frames after everything that samples them is submitted. `signal`
/// submits an empty batch that signals these semaphores (and values) and returns whether it
/// worked. The frames are handed over to their next user either way.
pub(crate) fn release(signal: impl FnOnce(&[(u64, u64)]) -> bool) {
    let used = USED.take();
    if used.is_empty() {
        return;
    }
    let signals: Vec<_> = used
        .iter()
        .map(|used| unsafe { ((*used.vk_frame).sem[0], (*used.vk_frame).sem_value[0] + 1) })
        .collect();
    if !signal(&signals) {
        eprintln!("fframes: can not hand video frames back to the decoder");
    }
    for used in used {
        unsafe {
            let vk_frame = &mut *used.vk_frame;
            vk_frame.sem_value[0] += 1;
            // Skia moved the image to the layout it sampled it in
            if let Some(info) = gpu::backend_textures::get_vk_image_info(&used.texture) {
                vk_frame.layout[0] = info.layout as i32;
            }
            vk_frame.access[0] = vk::AccessFlags2::SHADER_SAMPLED_READ.as_raw();
            let frames_hwctx = &*(*used.frames).hwctx.cast::<AVVulkanFramesContext>();
            if let Some(unlock) = frames_hwctx.unlock_frame {
                unlock(used.frames, used.vk_frame);
            }
        }
        drop(used.image);
        drop(used.frame);
    }
}

/// Ends the use of the frames of a frame that was not submitted: their state is untouched.
pub(crate) fn discard() {
    for used in USED.take() {
        unsafe {
            let frames_hwctx = &*(*used.frames).hwctx.cast::<AVVulkanFramesContext>();
            if let Some(unlock) = frames_hwctx.unlock_frame {
                unlock(used.frames, used.vk_frame);
            }
        }
    }
}

/// Wraps the image of a Vulkan frame for Skia, in the layout libav left it in.
unsafe fn backend_texture(
    frame: &AVVkFrame,
    frames: &AVVulkanFramesContext,
    ycbcr: gpu::vk::YcbcrConversionInfo,
    size: (i32, i32),
) -> gpu::BackendTexture {
    unsafe {
        let mut info = gpu::vk::ImageInfo::new(
            frame.img[0] as _,
            gpu::vk::Alloc::default(),
            gpu::vk::ImageTiling::OPTIMAL,
            // every layout a frame can be in is a variant of Skia's enum
            std::mem::transmute::<u32, gpu::vk::ImageLayout>(frame.layout[0] as u32),
            std::mem::transmute::<u32, gpu::vk::Format>(frames.format[0] as u32),
            1,
            // only frames usable by every queue family get here, see `FFmpegDecoder`
            frame.queue_family[0],
            Some(ycbcr),
            None,
            gpu::vk::SharingMode::CONCURRENT,
        );
        info.image_usage_flags = frames.usage as u32;
        info.sample_count = 1;
        backend_textures::make_vk(size, &info, "fframes video frame")
    }
}

/// The optimal tiling features of the format on the device, per (physical device, format).
fn format_features(frames: *mut AVHWFramesContext, format: i32) -> vk::FormatFeatureFlags {
    static CACHE: Mutex<Option<HashMap<(u64, i32), vk::FormatFeatureFlags>>> = Mutex::new(None);
    static ENTRY: OnceLock<Option<ash::Entry>> = OnceLock::new();
    unsafe {
        let device = (*frames).device_ctx;
        let hwctx = &*(*device).hwctx.cast::<AVVulkanDeviceContext>();
        let key = (hwctx.phys_dev as usize as u64, format);
        let mut cache = CACHE.lock().unwrap_or_else(PoisonError::into_inner);
        *cache
            .get_or_insert_default()
            .entry(key)
            .or_insert_with(|| {
                let Some(entry) = ENTRY.get_or_init(|| ash::Entry::load().ok()) else {
                    return vk::FormatFeatureFlags::empty();
                };
                let instance = ash::Instance::load(
                    entry.static_fn(),
                    vk::Instance::from_raw(hwctx.inst as usize as u64),
                );
                instance
                    .get_physical_device_format_properties(
                        vk::PhysicalDevice::from_raw(hwctx.phys_dev as usize as u64),
                        vk::Format::from_raw(format),
                    )
                    .optimal_tiling_features
            })
    }
}

/// How Skia converts the frame to RGB: the matrix and range of the stream, as the software
/// path (`sws_scale`) would. Streams that do not say are BT.601, limited range.
fn ycbcr_conversion(
    frames: *mut AVHWFramesContext,
    frame: &GpuVideoFrame,
) -> Option<gpu::vk::YcbcrConversionInfo> {
    use gpu::vk::{ChromaLocation, SamplerYcbcrModelConversion as Model, SamplerYcbcrRange};
    unsafe {
        let raw = frame.as_ptr();
        let frames_hwctx = &*(*frames).hwctx.cast::<AVVulkanFramesContext>();
        let format = frames_hwctx.format[0];
        let model = match frame.color_space {
            AVColorSpace::AVCOL_SPC_BT709 => Model::YCBCR_709,
            AVColorSpace::AVCOL_SPC_BT2020_NCL | AVColorSpace::AVCOL_SPC_BT2020_CL => {
                Model::YCBCR_2020
            }
            _ => Model::YCBCR_601,
        };
        let range = match frame.color_range {
            AVColorRange::AVCOL_RANGE_JPEG => SamplerYcbcrRange::ITU_FULL,
            _ => SamplerYcbcrRange::ITU_NARROW,
        };
        let (x, y) = match (*raw).chroma_location {
            AVChromaLocation::AVCHROMA_LOC_CENTER => {
                (ChromaLocation::MIDPOINT, ChromaLocation::MIDPOINT)
            }
            AVChromaLocation::AVCHROMA_LOC_TOPLEFT => {
                (ChromaLocation::COSITED_EVEN, ChromaLocation::COSITED_EVEN)
            }
            // H.264's default: left, centered vertically
            _ => (ChromaLocation::COSITED_EVEN, ChromaLocation::MIDPOINT),
        };
        let features = format_features(frames, format);
        let linear = features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER);
        let identity = gpu::vk::ComponentSwizzle::VK_COMPONENT_SWIZZLE_IDENTITY;
        Some(gpu::vk::YcbcrConversionInfo::new_with_format(
            std::mem::transmute::<u32, gpu::vk::Format>(format as u32),
            model,
            range,
            x,
            y,
            if linear {
                gpu::vk::Filter::LINEAR
            } else {
                gpu::vk::Filter::NEAREST
            },
            0,
            gpu::vk::ComponentMapping {
                r: identity,
                g: identity,
                b: identity,
                a: identity,
            },
            features.as_raw(),
        ))
    }
}
