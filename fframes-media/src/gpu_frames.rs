//! Decoded video frames that stay on the GPU (feature `vulkan`, [`DecodeMode::GpuFrames`]).
//!
//! A Vulkan decoder normally downloads every frame (`av_hwframe_transfer_data`) and converts
//! it to RGBA on the CPU. A render that draws on the GPU of the decoder's own device
//! (`fframes_skia_renderer` with `vulkan-video`) can use the decoded image as it is. The
//! decoder then keeps a reference to the libav frame, and the frame in the tree is a
//! [`GpuVideoFrame`]. The backend draws it from the GPU; anything that can not (a backend on
//! another device, an allocation that failed) reads [`GpuVideoFrame::pixels`], which is the
//! conversion the decoder always did.
//!
//! The frames of a decoder come from a pool of images in GPU memory. Two limits keep many
//! decoders (one per frame-tree thread) from filling it: the frames all decoders hold share
//! a byte budget ([`set_gpu_frame_budget`]), and a decoder keeps at most the frames it
//! reserved. Past either limit a frame is downloaded as usual.

use crate::markers::Markers;
use crate::video_decoder::FFmpegFrameBuf;
use crate::video_types::FrameConvertOptions;
use ffmpeg_sys_fframes::{AVColorRange, AVColorSpace, AVFrame};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use usvgr::PreloadedImageData;

/// The frames in trees: an image that stands for a [`GpuVideoFrame`], see [`Markers`].
static FRAMES: Markers<GpuVideoFrame> = Markers::new("fframes-gpu-video:");

/// The image that stands for `frame` in a tree, as large as the frame is shown. A renderer
/// that does not know it draws nothing; a backend resolves it with
/// [`resolve_gpu_video_frame`].
pub fn gpu_video_image(frame: GpuVideoFrame) -> crate::ImageData<'static> {
    let (width, height) = frame.display_size();
    FRAMES.create(frame, width, height)
}

/// The frame behind `image`, if the decoder kept it on the GPU ([`gpu_video_image`]).
/// Rendering backends call this for every raster image; a backend that does not draw such
/// frames never gets one, see [`DecodeMode::GpuFrames`](crate::DecodeMode::GpuFrames).
pub fn resolve_gpu_video_frame(image: &Arc<PreloadedImageData>) -> Option<Arc<GpuVideoFrame>> {
    FRAMES.resolve(image)
}

/// The least number of frames a decoder must be able to hold on the GPU to be worth it: the
/// one being drawn and the next one.
pub(crate) const MIN_GPU_FRAMES: usize = 2;

/// Frames of all decoders may take this much GPU memory unless the backend knows better
/// ([`set_gpu_frame_budget`]).
const DEFAULT_BUDGET: usize = 512 * 1024 * 1024;

static BUDGET: AtomicUsize = AtomicUsize::new(DEFAULT_BUDGET);
static RESERVED: AtomicUsize = AtomicUsize::new(0);

/// Sets how much GPU memory the frames that decoders keep on the GPU may take together
/// (default 512 MiB). A rendering backend that knows the size of the device sets a share of
/// it. Decoders that find the budget used up download their frames.
pub fn set_gpu_frame_budget(bytes: usize) {
    BUDGET.store(bytes, Ordering::SeqCst);
}

/// GPU memory reserved for the frames of one decoder, given back when it is dropped.
#[derive(Debug)]
pub(crate) struct Reservation {
    reserved: &'static AtomicUsize,
    bytes: usize,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.reserved.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}

/// Reserves up to `wanted` frames of `frame_bytes` from the budget. Returns the reservation
/// and the number of frames it covers, `None` when fewer than [`MIN_GPU_FRAMES`] are left.
pub(crate) fn reserve(frame_bytes: usize, wanted: usize) -> Option<(Reservation, usize)> {
    reserve_from(
        &RESERVED,
        BUDGET.load(Ordering::SeqCst),
        frame_bytes,
        wanted,
    )
}

fn reserve_from(
    reserved: &'static AtomicUsize,
    budget: usize,
    frame_bytes: usize,
    wanted: usize,
) -> Option<(Reservation, usize)> {
    let frame_bytes = frame_bytes.max(1);
    loop {
        let used = reserved.load(Ordering::SeqCst);
        let frames = (budget.saturating_sub(used) / frame_bytes).min(wanted);
        if frames < MIN_GPU_FRAMES {
            return None;
        }
        let bytes = frames * frame_bytes;
        if reserved
            .compare_exchange(used, used + bytes, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            return Some((Reservation { reserved, bytes }, frames));
        }
    }
}

/// How many frames of a decoder are alive, and how many it may keep.
#[derive(Debug)]
pub(crate) struct GpuFrames {
    _reservation: Reservation,
    live: Arc<AtomicUsize>,
    limit: usize,
}

impl GpuFrames {
    pub(crate) fn new(reservation: Reservation, limit: usize) -> Self {
        Self {
            _reservation: reservation,
            live: Arc::default(),
            limit,
        }
    }

    /// Whether the decoder may keep one more frame on the GPU.
    pub(crate) fn has_room(&self) -> bool {
        self.live.load(Ordering::SeqCst) < self.limit
    }

    /// Counts a frame that references the pool, until the guard is dropped.
    pub(crate) fn live_frame(&self) -> LiveFrame {
        self.live.fetch_add(1, Ordering::SeqCst);
        LiveFrame(Arc::clone(&self.live))
    }
}

/// A frame of the pool that something holds, see [`GpuFrames::live_frame`].
#[derive(Debug)]
pub(crate) struct LiveFrame(Arc<AtomicUsize>);

impl Clone for LiveFrame {
    fn clone(&self) -> Self {
        self.0.fetch_add(1, Ordering::SeqCst);
        Self(Arc::clone(&self.0))
    }
}

impl Drop for LiveFrame {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A decoded frame that never left the GPU: a reference to an `AV_PIX_FMT_VULKAN` frame.
///
/// The backend that draws it has to honor the timeline semaphore and the layout in its
/// `AVVkFrame` (`data[0]`) and update them after its last use, like every user of a libav
/// Vulkan frame. libav waits for the semaphore before it frees the frame, so a frame that
/// is still being read by the GPU outlives the decoder that made it.
pub struct GpuVideoFrame {
    frame: FFmpegFrameBuf,
    options: Option<FrameConvertOptions>,
    pixels: OnceLock<Option<Arc<PreloadedImageData>>>,
    picture: (u32, u32),
    display: (u32, u32),
    color_space: AVColorSpace,
    color_range: AVColorRange,
}

impl GpuVideoFrame {
    pub(crate) fn new(
        frame: FFmpegFrameBuf,
        options: Option<FrameConvertOptions>,
        picture: (u32, u32),
        display: (u32, u32),
        (color_space, color_range): (AVColorSpace, AVColorRange),
    ) -> Self {
        Self {
            frame,
            options,
            pixels: OnceLock::new(),
            picture,
            display,
            color_space,
            color_range,
        }
    }

    /// The `AV_PIX_FMT_VULKAN` frame. Valid for as long as `self`.
    pub fn as_ptr(&self) -> *mut AVFrame {
        self.frame.latest_frame_ptr()
    }

    /// The size of the picture in the image of the frame (the image itself has the coded
    /// size, e.g. 1920x1088 for 1080p H.264).
    pub fn picture_size(&self) -> (u32, u32) {
        self.picture
    }

    /// The size the frame is shown at: what the converted pixels would have. A frame whose
    /// pixels are not square (display size differing from the picture) is never kept on the
    /// GPU, so the picture is stretched to this size only by a resize.
    pub fn display_size(&self) -> (u32, u32) {
        self.display
    }

    /// Colors of the frame, the stream's where the frame says nothing.
    pub fn color(&self) -> (AVColorSpace, AVColorRange) {
        (self.color_space, self.color_range)
    }

    /// The frame as the premultiplied RGBA pixels the decoder otherwise delivers: downloaded
    /// from the GPU and converted. For a backend that can not draw the frame from the GPU.
    pub fn pixels(&self) -> Option<Arc<PreloadedImageData>> {
        self.pixels
            .get_or_init(|| unsafe {
                self.frame
                    .convert_last_decoded_frame_into_svg_image(self.options.as_ref())
                    .ok()
            })
            .clone()
    }
}

impl std::fmt::Debug for GpuVideoFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuVideoFrame")
            .field("picture", &self.picture)
            .field("display", &self.display)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: usize = 3_000_000;

    #[test]
    fn decoders_share_the_budget_and_give_it_back() {
        static RESERVED: AtomicUsize = AtomicUsize::new(0);
        let (first, frames) = reserve_from(&RESERVED, 10 * FRAME, FRAME, 6).unwrap();
        assert_eq!(frames, 6);
        // only four frames are left, and a decoder takes what is there
        let (second, frames) = reserve_from(&RESERVED, 10 * FRAME, FRAME, 6).unwrap();
        assert_eq!(frames, 4);
        // less than the minimum is left: that decoder downloads its frames
        assert!(reserve_from(&RESERVED, 10 * FRAME, FRAME, 6).is_none());
        assert_eq!(RESERVED.load(Ordering::SeqCst), 10 * FRAME);
        drop(first);
        // a decoder that ends gives its frames to the next one
        let (_third, frames) = reserve_from(&RESERVED, 10 * FRAME, FRAME, 9).unwrap();
        assert_eq!(frames, 6);
        drop(second);
        assert_eq!(RESERVED.load(Ordering::SeqCst), 6 * FRAME);
    }

    #[test]
    fn a_decoder_keeps_no_more_frames_than_it_reserved() {
        static RESERVED: AtomicUsize = AtomicUsize::new(0);
        let (reservation, frames) = reserve_from(&RESERVED, 4 * FRAME, FRAME, 3).unwrap();
        let gpu = GpuFrames::new(reservation, frames);
        let held: Vec<_> = (0..3).map(|_| gpu.live_frame()).collect();
        assert!(!gpu.has_room());
        drop(held);
        assert!(gpu.has_room());
        // clones count too: the frame in the tree and the one the user code holds
        let one = gpu.live_frame();
        let two = one.clone();
        assert_eq!(gpu.live.load(Ordering::SeqCst), 2);
        drop((one, two));
        assert_eq!(gpu.live.load(Ordering::SeqCst), 0);
    }
}
