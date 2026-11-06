//! Where the renderer gets decoded video frames that stayed on the GPU from.
//!
//! The decoder (`fframes_media::DecodeMode::GpuFrames`) keeps its frames in libav's pool of
//! Vulkan images, and the frame in the tree is a marker for a
//! [`GpuVideoFrame`](fframes::media::GpuVideoFrame). A GPU context that can draw it provides
//! a [`GpuVideoSource`] (see `frame_export::vulkan_video`); otherwise the pixels of the frame
//! are downloaded and drawn as usual.

use fframes::media::GpuVideoFrame;
use skia_safe::Image;
use skia_safe::gpu::RecordingContext;

/// Turns frames on the GPU into images of one Skia context.
pub trait GpuVideoSource: Send {
    /// The frame as an image that is drawn with `context`. `None` when the frame can not be
    /// drawn from the GPU (another device, no memory, a layout this source does not handle):
    /// the caller draws its pixels instead. The image is valid until [`Self::end_frame`].
    fn image(&mut self, context: &mut RecordingContext, frame: &GpuVideoFrame) -> Option<Image>;

    /// Everything handed out since the last call has been drawn (submitted), so the
    /// resources behind it can be used again.
    fn end_frame(&mut self);
}
