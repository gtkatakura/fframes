//! Decoded video frames drawn straight from the GPU (see `frame_export::vulkan_video`).
//!
//! Needs a Vulkan device whose driver encodes and decodes H.264 through Vulkan Video
//! (`h264_vulkan`); anywhere else (the lavapipe of CI, a driver without Vulkan Video) every
//! test prints why it is skipped and passes.

#![cfg(feature = "vulkan-video")]

use fframes::FFramesSyncedVideoFrame;
use fframes::media::{DecodeMode, FFmpegDecoder, set_gpu_frame_budget};
use fframes::{
    AudioMap, Color, Duration, EncoderOptions, FFramesContext, Frame, FrameConvertOptions,
    MediaDirectory, RenderOptions, ResizeVideoFrame, Svgr, SyncVideoFrameInput, Video,
    VideoEncoderInfo,
};
use fframes_skia_renderer::vulkan::SkiaVulkanCtx;
use fframes_skia_renderer::{
    SkiaFFramesRenderer, SkiaPipelineConcurrencyPolicy, SkiaPipelineConfig, gpu_video_frames_drawn,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

const WIDTH: usize = 640;
const HEIGHT: usize = 360;
const FPS: usize = 30;
const FRAMES: usize = 45;

/// Saturated bars with soft edges, a gradient and a moving disc: sharp chroma edges like the
/// ones of UI and screen footage, without the chroma that changes every other pixel (see
/// the module docs of `frame_export::vulkan_video` for what that looks like).
#[derive(Debug)]
struct Pattern;

impl Video for Pattern {
    const FPS: usize = FPS;
    const WIDTH: usize = WIDTH;
    const HEIGHT: usize = HEIGHT;
    const BACKGROUND_COLOR: Color = Color::BLACK;

    fn duration(&self) -> Duration<'_> {
        Duration::Frames(FRAMES)
    }

    fn audio(&self) -> AudioMap<'_> {
        AudioMap::none()
    }

    fn render_frame<'a>(&'a self, frame: Frame, _ctx: &FFramesContext<'a, '_>) -> Svgr<'a> {
        const COLORS: [&str; 6] = [
            "#e03030", "#30c040", "#3050e0", "#e0d020", "#20c0d0", "#c030c0",
        ];
        let stripes: Vec<Svgr> = (0..WIDTH / 16)
            .map(|i| {
                fframes::svgr!(
                    <rect x={i * 16} y="0" width="16" height={HEIGHT / 2} fill={COLORS[i % COLORS.len()]} />
                )
            })
            .collect();
        let x = 40.0 + frame.index as f32 * 12.0;
        fframes::svgr!(
            <svg xmlns="http://www.w3.org/2000/svg" width={Self::WIDTH} height={Self::HEIGHT}>
                <rect x="0" y="0" width={Self::WIDTH} height={Self::HEIGHT} fill="#303a48" />
                {stripes}
                <circle cx={x} cy="270" r="60" fill="#f0f0f0" />
                <circle cx={x + 30.0} cy="250" r="30" fill="#d06020" />
            </svg>
        )
    }
}

/// Shows the clip `clip.mp4` of the media folder full frame, and counts the frames that
/// arrive in the tree as decoded GPU frames.
#[derive(Debug, Default)]
struct Player {
    on_gpu: AtomicUsize,
    /// Ask for the frame at half the size, like `Clip::image_at_width`.
    resized: bool,
}

impl Video for Player {
    const FPS: usize = FPS;
    const WIDTH: usize = WIDTH;
    const HEIGHT: usize = HEIGHT;
    const BACKGROUND_COLOR: Color = Color::BLACK;

    fn duration(&self) -> Duration<'_> {
        Duration::Frames(FRAMES)
    }

    fn audio(&self) -> AudioMap<'_> {
        AudioMap::none()
    }

    fn render_frame<'a>(&'a self, frame: Frame, ctx: &FFramesContext<'a, '_>) -> Svgr<'a> {
        let Some(clip) =
            frame.get_synced_video_frame(ctx, "clip.mp4", &SyncVideoFrameInput::default())
        else {
            return Svgr::empty();
        };
        let image = if self.resized {
            let Some(image) = clip.into_resized_image(&FrameConvertOptions {
                resize: ResizeVideoFrame {
                    width: WIDTH as u32 / 2,
                    height: HEIGHT as u32 / 2,
                },
            }) else {
                return Svgr::empty();
            };
            image
        } else {
            clip.into_image()
        };
        if fframes::media::resolve_gpu_video_frame(&image.href()).is_some() {
            self.on_gpu.fetch_add(1, Ordering::Relaxed);
        }
        fframes::svgr!(
            <svg xmlns="http://www.w3.org/2000/svg" width={Self::WIDTH} height={Self::HEIGHT}>
                <image href={image.href()} x="0" y="0" width={Self::WIDTH} height={Self::HEIGHT} preserveAspectRatio="none" />
            </svg>
        )
    }
}

fn options<'a>(dir: &'a PathBuf, encoder: &'a str) -> RenderOptions<'a, 'a> {
    RenderOptions {
        logger: fframes::fframes_logger::FFramesLoggerVariant::Silent,
        tmp_files_directory: Some(dir),
        video_encoder_options: EncoderOptions {
            preferred_encoder: Some(encoder),
            // near lossless, so the two renders differ by what the clip's pixels took
            codec_params: Some(&[("qp", "10")]),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn skip(why: &str) {
    eprintln!("skipping: {why}");
}

/// The RGBA pixels of frame `index` of `video`, decoded the default way.
fn frame_pixels(video: &Path, index: usize) -> Vec<u8> {
    let mut decoder = unsafe { FFmpegDecoder::new(video, FPS, 1) }.unwrap();
    unsafe {
        assert!(
            decoder.decode_up_to(index as i64).unwrap(),
            "frame {index} is missing"
        );
        decoder
            .get_raw_frame()
            .convert_last_decoded_frame_into_svg_image(None)
            .unwrap()
            .data
            .to_vec()
    }
}

/// PSNR over the RGB channels of two RGBA images.
fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (sum, count) = a
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.as_chunks::<4>().0)
        .flat_map(|(a, b)| (0..3).map(move |c| f64::from(a[c]) - f64::from(b[c])))
        .fold((0.0, 0_u64), |(sum, count), d| (sum + d * d, count + 1));
    10.0 * (255.0_f64 * 255.0 / (sum / count as f64).max(1e-9)).log10()
}

/// The tests share the GPU, the frame budget and the counter of frames drawn from the GPU, and
/// creating Vulkan devices on several threads at once crashes some drivers: one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

struct Setup {
    dir: PathBuf,
    vulkan: SkiaVulkanCtx,
    /// Dropped last: the test is over when the device and the files are gone.
    _serial: MutexGuard<'static, ()>,
}

impl Drop for Setup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// The clip, rendered through Vulkan Video, or why this machine can not run the tests.
fn setup() -> Option<Setup> {
    let serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let Ok(vulkan) = SkiaVulkanCtx::new_shared_with_encoder(WIDTH, HEIGHT) else {
        skip("FFmpeg has no Vulkan device");
        return None;
    };
    let dir = std::env::temp_dir().join(format!("fframes-vulkan-decode-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("media")).unwrap();
    let clip = dir.join("media/clip.mp4");
    let chunks = dir.join("chunks");
    let options = options(&chunks, "h264_vulkan");

    let info = VideoEncoderInfo::for_output(
        &clip,
        (WIDTH as i32, HEIGHT as i32, FPS as i32),
        &options.video_encoder_options,
    )
    .unwrap();
    let backend = SkiaFFramesRenderer::new_vulkan(&vulkan, SkiaPipelineConfig::default()).unwrap();
    let hardware = info.name() == "h264_vulkan"
        && fframes::FFramesRenderBackend::negotiate_encoder_input(&backend, &info)
            .is_ok_and(|input| input.is_hardware());
    if !hardware {
        skip("the driver has no h264_vulkan encoder");
        return None;
    }
    fframes::render(&clip, &Pattern, backend, &options).unwrap();

    // Does the driver decode it with Vulkan Video, and keep the frame on the GPU?
    let decoder = unsafe { FFmpegDecoder::new_with_mode(&clip, FPS, 2, DecodeMode::GpuFrames) };
    let mut decoder = decoder.unwrap();
    let on_gpu = unsafe { decoder.decode_up_to(0) }.unwrap_or(false)
        && decoder.get_raw_frame().gpu_frame(None).is_some();
    drop(decoder);
    if !on_gpu {
        skip("the driver has no Vulkan Video decoder for H.264");
        return None;
    }
    Some(Setup {
        dir,
        vulkan,
        _serial: serial,
    })
}

fn render_player(
    setup: &Setup,
    name: &str,
    gpu_video_frames: bool,
    resized: bool,
) -> (PathBuf, usize, u64) {
    let output = setup.dir.join(name);
    let folder = MediaDirectory::read_folder(setup.dir.join("media")).unwrap();
    let media = folder.process_media_source().unwrap();
    let chunks = setup.dir.join(format!("{name}-chunks"));
    let mut options = options(&chunks, "h264_vulkan");
    options.media = Some(&media);

    let backend = SkiaFFramesRenderer::new_vulkan(
        &setup.vulkan,
        SkiaPipelineConfig {
            encoder_threads: 4,
            concurrency_policy: SkiaPipelineConcurrencyPolicy::Concurrency(2),
            ..Default::default()
        },
    )
    .unwrap()
    .gpu_video_frames(gpu_video_frames);

    let player = Player {
        resized,
        ..Player::default()
    };
    let drawn_before = gpu_video_frames_drawn();
    fframes::render(&output, &player, backend, &options).unwrap();
    (
        output,
        player.on_gpu.load(Ordering::Relaxed),
        gpu_video_frames_drawn() - drawn_before,
    )
}

#[test]
fn decoded_frames_drawn_from_the_gpu_match_the_cpu_conversion() {
    let Some(setup) = setup() else { return };

    let (on_cpu, kept, drawn) = render_player(&setup, "cpu.mp4", false, false);
    assert_eq!(
        (kept, drawn),
        (0, 0),
        "the option did not turn the GPU path off"
    );

    let (on_gpu, kept, drawn) = render_player(&setup, "gpu.mp4", true, false);
    // every frame arrives in the tree as a GPU frame and is drawn from there
    assert_eq!(kept, FRAMES, "the decoders downloaded frames");
    assert_eq!(
        drawn, FRAMES as u64,
        "frames were drawn from their pixels instead"
    );

    // The frames differ at sharp chroma edges only: Skia upsamples chroma bilinearly and
    // centered between the luma samples, swscale as the stream says (H.264: sited on the left
    // luma sample). 32 dB on these hard bars; the footage of the examples is around 43 dB.
    let mut worst = f64::MAX;
    for index in (0..FRAMES).step_by(7) {
        let gpu = frame_pixels(&on_gpu, index);
        let cpu = frame_pixels(&on_cpu, index);
        let psnr = psnr(&gpu, &cpu);
        eprintln!("frame {index}: {psnr:.1} dB");
        worst = worst.min(psnr);
    }
    assert!(
        worst > 30.0,
        "the GPU path differs from the CPU path: {worst:.1} dB"
    );
}

#[test]
fn frames_asked_for_at_another_size_are_drawn_at_that_size_from_the_gpu() {
    let Some(setup) = setup() else { return };

    let (on_cpu, kept, drawn) = render_player(&setup, "small-cpu.mp4", false, true);
    assert_eq!((kept, drawn), (0, 0));
    let (on_gpu, kept, drawn) = render_player(&setup, "small-gpu.mp4", true, true);
    assert_eq!((kept, drawn), (FRAMES, FRAMES as u64));

    // the same picture, drawn to the same place: compared over the whole frame
    let worst = (0..FRAMES)
        .step_by(11)
        .map(|index| psnr(&frame_pixels(&on_gpu, index), &frame_pixels(&on_cpu, index)))
        .fold(f64::MAX, f64::min);
    assert!(
        worst > 25.0,
        "the resized GPU frame differs from the CPU one: {worst:.1} dB"
    );
}

#[test]
fn a_frame_outlives_its_decoder_and_is_still_readable() {
    let Some(setup) = setup() else { return };
    let clip = setup.dir.join("media/clip.mp4");

    let mut decoder =
        unsafe { FFmpegDecoder::new_with_mode(&clip, FPS, 2, DecodeMode::GpuFrames) }.unwrap();
    assert!(unsafe { decoder.decode_up_to(5) }.unwrap());
    let frame = decoder
        .get_raw_frame()
        .gpu_frame(None)
        .expect("a frame on the GPU");
    drop(decoder);

    // the pool of the dropped decoder stays alive for the frame, libav waits for the GPU
    // before it frees the images
    let pixels = frame
        .pixels()
        .expect("the frame can be downloaded after its decoder is gone");
    let reference = frame_pixels(&clip, 5);
    assert!(psnr(&pixels.data, &reference) > 40.0);
}

#[test]
fn without_gpu_memory_for_frames_the_decoder_downloads_them() {
    let Some(setup) = setup() else { return };
    let clip = setup.dir.join("media/clip.mp4");

    // a budget with no room for two frames
    set_gpu_frame_budget(0);
    let mut decoder =
        unsafe { FFmpegDecoder::new_with_mode(&clip, FPS, 2, DecodeMode::GpuFrames) }.unwrap();
    set_gpu_frame_budget(512 * 1024 * 1024);
    assert!(unsafe { decoder.decode_up_to(3) }.unwrap());
    let frame = decoder.get_raw_frame();
    assert!(
        frame.gpu_frame(None).is_none(),
        "the decoder kept a frame without a budget"
    );
    // and the frame is there to read as always
    let image = unsafe { frame.convert_last_decoded_frame_into_svg_image(None) }.unwrap();
    assert_eq!(
        (image.width as usize, image.height as usize),
        (WIDTH, HEIGHT)
    );
}
