use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use super::FFramesLogger;
use super::concatenator::AvPacketAutoFree;
use super::encoder::{Encoder, EncoderOutput};
use super::frame_export::{EncoderInput, RgbaFrameConverter, VideoEncoderInfo, VideoFrame};
use super::renderer_error::{RenderEncodingError, RenderEncodingResult};
use super::scheduler::FrameClaim;
use crate::RenderOptions;

/// How long a segment waits for a hardware encoder session before the render fails with the
/// encoder's error. Sessions are released within milliseconds when a segment ends, so this is
/// far longer than a healthy render ever waits.
const SESSION_WAIT: Duration = Duration::from_secs(60);

/// How long the first segment of a render waits for a session after the probe closed its own:
/// the driver may release them a moment later than the encoders are dropped.
const PROBE_RELEASE_GRACE: Duration = Duration::from_secs(2);

/// Pause between two attempts to open an encoder while other segments hold sessions. Waiting
/// for a segment to end is woken up sooner; this covers a driver that releases a session a
/// moment after the encoder was closed.
const SESSION_RETRY: Duration = Duration::from_millis(100);

/// The encoder sessions of the open segments of a writer. Hardware encoders can refuse a new
/// session (NVIDIA's consumer GPUs allow a handful of concurrent encodes; the limit depends on
/// the GPU and the driver), and there is no portable way to ask for the limit.
#[derive(Default)]
struct Sessions {
    open: Mutex<usize>,
    released: Condvar,
}

impl Sessions {
    /// Calls `open`. When it fails while other segments hold sessions the device may have
    /// refused one, so it is called again as they end, for up to `wait`. It fails at once
    /// when no other session is open (nothing to wait for, the encoder is broken) or `wait`
    /// is zero (software encoders), unless `alone` allows a short retry for sessions that
    /// are still being released.
    fn open_with_retry<T, E>(
        this: &Arc<Self>,
        (wait, alone): (Duration, Duration),
        retry: Duration,
        mut open: impl FnMut() -> Result<T, E>,
    ) -> Result<(T, SessionPermit), E> {
        let started = Instant::now();
        loop {
            match open() {
                Ok(opened) => {
                    *this
                        .open
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
                    return Ok((opened, SessionPermit(Arc::clone(this))));
                }
                Err(err) => {
                    let open = this
                        .open
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let limit = if *open == 0 { alone } else { wait };
                    if started.elapsed() >= limit {
                        return Err(err);
                    }
                    // a segment that ends wakes this up; the timeout covers a slow release
                    drop(
                        this.released
                            .wait_timeout(open, retry)
                            .unwrap_or_else(std::sync::PoisonError::into_inner),
                    );
                }
            }
        }
    }
}

/// One open encoder session, given back when the segment ends.
struct SessionPermit(Arc<Sessions>);

impl Drop for SessionPermit {
    fn drop(&mut self) {
        *self
            .0
            .open
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) -= 1;
        self.0.released.notify_all();
    }
}

struct Segment {
    encoder: Encoder,
    /// Declared after the encoder: the session is released once the encoder is closed.
    _session: SessionPermit,
    packet: AvPacketAutoFree,
    next_frame: usize,
    end: Option<usize>,
    /// Frames that arrived before the frames preceding them.
    pending: BTreeMap<usize, VideoFrame>,
    /// Only for frames submitted as RGBA pixels.
    converter: Option<RgbaFrameConverter>,
}

type SegmentSlot = Arc<Mutex<Option<Segment>>>;

/// Encodes the frames handed out by a [`super::FrameScheduler`] into one intermediate
/// file per segment. Frames of a segment may arrive out of order from several threads;
/// they are encoded in order and the file is finalized as soon as its last frame is in.
#[doc(hidden)]
pub struct SegmentWriter<'a, 'o, 'm> {
    directory: &'a Path,
    extension: String,
    width: i32,
    height: i32,
    fps: i32,
    render_options: &'a RenderOptions<'o, 'm>,
    logger: &'a Arc<dyn FFramesLogger>,
    input: EncoderInput,
    /// The encoder is dedicated hardware, which can limit its concurrent sessions.
    hardware_encoder: bool,
    /// The sessions were probed, see [`Self::probe_encoder_sessions`].
    probed: AtomicBool,
    sessions: Arc<Sessions>,
    open: Mutex<HashMap<usize, SegmentSlot>>,
    finished: Mutex<Vec<(usize, PathBuf)>>,
}

impl<'a, 'o, 'm> SegmentWriter<'a, 'o, 'm> {
    /// A writer whose encoders take the pixel format requested in `render_options`.
    pub fn new(
        directory: &'a Path,
        extension: &str,
        (width, height, fps): (i32, i32, i32),
        render_options: &'a RenderOptions<'o, 'm>,
        logger: &'a Arc<dyn FFramesLogger>,
    ) -> Self {
        Self {
            directory,
            extension: extension.to_owned(),
            width,
            height,
            fps,
            render_options,
            logger,
            input: EncoderInput::for_options(&render_options.video_encoder_options),
            hardware_encoder: false,
            probed: AtomicBool::new(false),
            sessions: Arc::default(),
            open: Mutex::new(HashMap::new()),
            finished: Mutex::new(Vec::new()),
        }
    }

    /// Opens the encoders for the frames a backend negotiated, see [`Self::encoder_info`].
    pub fn with_encoder_input(mut self, input: EncoderInput) -> Self {
        self.input = input;
        self
    }

    /// The encoders of the segments run on dedicated hardware (`VideoEncoderInfo::is_hardware`).
    /// A segment whose encoder is refused waits for the other segments to end and tries again.
    /// Software encoders are opened as they always were.
    pub fn with_hardware_encoder(mut self, hardware_encoder: bool) -> Self {
        self.hardware_encoder = hardware_encoder;
        self
    }

    /// How many of `wanted` hardware encoders the device opens at the same time, found by
    /// opening them (and closing them again). The encoder refused a session when fewer
    /// opened; a render should run [`concurrent_segments`] segments then.
    pub fn probe_encoder_sessions(&self, wanted: usize) -> usize {
        self.probed.store(true, Ordering::Relaxed);
        let probes: Vec<_> = (0..wanted)
            .map_while(|index| {
                let path = self
                    .directory
                    .join(format!("probe-{index:010}.{}", self.extension));
                let encoder = self.open_encoder(&path).ok();
                encoder.map(|encoder| (encoder, path))
            })
            .collect();
        let opened = probes.len();
        for (encoder, path) in probes {
            drop(encoder);
            let _ = std::fs::remove_file(path);
        }
        opened
    }

    /// The video encoder the segments will be encoded with, to negotiate its input
    /// (`FFramesRenderBackend::negotiate_encoder_input`).
    pub fn encoder_info(&self) -> RenderEncodingResult<VideoEncoderInfo<'a>> {
        VideoEncoderInfo::for_output(
            &self.segment_path(0),
            (self.width, self.height, self.fps),
            &self.render_options.video_encoder_options,
        )
    }

    /// What every frame given to [`Self::submit_frame`] has to be.
    pub fn encoder_input(&self) -> &EncoderInput {
        &self.input
    }

    fn segment_path(&self, segment: usize) -> PathBuf {
        self.directory
            .join(format!("{segment:010}.{}", self.extension))
    }

    /// Encodes a claimed frame.
    pub fn submit_frame(&self, claim: FrameClaim, frame: VideoFrame) -> RenderEncodingResult<()> {
        self.submit_with(claim, |_| Ok(frame))
    }

    /// Encodes the premultiplied RGBA pixels of a claimed frame, converting them on the CPU.
    pub fn submit(&self, claim: FrameClaim, rgba: &[u8]) -> RenderEncodingResult<()> {
        self.submit_with(claim, |segment| self.convert(segment, rgba))
    }

    /// Like [`Self::submit`] for callers that recycle their buffers. Returns the buffers
    /// that are not needed anymore.
    pub fn submit_owned(
        &self,
        claim: FrameClaim,
        rgba: Vec<u8>,
    ) -> RenderEncodingResult<Vec<Vec<u8>>> {
        self.submit(claim, &rgba)?;
        Ok(vec![rgba])
    }

    fn convert(&self, segment: &mut Segment, rgba: &[u8]) -> RenderEncodingResult<VideoFrame> {
        if segment.converter.is_none() {
            segment.converter = Some(RgbaFrameConverter::for_input(
                &self.input,
                self.width,
                self.height,
            )?);
        }

        segment
            .converter
            .as_mut()
            .expect("the converter was just created")
            .convert(rgba)
    }

    fn submit_with(
        &self,
        claim: FrameClaim,
        frame: impl FnOnce(&mut Segment) -> RenderEncodingResult<VideoFrame>,
    ) -> RenderEncodingResult<()> {
        let slot = self
            .open
            .lock()
            .unwrap()
            .entry(claim.segment)
            .or_default()
            .clone();

        let mut guard = slot.lock().unwrap();
        if guard.is_none() {
            let (encoder, session) =
                self.open_segment_encoder(&self.segment_path(claim.segment))?;
            *guard = Some(Segment {
                encoder,
                _session: session,
                packet: AvPacketAutoFree::new(),
                next_frame: claim.segment,
                end: None,
                pending: BTreeMap::new(),
                converter: None,
            });
        }

        let segment = guard.as_mut().unwrap();
        if claim.last_in_segment {
            segment.end = Some(claim.frame + 1);
        }

        let frame = frame(segment)?;
        if claim.frame == segment.next_frame {
            Self::encode(segment, claim.frame, &frame)?;
            // the encoder holds its own reference for as long as it needs the pixels
            drop(frame);
            while let Some(frame) = segment.pending.remove(&segment.next_frame) {
                Self::encode(segment, segment.next_frame, &frame)?;
            }
        } else {
            segment.pending.insert(claim.frame, frame);
        }

        if segment.end == Some(segment.next_frame) {
            let segment = guard.take().unwrap();
            unsafe {
                segment
                    .encoder
                    .flush_stream(&segment.encoder.video_stream)?;
            }
            // dropping the encoder writes the trailer
            drop(segment);

            self.open.lock().unwrap().remove(&claim.segment);
            self.finished
                .lock()
                .unwrap()
                .push((claim.segment, self.segment_path(claim.segment)));
        }

        Ok(())
    }

    fn open_encoder(&self, path: &Path) -> RenderEncodingResult<Encoder> {
        unsafe {
            Encoder::new_with_input(
                EncoderOutput::IntermediateChunk,
                self.width,
                self.height,
                self.fps,
                path,
                self.render_options,
                &self.input,
                self.logger,
            )
        }
    }

    /// Opens the encoder of a segment, see [`Sessions::open_with_retry`].
    fn open_segment_encoder(&self, path: &Path) -> RenderEncodingResult<(Encoder, SessionPermit)> {
        let wait = if !self.hardware_encoder {
            (Duration::ZERO, Duration::ZERO)
        } else if self.probed.load(Ordering::Relaxed) {
            (SESSION_WAIT, PROBE_RELEASE_GRACE)
        } else {
            (SESSION_WAIT, Duration::ZERO)
        };
        Sessions::open_with_retry(&self.sessions, wait, SESSION_RETRY, || {
            self.open_encoder(path)
        })
    }

    fn encode(segment: &mut Segment, index: usize, frame: &VideoFrame) -> RenderEncodingResult<()> {
        unsafe {
            // frame indexes are used as pts, av_packet_rescale_ts converts them into the
            // stream time base
            segment
                .encoder
                .send_video_frame(frame, index as i64, segment.packet.get())?;
        }
        segment.next_frame = index + 1;
        Ok(())
    }

    /// The finished segment files in timeline order.
    pub fn finish(self) -> RenderEncodingResult<Vec<PathBuf>> {
        if !self.open.lock().unwrap().is_empty() {
            return Err(RenderEncodingError::Internal(
                "some video segments did not receive all of their frames".to_owned(),
            ));
        }

        let mut finished = self.finished.into_inner().unwrap();
        finished.sort_by_key(|(segment, _)| *segment);
        Ok(finished.into_iter().map(|(_, path)| path).collect())
    }
}

/// How many segments a render runs at once on a device that opened `opened` of the `2 * wanted`
/// encoders of [`SegmentWriter::probe_encoder_sessions`]. A segment that ends keeps its
/// encoder open until its last frames are encoded, while the next segment of the same
/// generator already starts, so a render of `n` segments needs up to `2 * n` sessions. A
/// device that opened all of them is not limited. One that stopped earlier is, and the render
/// runs half as many segments as it opened, which keeps every session it needs.
#[doc(hidden)]
pub fn concurrent_segments(wanted: usize, opened: usize) -> usize {
    if opened >= 2 * wanted {
        wanted
    } else {
        (opened / 2).clamp(1, wanted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_WAIT: (Duration, Duration) = (Duration::ZERO, Duration::ZERO);

    #[test]
    fn a_device_that_never_refuses_keeps_all_segments() {
        assert_eq!(concurrent_segments(16, 32), 16);
        assert_eq!(concurrent_segments(1, 2), 1);
    }

    #[test]
    fn a_device_that_refuses_runs_half_as_many_segments_as_it_opened_sessions() {
        // NVIDIA's consumer GPUs open 8: 4 segments, each with a successor starting
        assert_eq!(concurrent_segments(16, 8), 4);
        assert_eq!(concurrent_segments(16, 31), 15);
        assert_eq!(concurrent_segments(4, 7), 3);
        assert_eq!(concurrent_segments(16, 2), 1);
        // refused at once: the render reports the encoder's error from one segment
        assert_eq!(concurrent_segments(16, 1), 1);
        assert_eq!(concurrent_segments(16, 0), 1);
    }

    #[test]
    fn a_refused_encoder_is_retried_as_other_sessions_end() {
        let sessions = Arc::<Sessions>::default();
        let ((), held) =
            Sessions::open_with_retry(&sessions, NO_WAIT, Duration::ZERO, || Ok::<_, ()>(()))
                .unwrap();

        let mut refusals = 0;
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            drop(held);
        });
        let opened = Sessions::open_with_retry(
            &sessions,
            (Duration::from_secs(10), Duration::ZERO),
            Duration::from_millis(5),
            || {
                // refused while the other session is open
                if *sessions.open.lock().unwrap() > 0 {
                    refusals += 1;
                    Err("refused")
                } else {
                    Ok(())
                }
            },
        );
        releaser.join().unwrap();
        assert!(opened.is_ok());
        assert!(refusals > 0, "the first attempts must have been refused");
    }

    #[test]
    fn a_refusal_without_other_sessions_or_for_software_fails_at_once() {
        let sessions = Arc::<Sessions>::default();
        // nothing else holds a session: nothing to wait for
        let alone = Sessions::open_with_retry(
            &sessions,
            (Duration::from_secs(10), Duration::ZERO),
            Duration::from_secs(10),
            || Err::<(), _>("refused"),
        );
        assert_eq!(alone.err(), Some("refused"));

        let ((), _held) =
            Sessions::open_with_retry(&sessions, NO_WAIT, Duration::ZERO, || Ok::<_, &str>(()))
                .unwrap();
        // software encoders pass a zero wait, whatever else is open
        let software =
            Sessions::open_with_retry(&sessions, NO_WAIT, Duration::from_secs(10), || {
                Err::<(), _>("failed")
            });
        assert_eq!(software.err(), Some("failed"));
    }

    #[test]
    fn the_first_segment_after_a_probe_retries_while_the_probe_sessions_are_released() {
        let sessions = Arc::<Sessions>::default();
        let mut refusals = 2;
        let opened = Sessions::open_with_retry(
            &sessions,
            (Duration::from_secs(10), Duration::from_secs(5)),
            Duration::from_millis(1),
            || {
                if refusals > 0 {
                    refusals -= 1;
                    Err("still held by the probe")
                } else {
                    Ok(())
                }
            },
        );
        assert!(opened.is_ok());
        assert_eq!(refusals, 0);
    }

    #[test]
    fn the_wait_is_bounded() {
        let sessions = Arc::<Sessions>::default();
        let ((), _held) =
            Sessions::open_with_retry(&sessions, NO_WAIT, Duration::ZERO, || Ok::<_, &str>(()))
                .unwrap();
        let started = Instant::now();
        let stuck = Sessions::open_with_retry(
            &sessions,
            (Duration::from_millis(60), Duration::ZERO),
            Duration::from_millis(10),
            || Err::<(), _>("refused"),
        );
        assert_eq!(stuck.err(), Some("refused"));
        assert!(started.elapsed() < Duration::from_secs(5));
    }
}
