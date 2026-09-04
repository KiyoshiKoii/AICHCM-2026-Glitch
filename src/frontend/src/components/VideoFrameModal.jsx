import { useEffect, useMemo, useRef, useState } from 'react';
import { getFrameTimeline } from '../api/apiClient.js';

function videoIdFromResult(result) {
  const explicit = result?.video_name || result?.video_id;
  if (explicit) return String(explicit).replace(/\.mp4$/i, '');
  return result?.frame_id?.match(/^(.*)_f\d+$/)?.[1] || '';
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return '00:00.000';
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${String(minutes).padStart(2, '0')}:${remainder.toFixed(3).padStart(6, '0')}`;
}

function nativeFrame(result) {
  const value = result?.frame_index ?? result?.native_frame_idx;
  return Number.isFinite(Number(value)) ? Number(value) : null;
}

export function interpolateTimelinePosition(points, playheadX) {
  if (!points.length) return null;
  if (playheadX <= points[0].x) return {
    timestampMs: Number(points[0].frame.timestamp_ms),
    frameIndex: nativeFrame(points[0].frame),
    fps: Number(points[0].frame.fps),
  };
  const last = points.at(-1);
  if (playheadX >= last.x) return {
    timestampMs: Number(last.frame.timestamp_ms),
    frameIndex: nativeFrame(last.frame),
    fps: Number(last.frame.fps),
  };

  const rightIndex = points.findIndex((point) => point.x >= playheadX);
  const left = points[rightIndex - 1];
  const right = points[rightIndex];
  const ratio = (playheadX - left.x) / Math.max(right.x - left.x, 1);
  const leftTimestamp = Number(left.frame.timestamp_ms);
  const rightTimestamp = Number(right.frame.timestamp_ms);
  const leftFrame = nativeFrame(left.frame);
  const rightFrame = nativeFrame(right.frame);
  const fps = Number(left.frame.fps || right.frame.fps);
  const timestampMs = leftTimestamp + ((rightTimestamp - leftTimestamp) * ratio);
  const frameIndex = leftFrame !== null && rightFrame !== null
    ? Math.round(leftFrame + ((rightFrame - leftFrame) * ratio))
    : Number.isFinite(fps) && fps > 0
      ? Math.round(timestampMs * fps / 1_000)
      : null;
  return { timestampMs, frameIndex, fps };
}

export function interpolateTimelineX(points, timestampMs) {
  if (!points.length) return null;
  const firstTimestamp = Number(points[0].frame.timestamp_ms);
  if (timestampMs <= firstTimestamp) return points[0].x;
  const last = points.at(-1);
  const lastTimestamp = Number(last.frame.timestamp_ms);
  if (timestampMs >= lastTimestamp) return last.x;

  const rightIndex = points.findIndex((point) => Number(point.frame.timestamp_ms) >= timestampMs);
  const left = points[rightIndex - 1];
  const right = points[rightIndex];
  const leftTimestamp = Number(left.frame.timestamp_ms);
  const rightTimestamp = Number(right.frame.timestamp_ms);
  const ratio = (timestampMs - leftTimestamp) / Math.max(rightTimestamp - leftTimestamp, 1);
  return left.x + ((right.x - left.x) * ratio);
}

function VideoFrameModal({ result, onClose }) {
  const videoRef = useRef(null);
  const filmstripRef = useRef(null);
  const dragRef = useRef({
    active: false,
    moved: false,
    pointerId: null,
    startX: 0,
    scrollLeft: 0,
  });
  const initialSeekDoneRef = useRef(false);
  const [context, setContext] = useState(null);
  const [contextError, setContextError] = useState('');
  const [activeFrame, setActiveFrame] = useState(result);
  const [playbackFrame, setPlaybackFrame] = useState(nativeFrame(result));
  const [playbackTime, setPlaybackTime] = useState(0);
  const requestedNativeFrame = Number(result?.metadata?.seek_frame_index);
  const hasRequestedNativeFrame = Number.isInteger(requestedNativeFrame)
    && requestedNativeFrame >= 0;

  const videoId = videoIdFromResult(result);
  const frames = useMemo(() => {
    if (!context) return [];
    return [
      ...(context.before_frames || []),
      context.center_frame,
      ...(context.after_frames || []),
    ].filter(Boolean);
  }, [context]);

  const fps = Number(activeFrame?.fps || context?.center_frame?.fps || result?.metadata?.fps);

  useEffect(() => {
    initialSeekDoneRef.current = false;
    setContext(null);
    setContextError('');
    setActiveFrame(result);
    setPlaybackFrame(nativeFrame(result));

    if (!result?.frame_id) return undefined;
    let cancelled = false;
    getFrameTimeline(result.frame_id)
      .then((response) => {
        if (cancelled) return;
        const nextContext = response?.data || null;
        setContext(nextContext);
        if (nextContext?.center_frame) {
          setActiveFrame(hasRequestedNativeFrame
            ? {
                ...result,
                ...nextContext.center_frame,
                frame_id: null,
                frame_index: requestedNativeFrame,
              }
            : { ...result, ...nextContext.center_frame });
        }
      })
      .catch(() => {
        if (!cancelled) setContextError('Không tải được các keyframe lân cận.');
      });
    return () => {
      cancelled = true;
    };
  }, [result]);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') onClose?.();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const seekToFrame = (frame) => {
    const timestampMs = Number(frame?.timestamp_ms ?? result?.metadata?.timestamp_ms);
    const frameIndex = nativeFrame(frame) ?? nativeFrame(result);
    videoRef.current?.pause();
    setActiveFrame(frame);
    if (frameIndex !== null) setPlaybackFrame(frameIndex);
    if (videoRef.current && Number.isFinite(timestampMs)) {
      videoRef.current.currentTime = timestampMs / 1_000;
      setPlaybackTime(timestampMs / 1_000);
    }
  };

  const seekInitialFrame = () => {
    if (initialSeekDoneRef.current) return;
    const exactSeekTimestamp = Number(result?.metadata?.seek_timestamp_ms);
    const timelineFps = Number(context?.center_frame?.fps || result?.metadata?.fps);
    const requestedFrameTimestamp = hasRequestedNativeFrame
      && Number.isFinite(timelineFps)
      && timelineFps > 0
      ? requestedNativeFrame * 1_000 / timelineFps
      : Number.NaN;
    const initialFrame = {
      ...result,
      ...(context?.center_frame || {}),
      ...(hasRequestedNativeFrame
        ? { frame_id: null, frame_index: requestedNativeFrame, fps: timelineFps }
        : {}),
      timestamp_ms: Number.isFinite(requestedFrameTimestamp)
        ? requestedFrameTimestamp
        : Number.isFinite(exactSeekTimestamp)
          ? exactSeekTimestamp
          : context?.center_frame?.timestamp_ms ?? result?.metadata?.timestamp_ms,
    };
    const timestampMs = Number(initialFrame.timestamp_ms);
    if (!Number.isFinite(timestampMs)) return;
    initialSeekDoneRef.current = true;
    seekToFrame(initialFrame);
  };

  useEffect(() => {
    if (videoRef.current?.readyState >= 1) seekInitialFrame();
  }, [context]);

  useEffect(() => {
    const track = filmstripRef.current;
    const selected = track?.querySelector('.video-filmstrip-frame.selected');
    if (!track || !selected) return;
    const left = selected.offsetLeft - (track.clientWidth / 2) + (selected.clientWidth / 2);
    track.scrollTo?.({ left, behavior: 'smooth' });
  }, [activeFrame?.frame_id, frames.length]);

  const timelinePoints = () => {
    const track = filmstripRef.current;
    if (!track || !frames.length) return [];
    return [...track.querySelectorAll('.video-filmstrip-frame')]
      .map((element) => ({
        x: element.offsetLeft + (element.offsetWidth / 2),
        frame: frames.find((frame) => frame.frame_id === element.dataset.frameId),
      }))
      .filter((point) => point.frame && Number.isFinite(Number(point.frame.timestamp_ms)));
  };

  const syncFilmstripToTime = (seconds) => {
    if (dragRef.current.active) return;
    const track = filmstripRef.current;
    const timestampMs = seconds * 1_000;
    const x = interpolateTimelineX(timelinePoints(), timestampMs);
    if (!track || x === null) return;
    track.scrollLeft = Math.max(0, x - (track.clientWidth / 2));
  };

  const updatePlaybackPosition = () => {
    const seconds = videoRef.current?.currentTime || 0;
    setPlaybackTime(seconds);
    if (Number.isFinite(fps) && fps > 0) {
      setPlaybackFrame(Math.round(seconds * fps));
    }
    syncFilmstripToTime(seconds);
  };

  const handleVideoSeeking = () => {
    videoRef.current?.pause();
    updatePlaybackPosition();
    const seconds = videoRef.current?.currentTime || 0;
    const frameIndex = Number.isFinite(fps) && fps > 0
      ? Math.round(seconds * fps)
      : null;
    setActiveFrame({
      frame_id: null,
      timestamp_ms: seconds * 1_000,
      frame_index: frameIndex,
      fps,
    });
  };

  const startDragging = (event) => {
    const track = filmstripRef.current;
    if (!track) return;
    event.preventDefault();
    videoRef.current?.pause();
    dragRef.current = {
      active: true,
      moved: false,
      pointerId: event.pointerId,
      startX: event.clientX,
      scrollLeft: track.scrollLeft,
    };
    track.classList.add('dragging');
    track.setPointerCapture?.(event.pointerId);
  };

  const dragFilmstrip = (event) => {
    if (!dragRef.current.active || !filmstripRef.current) return;
    event.preventDefault();
    if (Math.abs(event.clientX - dragRef.current.startX) > 4) {
      dragRef.current.moved = true;
    }
    filmstripRef.current.scrollLeft = (
      dragRef.current.scrollLeft - (event.clientX - dragRef.current.startX)
    );
    if (dragRef.current.moved) seekToPlayheadPosition();
  };

  const timelinePositionAtPlayhead = () => {
    const track = filmstripRef.current;
    if (!track || !frames.length) return null;
    const playheadX = track.scrollLeft + (track.clientWidth / 2);
    return interpolateTimelinePosition(timelinePoints(), playheadX);
  };

  const seekToPlayheadPosition = () => {
    const position = timelinePositionAtPlayhead();
    if (!position || !Number.isFinite(position.timestampMs)) return;
    videoRef.current?.pause();
    if (videoRef.current) videoRef.current.currentTime = position.timestampMs / 1_000;
    setPlaybackTime(position.timestampMs / 1_000);
    setPlaybackFrame(position.frameIndex);
    setActiveFrame({
      frame_id: null,
      timestamp_ms: position.timestampMs,
      frame_index: position.frameIndex,
      fps: position.fps,
    });
  };

  const stopDragging = () => {
    if (!dragRef.current.active) return;
    const moved = dragRef.current.moved;
    dragRef.current.active = false;
    filmstripRef.current?.classList.remove('dragging');
    if (moved) {
      seekToPlayheadPosition();
    }
  };

  const selectFilmstripFrame = (frame) => {
    if (dragRef.current.moved) {
      dragRef.current.moved = false;
      return;
    }
    seekToFrame(frame);
  };

  if (!result) return null;

  return (
    <div className="video-frame-overlay" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose?.();
    }}>
      <section className="video-frame-modal" role="dialog" aria-modal="true" aria-label={`Video ${videoId}`}>
        <header className="video-frame-header">
          <div>
            <strong>{videoId}</strong>
            <span>{hasRequestedNativeFrame ? `Frame ${requestedNativeFrame}` : result.frame_id}</span>
          </div>
          <button type="button" className="video-frame-close" onClick={onClose} aria-label="Đóng video">
            ×
          </button>
        </header>

        <div className="video-frame-stage">
          <video
            ref={videoRef}
            controls
            preload="metadata"
            src={`/media/videos/${videoId}.mp4`}
            onLoadedMetadata={seekInitialFrame}
            onTimeUpdate={updatePlaybackPosition}
            onSeeking={handleVideoSeeking}
            aria-label={`Phát ${videoId}`}
          />
          <div className="video-frame-readout" aria-live="polite">
            <span>Frame</span>
            <strong>{playbackFrame ?? '—'}</strong>
            <small>{formatTime(playbackTime)}</small>
          </div>
        </div>

        <div className="video-filmstrip-shell">
          <button
            type="button"
            className="video-filmstrip-arrow"
            aria-label="Cuộn timeline sang trái"
            onClick={() => filmstripRef.current?.scrollBy({ left: -320, behavior: 'smooth' })}
          >
            ‹
          </button>
          <div
            className="video-filmstrip"
            ref={filmstripRef}
            onPointerDown={startDragging}
            onPointerMove={dragFilmstrip}
            onPointerUp={stopDragging}
            onPointerCancel={stopDragging}
            onLostPointerCapture={stopDragging}
          >
            <div className="video-filmstrip-track">
              {!frames.length && !contextError && (
                <span className="video-filmstrip-loading">Đang tải timeline…</span>
              )}
              {frames.map((frame) => {
                const selected = activeFrame?.frame_id === frame.frame_id;
                return (
                  <button
                    type="button"
                    key={frame.frame_id}
                    data-frame-id={frame.frame_id}
                    className={selected ? 'video-filmstrip-frame selected' : 'video-filmstrip-frame'}
                    onClick={() => selectFilmstripFrame(frame)}
                    aria-label={`Đi tới frame ${frame.frame_index ?? frame.frame_id}`}
                  >
                    <img
                      src={frame.thumbnail_url}
                      alt=""
                      draggable="false"
                      loading="lazy"
                    />
                    <span>{frame.frame_index ?? frame.frame_id.split('_f').at(-1)}</span>
                  </button>
                );
              })}
            </div>
          </div>
          <div className="video-filmstrip-playhead" aria-hidden="true" />
          <button
            type="button"
            className="video-filmstrip-arrow"
            aria-label="Cuộn timeline sang phải"
            onClick={() => filmstripRef.current?.scrollBy({ left: 320, behavior: 'smooth' })}
          >
            ›
          </button>
        </div>
        {contextError && <p className="video-filmstrip-error">{contextError}</p>}
        <p className="video-filmstrip-hint">Kéo timeline qua trái/phải hoặc bấm thumbnail để nhảy tới frame.</p>
      </section>
    </div>
  );
}

export default VideoFrameModal;
