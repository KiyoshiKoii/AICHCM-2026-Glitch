import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

vi.mock('../api/apiClient.js', () => ({
  getFrameTimeline: vi.fn(async () => ({
    data: {
      before_frames: [{
        frame_id: 'L26_V001_f0001',
        thumbnail_url: '/media/thumbnails/L26_V001_f0001.jpg',
        frame_index: 0,
        timestamp_ms: 0,
        fps: 25,
      }],
      center_frame: {
        frame_id: 'L26_V001_f0002',
        thumbnail_url: '/media/thumbnails/L26_V001_f0002.jpg',
        frame_index: 21,
        timestamp_ms: 840,
        fps: 25,
      },
      after_frames: [{
        frame_id: 'L26_V001_f0003',
        thumbnail_url: '/media/thumbnails/L26_V001_f0003.jpg',
        frame_index: 78,
        timestamp_ms: 3120,
        fps: 25,
      }],
    },
  })),
}));

import VideoFrameModal, {
  interpolateTimelinePosition,
  interpolateTimelineX,
} from './VideoFrameModal.jsx';

const result = {
  frame_id: 'L26_V001_f0002',
  video_name: 'L26_V001',
  frame_index: 21,
  thumbnail_url: '/media/thumbnails/L26_V001_f0002.jpg',
};

describe('VideoFrameModal', () => {
  it('interpolates the exact native frame instead of snapping to a keyframe', () => {
    const position = interpolateTimelinePosition([
      {
        x: 100,
        frame: { timestamp_ms: 840, frame_index: 21, fps: 25 },
      },
      {
        x: 200,
        frame: { timestamp_ms: 3120, frame_index: 78, fps: 25 },
      },
    ], 150);

    expect(position.timestampMs).toBe(1980);
    expect(position.frameIndex).toBe(50);
    expect(position.fps).toBe(25);
  });

  it('maps a manually seeked video timestamp back onto the filmstrip', () => {
    const x = interpolateTimelineX([
      { x: 100, frame: { timestamp_ms: 840 } },
      { x: 200, frame: { timestamp_ms: 3120 } },
    ], 1980);

    expect(x).toBe(150);
  });

  it('loads a filmstrip and seeks the video when a frame is selected', async () => {
    const user = userEvent.setup();
    const pause = vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
    render(<VideoFrameModal result={result} onClose={vi.fn()} />);

    const video = screen.getByLabelText('Phát L26_V001');
    expect(video.autoplay).toBe(false);
    Object.defineProperty(video, 'readyState', { configurable: true, value: 1 });
    fireEvent.loadedMetadata(video);

    const target = await screen.findByRole('button', { name: 'Đi tới frame 78' });
    await user.click(target);

    expect(video.currentTime).toBeCloseTo(3.12);
    expect(pause).toHaveBeenCalled();
    expect(screen.getByText('78', { selector: '.video-frame-readout strong' })).toBeInTheDocument();
    pause.mockRestore();
  });

  it('pauses playback as soon as the filmstrip starts dragging', async () => {
    const pause = vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
    const { container } = render(<VideoFrameModal result={result} onClose={vi.fn()} />);
    await screen.findByRole('button', { name: 'Đi tới frame 78' });

    fireEvent.pointerDown(container.querySelector('.video-filmstrip'), {
      pointerId: 1,
      clientX: 200,
    });

    expect(pause).toHaveBeenCalled();
    pause.mockRestore();
  });

  it('opens an ASR result at the exact transcript timestamp', async () => {
    const pause = vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
    render(
      <VideoFrameModal
        result={{ ...result, metadata: { seek_timestamp_ms: 2345 } }}
        onClose={vi.fn()}
      />,
    );

    const video = document.querySelector('video');
    Object.defineProperty(video, 'readyState', { configurable: true, value: 1 });
    fireEvent.loadedMetadata(video);

    await waitFor(() => expect(video.currentTime).toBeCloseTo(2.345));
    expect(pause).toHaveBeenCalled();
    pause.mockRestore();
  });

  it('closes when Escape is pressed', async () => {
    const onClose = vi.fn();
    render(<VideoFrameModal result={result} onClose={onClose} />);

    fireEvent.keyDown(window, { key: 'Escape' });

    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
  });
});
