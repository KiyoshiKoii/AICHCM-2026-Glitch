import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ResultGrid from './ResultGrid.jsx';

const makeResult = (i, overrides = {}) => ({
  frame_id: `vid01_f${i}`,
  video_name: 'vid01.mp4',
  frame_index: i,
  score: 0.9,
  thumbnail_url: `/media/thumbnails/vid01_f${i}.jpg`,
  metadata: { camera_id: 'cam_01', timestamp: '00:00:10' },
  ...overrides,
});

describe('ResultGrid', () => {
  it('renders an image card for each result', () => {
    const results = [makeResult(1), makeResult(2), makeResult(3)];
    render(<ResultGrid results={results} />);
    expect(screen.getAllByRole('img')).toHaveLength(3);
  });

  it('truncates to the top 15 results when more are provided', () => {
    const results = Array.from({ length: 35 }, (_, i) => makeResult(i));
    render(<ResultGrid results={results} />);
    expect(screen.getAllByRole('img')).toHaveLength(15);
  });

  it('skips entries missing the must-have frame_id or thumbnail_url fields', () => {
    const results = [
      makeResult(1),
      makeResult(2, { frame_id: undefined }),
      makeResult(3, { thumbnail_url: undefined }),
    ];
    render(<ResultGrid results={results} />);
    expect(screen.getAllByRole('img')).toHaveLength(1);
  });

  it('renders an empty grid without crashing when results is missing', () => {
    const { container } = render(<ResultGrid />);
    expect(screen.queryAllByRole('img')).toHaveLength(0);
    expect(container.querySelector('.result-grid')).toBeInTheDocument();
  });

  it('passes the complete result when a thumbnail is clicked', async () => {
    const user = userEvent.setup();
    const onCardClick = vi.fn();
    const result = makeResult(12);
    render(<ResultGrid results={[result]} onCardClick={onCardClick} />);

    await user.click(screen.getByRole('img'));

    expect(onCardClick).toHaveBeenCalledWith(result);
  });

  it('keeps separate ASR passages that share one thumbnail keyframe', () => {
    const first = makeResult(1, {
      metadata: { asr_id: 'asr-1', transcript: 'đoạn thoại thứ nhất' },
    });
    const second = makeResult(1, {
      metadata: { asr_id: 'asr-2', transcript: 'đoạn thoại thứ hai' },
    });

    render(<ResultGrid results={[first, second]} />);

    expect(screen.getAllByRole('img')).toHaveLength(2);
  });

  it('labels KIS candidates with their temporal event', () => {
    render(<ResultGrid results={[makeResult(1, {
      event_id: 'E2',
      metadata: {
        anchor_type: 'kis_candidate',
        rank: 3,
        event_description: 'the first fish touches the oil',
      },
    })]} />);

    expect(screen.getByText('E2 · KIS #3')).toBeInTheDocument();
    expect(screen.getByText('the first fish touches the oil')).toBeInTheDocument();
  });

  it('groups temporal KIS candidates by event', () => {
    const results = [
      makeResult(1, { event_id: 'E1', metadata: { event_description: 'first action' } }),
      makeResult(2, { event_id: 'E1', metadata: { event_description: 'first action' } }),
      makeResult(3, { event_id: 'E2', metadata: { event_description: 'second action' } }),
    ];

    render(<ResultGrid results={results} groupTemporalEvents />);

    expect(screen.getAllByText('first action')).toHaveLength(3);
    expect(screen.getAllByText('second action')).toHaveLength(2);
    expect(screen.getByText('2 frame ứng viên')).toBeInTheDocument();
    expect(screen.getByText('1 frame ứng viên')).toBeInTheDocument();
  });

  it('keeps the same frame when it belongs to different events', () => {
    const sharedFrame = makeResult(7);
    render(<ResultGrid results={[
      { ...sharedFrame, event_id: 'E1', metadata: { event_description: 'first event' } },
      { ...sharedFrame, event_id: 'E3', metadata: { event_description: 'third event' } },
    ]} groupTemporalEvents />);

    expect(screen.getAllByRole('img')).toHaveLength(2);
    expect(screen.getAllByText('E1')).toHaveLength(2);
    expect(screen.getAllByText('E3')).toHaveLength(2);
  });
});
