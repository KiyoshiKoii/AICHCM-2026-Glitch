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

  it('truncates to the top 20 results when more are provided', () => {
    const results = Array.from({ length: 35 }, (_, i) => makeResult(i));
    render(<ResultGrid results={results} />);
    expect(screen.getAllByRole('img')).toHaveLength(20);
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
});
