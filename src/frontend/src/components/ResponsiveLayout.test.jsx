import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ResultGrid from './ResultGrid.jsx';
import TimelineViewer from './TimelineViewer.jsx';

const makeResult = (i) => ({
  frame_id: `vid01_f${i}`,
  thumbnail_url: `/media/thumbnails/vid01_f${i}.jpg`,
});

const makeFrame = (id) => ({
  frame_id: id,
  thumbnail_url: `https://placehold.co/400x300?text=${id}`,
});

describe('ResultGrid responsive layout', () => {
  it('keeps the grid container present with zero results, so the layout does not collapse', () => {
    const { container } = render(<ResultGrid results={[]} />);
    expect(container.querySelector('.result-grid')).toBeInTheDocument();
    expect(screen.queryAllByRole('img')).toHaveLength(0);
  });

  it('lays out a single result the same way as many', () => {
    const { container } = render(<ResultGrid results={[makeResult(1)]} />);
    expect(container.querySelector('.result-grid')).toBeInTheDocument();
    expect(screen.getAllByRole('img')).toHaveLength(1);
  });
});

describe('TimelineViewer responsive layout', () => {
  it('keeps the horizontal-scroll container class present so overflow-x styling applies', () => {
    const frameContext = { center_frame: makeFrame('vid05_f1024') };
    const { container } = render(<TimelineViewer frameContext={frameContext} />);
    expect(container.querySelector('.timeline-viewer')).toBeInTheDocument();
  });

  it('does not break when before/after frame counts are asymmetric (e.g. near the start of a video)', () => {
    const frameContext = {
      center_frame: makeFrame('vid01_f002'),
      before_frames: [makeFrame('vid01_f000'), makeFrame('vid01_f001')],
      after_frames: [
        makeFrame('vid01_f003'),
        makeFrame('vid01_f004'),
        makeFrame('vid01_f005'),
        makeFrame('vid01_f006'),
        makeFrame('vid01_f007'),
      ],
    };
    render(<TimelineViewer frameContext={frameContext} />);
    expect(screen.getAllByRole('img')).toHaveLength(8);
  });
});
