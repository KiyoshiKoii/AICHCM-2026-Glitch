import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import TimelineViewer from './TimelineViewer.jsx';

const makeFrame = (id) => ({
  frame_id: id,
  thumbnail_url: `https://placehold.co/400x300?text=${id}`,
});

const frameContext = {
  center_frame: makeFrame('vid05_f1024'),
  before_frames: [
    makeFrame('vid05_f1019'),
    makeFrame('vid05_f1020'),
    makeFrame('vid05_f1021'),
    makeFrame('vid05_f1022'),
    makeFrame('vid05_f1023'),
  ],
  after_frames: [
    makeFrame('vid05_f1025'),
    makeFrame('vid05_f1026'),
    makeFrame('vid05_f1027'),
    makeFrame('vid05_f1028'),
    makeFrame('vid05_f1029'),
  ],
};

describe('TimelineViewer', () => {
  it('renders nothing when no frameContext is provided', () => {
    const { container } = render(<TimelineViewer frameContext={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders all 11 frames (5 before, center, 5 after)', () => {
    render(<TimelineViewer frameContext={frameContext} />);
    expect(screen.getAllByRole('img')).toHaveLength(11);
  });

  it('highlights the center frame distinctly', () => {
    const { container } = render(<TimelineViewer frameContext={frameContext} />);
    const centerWrapper = container.querySelector('.timeline-center');
    expect(centerWrapper).toBeInTheDocument();
    expect(centerWrapper.querySelector('img').alt).toBe('vid05_f1024');
  });

  it('renders just the center frame when before/after arrays are missing', () => {
    render(
      <TimelineViewer frameContext={{ center_frame: makeFrame('vid05_f1024') }} />
    );
    expect(screen.getAllByRole('img')).toHaveLength(1);
  });
});
