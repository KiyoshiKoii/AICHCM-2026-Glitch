import { useRef } from 'react';
import ImageCard from './ImageCard.jsx';

function TimelineViewer({ frameContext }) {
  const trackRef = useRef(null);
  const dragState = useRef({ isDragging: false, startX: 0, scrollLeft: 0 });

  if (!frameContext) return null;

  const {
    center_frame: centerFrame,
    before_frames: beforeFrames = [],
    after_frames: afterFrames = [],
  } = frameContext;

  const handleMouseDown = (e) => {
    const track = trackRef.current;
    dragState.current = {
      isDragging: true,
      startX: e.pageX - track.offsetLeft,
      scrollLeft: track.scrollLeft,
    };
  };

  const handleMouseMove = (e) => {
    if (!dragState.current.isDragging) return;
    e.preventDefault();
    const track = trackRef.current;
    const x = e.pageX - track.offsetLeft;
    const walk = x - dragState.current.startX;
    track.scrollLeft = dragState.current.scrollLeft - walk;
  };

  const stopDragging = () => {
    dragState.current.isDragging = false;
  };

  return (
    <div
      className="timeline-viewer"
      ref={trackRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={stopDragging}
      onMouseLeave={stopDragging}
    >
      <div className="timeline-track">
        {beforeFrames.map((frame) => (
          <ImageCard key={frame.frame_id} result={frame} />
        ))}
        {centerFrame && (
          <div className="timeline-center">
            <ImageCard result={centerFrame} />
          </div>
        )}
        {afterFrames.map((frame) => (
          <ImageCard key={frame.frame_id} result={frame} />
        ))}
      </div>
    </div>
  );
}

export default TimelineViewer;
