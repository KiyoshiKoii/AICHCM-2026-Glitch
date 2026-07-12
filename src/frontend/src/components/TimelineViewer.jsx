import { useRef } from 'react';

function TimelineViewer({ frameContext }) {
  const trackRef = useRef(null);
  const dragState = useRef({ isDragging: false, startX: 0, scrollLeft: 0 });

  if (!frameContext) return null;

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
      <div className="timeline-track"></div>
    </div>
  );
}

export default TimelineViewer;
