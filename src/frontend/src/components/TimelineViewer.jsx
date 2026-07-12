function TimelineViewer({ frameContext }) {
  if (!frameContext) return null;

  return (
    <div className="timeline-viewer">
      <div className="timeline-track"></div>
    </div>
  );
}

export default TimelineViewer;
