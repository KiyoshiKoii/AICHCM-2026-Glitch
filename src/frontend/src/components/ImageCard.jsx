function ImageCard({ result = {}, onDoubleClick, onClick }) {
  const {
    frame_id: frameId,
    thumbnail_url: thumbnailUrl,
    video_name: videoName,
    frame_index: frameIndex,
  } = result;

  const videoLabel = videoName || 'Unknown video';
  const fPart = frameId?.includes('_f') ? frameId.split('_f')[1] : null;
  const fDisplay = fPart ? `[f${fPart}]` : '';
  const frameDisplay = frameIndex !== undefined 
    ? `${fDisplay} Frame: ${frameIndex}`.trim() 
    : 'Unknown frame';

  return (
    <figure
      className="image-card"
      onClick={onClick}
      onDoubleClick={() => onDoubleClick?.(frameId)}
    >
      <img src={thumbnailUrl} alt={frameId ?? 'unknown frame'} loading="lazy" />
      <figcaption>
        <span className="video-name">{videoLabel}</span>
        <span className="timestamp">{frameDisplay}</span>
      </figcaption>
    </figure>
  );
}

export default ImageCard;
