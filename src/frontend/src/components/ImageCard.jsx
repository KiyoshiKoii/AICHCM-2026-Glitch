function ImageCard({ result = {}, onDoubleClick }) {
  const {
    frame_id: frameId,
    thumbnail_url: thumbnailUrl,
    video_name: videoName,
    frame_index: frameIndex,
  } = result;

  const videoLabel = videoName || 'Unknown video';
  const frameDisplay = frameIndex !== undefined ? `Frame: ${frameIndex}` : 'Unknown frame';

  return (
    <figure
      className="image-card"
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
