function ImageCard({ result }) {
  const {
    frame_id: frameId,
    thumbnail_url: thumbnailUrl,
    video_name: videoName,
    metadata,
  } = result;

  const videoLabel = videoName || 'Unknown video';
  const timestamp = metadata?.timestamp ?? 'Unknown time';

  return (
    <figure className="image-card">
      <img src={thumbnailUrl} alt={frameId} loading="lazy" />
      <figcaption>
        <span className="video-name">{videoLabel}</span>
        <span className="timestamp">{timestamp}</span>
      </figcaption>
    </figure>
  );
}

export default ImageCard;
