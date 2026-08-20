function ImageCard({ result = {}, onDoubleClick, onClick }) {
  const {
    frame_id: frameId,
    thumbnail_url: thumbnailUrl,
    video_name: videoName,
    frame_index: frameIndex,
    answer,
    confidence,
    metadata = {},
  } = result;

  const videoLabel = videoName || 'Unknown video';
  const fPart = frameId?.includes('_f') ? frameId.split('_f')[1] : null;
  const fDisplay = fPart ? `[f${fPart}]` : '';
  const frameDisplay = frameIndex !== undefined 
    ? `${fDisplay} Frame: ${frameIndex}`.trim() 
    : 'Unknown frame';
  const formatTimestamp = (timestampMs) => {
    const totalSeconds = Math.max(0, Math.floor(Number(timestampMs) / 1000));
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return hours > 0
      ? `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
      : `${minutes}:${String(seconds).padStart(2, '0')}`;
  };
  const hasAsrRange = Number.isFinite(Number(metadata.asr_start_ms))
    && Number.isFinite(Number(metadata.asr_end_ms));

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
        {hasAsrRange && (
          <span className="asr-time-range">
            Audio {formatTimestamp(metadata.asr_start_ms)}–{formatTimestamp(metadata.asr_end_ms)}
          </span>
        )}
        {metadata.transcript && (
          <span className="asr-transcript">{metadata.transcript}</span>
        )}
        {answer && (
          <span className="vqa-answer">
            Answer: {answer}{confidence !== null && confidence !== undefined ? ` (${Math.round(confidence * 100)}%)` : ''}
          </span>
        )}
      </figcaption>
    </figure>
  );
}

export default ImageCard;
