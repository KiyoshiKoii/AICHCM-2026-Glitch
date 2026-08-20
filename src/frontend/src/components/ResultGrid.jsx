import ImageCard from './ImageCard.jsx';

const MAX_RESULTS = 20;

function ResultGrid({ results, onCardDoubleClick, onCardClick }) {
  // ASR may return multiple passages whose thumbnails use the same keyframe.
  // Keep those passages distinct while preserving frame deduplication elsewhere.
  const uniqueResults = [];
  const seenResultIds = new Set();
  
  for (const result of (results ?? [])) {
    const resultId = result?.metadata?.asr_id || result?.frame_id;
    if (result?.frame_id && result?.thumbnail_url && !seenResultIds.has(resultId)) {
      seenResultIds.add(resultId);
      uniqueResults.push(result);
    }
  }

  const visibleResults = uniqueResults.slice(0, MAX_RESULTS);

  return (
    <div className="result-grid">
      {visibleResults.map((result, index) => (
        <ImageCard
          key={`${result.metadata?.asr_id || result.frame_id}-${index}`}
          result={result}
          onDoubleClick={onCardDoubleClick}
          onClick={() => onCardClick?.(result)}
        />
      ))}
    </div>
  );
}

export default ResultGrid;
