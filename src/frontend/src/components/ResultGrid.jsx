import ImageCard from './ImageCard.jsx';

const MAX_RESULTS = 20;

function ResultGrid({ results, onCardDoubleClick, onCardClick }) {
  // Deduplicate by frame_id to prevent React key collisions and visual bugs
  const uniqueResults = [];
  const seenFrameIds = new Set();
  
  for (const result of (results ?? [])) {
    if (result?.frame_id && result?.thumbnail_url && !seenFrameIds.has(result.frame_id)) {
      seenFrameIds.add(result.frame_id);
      uniqueResults.push(result);
    }
  }

  const visibleResults = uniqueResults.slice(0, MAX_RESULTS);

  return (
    <div className="result-grid">
      {visibleResults.map((result, index) => (
        <ImageCard
          key={`${result.frame_id}-${index}`}
          result={result}
          onDoubleClick={onCardDoubleClick}
          onClick={() => onCardClick?.(result.thumbnail_url)}
        />
      ))}
    </div>
  );
}

export default ResultGrid;
