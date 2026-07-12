import ImageCard from './ImageCard.jsx';

const MAX_RESULTS = 20;

function ResultGrid({ results }) {
  const visibleResults = (results ?? [])
    .filter((result) => result?.frame_id && result?.thumbnail_url)
    .slice(0, MAX_RESULTS);

  return (
    <div className="result-grid">
      {visibleResults.map((result) => (
        <ImageCard key={result.frame_id} result={result} />
      ))}
    </div>
  );
}

export default ResultGrid;
