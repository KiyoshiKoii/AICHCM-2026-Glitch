import ImageCard from './ImageCard.jsx';

const MAX_RESULTS = 15;

function ResultGrid({
  results,
  onCardDoubleClick,
  onCardClick,
  groupTemporalEvents = false,
}) {
  // ASR may return multiple passages whose thumbnails use the same keyframe.
  // Keep those passages distinct while preserving frame deduplication elsewhere.
  const uniqueResults = [];
  const seenResultIds = new Set();
  
  for (const result of (results ?? [])) {
    const eventId = result?.event_id || result?.metadata?.event_id;
    const resultId = groupTemporalEvents && eventId
      ? `${eventId}:${result?.frame_id}`
      : result?.metadata?.asr_id || result?.frame_id;
    if (result?.frame_id && result?.thumbnail_url && !seenResultIds.has(resultId)) {
      seenResultIds.add(resultId);
      uniqueResults.push(result);
    }
  }

  const visibleResults = uniqueResults.slice(0, MAX_RESULTS);
  const renderCard = (result, index) => (
    <ImageCard
      key={`${result.metadata?.asr_id || result.frame_id}-${index}`}
      result={result}
      onDoubleClick={onCardDoubleClick}
      onClick={() => onCardClick?.(result)}
    />
  );

  if (groupTemporalEvents) {
    const groups = new Map();
    for (const result of uniqueResults) {
      const eventId = result.event_id || result.metadata?.event_id || 'Other';
      if (!groups.has(eventId)) groups.set(eventId, []);
      groups.get(eventId).push(result);
    }

    return (
      <div className="temporal-event-result-groups">
        {[...groups.entries()].map(([eventId, eventResults]) => {
          const description = eventResults[0]?.metadata?.event_description;
          return (
            <section className="temporal-event-result-group" key={eventId}>
              <header>
                <span>{eventId}</span>
                <div>
                  <strong>{description || 'KIS candidates'}</strong>
                  <small>{Math.min(eventResults.length, 10)} frame ứng viên</small>
                </div>
              </header>
              <div className="result-grid">
                {eventResults.slice(0, 10).map(renderCard)}
              </div>
            </section>
          );
        })}
      </div>
    );
  }

  return (
    <div className="result-grid">
      {visibleResults.map(renderCard)}
    </div>
  );
}

export default ResultGrid;
