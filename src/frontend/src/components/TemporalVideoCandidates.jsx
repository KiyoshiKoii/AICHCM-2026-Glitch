import { useEffect, useMemo, useState } from 'react';
import Pagination from './Pagination.jsx';

const PAGE_SIZE = 15;

function formatScore(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(3) : '—';
}

function getEventEvidence(evidence) {
  const firstEvidenceByEvent = new Map();
  for (const item of evidence || []) {
    const eventId = item?.event_id;
    if (typeof eventId !== 'string' || !eventId || firstEvidenceByEvent.has(eventId)) continue;
    firstEvidenceByEvent.set(eventId, item);
  }
  return Array.from(firstEvidenceByEvent.values());
}

function TemporalVideoCard({ candidate, index, isSelected, onSelect }) {
  const videoId = candidate.video_id;
  const summary = candidate.summary_vi || candidate.summary_en;
  const eventEvidence = useMemo(
    () => getEventEvidence(candidate.kis_evidence),
    [candidate.kis_evidence],
  );
  const [eventIndex, setEventIndex] = useState(0);

  useEffect(() => {
    setEventIndex(0);
  }, [videoId, candidate.kis_evidence]);

  const activeEvidence = eventEvidence[
    eventEvidence.length ? Math.min(eventIndex, eventEvidence.length - 1) : 0
  ];
  const eventThumbnailUrl = activeEvidence?.thumbnail_url
    || (activeEvidence?.frame_id
      ? `/media/thumbnails/${activeEvidence.frame_id}.jpg`
      : null);
  const hasMultipleEvents = eventEvidence.length > 1;
  const moveEvent = (direction) => {
    if (!eventEvidence.length) return;
    setEventIndex((current) => (
      (current + direction + eventEvidence.length) % eventEvidence.length
    ));
  };

  return (
    <article className={isSelected ? 'temporal-video-card selected' : 'temporal-video-card'}>
      <div className="temporal-video-preview">
        {eventThumbnailUrl ? (
          <img
            src={eventThumbnailUrl}
            alt={`${activeEvidence.event_id} candidate for ${videoId}`}
          />
        ) : (
          <video
            controls
            preload="metadata"
            src={candidate.video_url || `/media/videos/${videoId}.mp4`}
            aria-label={`Preview ${videoId}`}
          />
        )}
        <span className="temporal-video-rank">#{index + 1}</span>
        {activeEvidence && (
          <div className="temporal-video-event-switcher">
            {hasMultipleEvents && (
              <button
                type="button"
                onClick={() => moveEvent(-1)}
                aria-label={`Previous event evidence for ${videoId}`}
                title="Sự kiện trước"
              >
                ‹
              </button>
            )}
            <span title={activeEvidence.caption || activeEvidence.frame_id}>
              {activeEvidence.event_id} · {eventIndex + 1}/{eventEvidence.length}
            </span>
            {hasMultipleEvents && (
              <button
                type="button"
                onClick={() => moveEvent(1)}
                aria-label={`Next event evidence for ${videoId}`}
                title="Sự kiện tiếp"
              >
                ›
              </button>
            )}
          </div>
        )}
      </div>

      <div className="temporal-video-card-body">
        <div className="temporal-video-title-row">
          <strong>{videoId}</strong>
          <span className="temporal-video-fused-score">
            Tổng {formatScore(candidate.score)}
          </span>
        </div>
        <p
          className="temporal-video-summary"
          tabIndex={0}
          aria-label={`Video summary for ${videoId}`}
        >
          {summary || (
          candidate.kis_score > 0
            ? 'Chưa có summary được lập chỉ mục; video này được đề xuất từ KIS.'
            : 'Chưa có mô tả video.'
          )}
        </p>
        <div className="temporal-video-score-row">
          <span>Summary {formatScore(candidate.summary_score)}</span>
          <span>KIS {formatScore(candidate.kis_score ?? candidate.event_score)}</span>
        </div>
        <button
          type="button"
          className={isSelected ? 'btn btn-primary' : 'btn btn-outline'}
          onClick={() => onSelect?.(candidate)}
        >
          {isSelected ? 'Đã chọn video này' : 'Chọn để tìm sự kiện'}
        </button>
      </div>
    </article>
  );
}

function TemporalVideoCandidates({ candidates = [], selectedVideoId = null, onSelect }) {
  const [currentPage, setCurrentPage] = useState(1);
  const totalPages = Math.ceil(candidates.length / PAGE_SIZE);
  const pageCandidates = candidates.slice(
    (currentPage - 1) * PAGE_SIZE,
    currentPage * PAGE_SIZE,
  );

  useEffect(() => {
    setCurrentPage(1);
  }, [candidates]);

  if (!candidates.length) return null;

  return (
    <section className="temporal-video-results" aria-label="Temporal video candidates">
      <div className="temporal-video-results-heading">
        <div>
          <span className="temporal-step-kicker">Bước 1 · Kiểm tra video</span>
          <h2>Chọn video trước khi tìm sự kiện</h2>
        </div>
        <span>{candidates.length} video đề xuất · trang {currentPage}/{totalPages}</span>
      </div>

      <div className="temporal-video-grid">
        {pageCandidates.map((candidate, index) => (
          <TemporalVideoCard
            key={candidate.video_id}
            candidate={candidate}
            index={(currentPage - 1) * PAGE_SIZE + index}
            isSelected={selectedVideoId === candidate.video_id}
            onSelect={onSelect}
          />
        ))}
      </div>
      <Pagination
        currentPage={currentPage}
        totalPages={totalPages}
        onPageChange={setCurrentPage}
      />
    </section>
  );
}

export default TemporalVideoCandidates;
