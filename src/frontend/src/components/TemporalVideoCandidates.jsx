function formatScore(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(3) : '—';
}

function TemporalVideoCandidates({ candidates = [], selectedVideoId = null, onSelect }) {
  if (!candidates.length) return null;

  return (
    <section className="temporal-video-results" aria-label="Temporal video candidates">
      <div className="temporal-video-results-heading">
        <div>
          <span className="temporal-step-kicker">Bước 1 · Kiểm tra video</span>
          <h2>Chọn video trước khi tìm sự kiện</h2>
        </div>
        <span>{candidates.length} video đề xuất</span>
      </div>

      <div className="temporal-video-grid">
        {candidates.map((candidate, index) => {
          const videoId = candidate.video_id;
          const isSelected = selectedVideoId === videoId;
          const summary = candidate.summary_vi || candidate.summary_en;
          const firstEventEvidence = candidate.kis_evidence?.find(
            (item) => item?.event_id === 'E1',
          ) || candidate.kis_evidence?.[0];
          const eventThumbnailUrl = firstEventEvidence?.thumbnail_url
            || (firstEventEvidence?.frame_id
              ? `/media/thumbnails/${firstEventEvidence.frame_id}.jpg`
              : null);
          return (
            <article
              className={isSelected ? 'temporal-video-card selected' : 'temporal-video-card'}
              key={videoId}
            >
              <div className="temporal-video-preview">
                {eventThumbnailUrl ? (
                  <img
                    src={eventThumbnailUrl}
                    alt={`E1 candidate for ${videoId}`}
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
              </div>

              <div className="temporal-video-card-body">
                <div className="temporal-video-title-row">
                  <strong>{videoId}</strong>
                  <span className="temporal-video-fused-score">
                    Tổng {formatScore(candidate.score)}
                  </span>
                </div>
                <p>{summary || (
                  candidate.kis_score > 0
                    ? 'Chưa có summary được lập chỉ mục; video này được đề xuất từ KIS.'
                    : 'Chưa có mô tả video.'
                )}</p>
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
        })}
      </div>
    </section>
  );
}

export default TemporalVideoCandidates;
