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
          return (
            <article
              className={isSelected ? 'temporal-video-card selected' : 'temporal-video-card'}
              key={videoId}
            >
              <div className="temporal-video-preview">
                <video
                  controls
                  preload="metadata"
                  src={candidate.video_url || `/media/videos/${videoId}.mp4`}
                  aria-label={`Preview ${videoId}`}
                />
                <span className="temporal-video-rank">#{index + 1}</span>
              </div>

              <div className="temporal-video-card-body">
                <div className="temporal-video-title-row">
                  <strong>{videoId}</strong>
                  <span className="temporal-video-fused-score">
                    Tổng {formatScore(candidate.score)}
                  </span>
                </div>
                <p>{candidate.summary_vi || 'Chưa có mô tả video.'}</p>
                <div className="temporal-video-score-row">
                  <span>Summary {formatScore(candidate.summary_score)}</span>
                  <span>Event {formatScore(candidate.event_score)}</span>
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
