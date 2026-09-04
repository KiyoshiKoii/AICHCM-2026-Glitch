import { useState } from 'react';

const TABS = [
  { id: 'text', label: 'Text Search' },
  { id: 'kis-asr', label: 'KIS + ASR' },
  { id: 'asr', label: 'ASR Search' },
  { id: 'vqa', label: 'VQA' },
  { id: 'video', label: 'Video Search' },
  { id: 'temporal', label: 'Temporal Events' },
];

const normalizeBatchId = (value) => {
  const digits = String(value || '').match(/\d{1,2}/)?.[0];
  if (!digits) return '';
  return `L${String(Number(digits)).padStart(2, '0')}`;
};

const normalizeVideoSuffix = (value) => {
  const digits = String(value || '').match(/\d{1,3}/)?.[0];
  if (!digits) return '';
  return `V${String(Number(digits)).padStart(3, '0')}`;
};

const normalizeNativeFrame = (value) => {
  const normalized = String(value || '').trim();
  if (!normalized) return null;
  if (!/^\d{1,9}$/.test(normalized)) return undefined;
  return Number(normalized);
};

const createTemporalEvent = () => ({
  description: '',
  textWeightPercent: 50,
  useRerank: false,
  requiresAfterPrevious: false,
  verifyCameraMotion: false,
  motionWeightPercent: 70,
  settingsOpen: false,
});

function SearchBar({
  onSearch,
  onVqaSearch,
  onAsrSearch,
  onTemporalVideoSearch,
  onTemporalEventSearch,
  onVideoSearch,
  selectedTemporalVideo = null,
  onClearTemporalVideo,
  filters = { batchIds: [], videoIds: [] },
}) {
  const [activeTab, setActiveTab] = useState('text');
  const [query, setQuery] = useState('');
  const [question, setQuestion] = useState('');
  const [useRerank, setUseRerank] = useState(false);
  const [textWeightPercent, setTextWeightPercent] = useState(50);
  const [kisFusionWeightPercent, setKisFusionWeightPercent] = useState(65);
  const [temporalContext, setTemporalContext] = useState('');
  const [temporalEvents, setTemporalEvents] = useState([createTemporalEvent()]);
  const [summaryWeightPercent, setSummaryWeightPercent] = useState(75);
  const [temporalSearchMode, setTemporalSearchMode] = useState('summary-kis');
  const [videoBatchInput, setVideoBatchInput] = useState('L21');
  const [videoIdInput, setVideoIdInput] = useState('001');
  const [videoFrameInput, setVideoFrameInput] = useState('');

  const handleSubmit = (event) => {
    event?.preventDefault();
    if (activeTab === 'vqa') {
      const trimmed = query.trim();
      const trimmedQuestion = question.trim();
      if (trimmed && trimmedQuestion) {
        onVqaSearch?.({ query: trimmed, question: trimmedQuestion, useRerank });
      }
      return;
    }
    if (activeTab === 'video') {
      const batchId = normalizeBatchId(videoBatchInput);
      const videoId = normalizeVideoSuffix(videoIdInput);
      const frameIndex = normalizeNativeFrame(videoFrameInput);
      if (!batchId || !videoId || frameIndex === undefined) return;
      onVideoSearch?.({ batchId, videoId, frameIndex });
      return;
    }
    if (activeTab === 'temporal') {
      const context = temporalContext.trim();
      const events = temporalEvents
        .map((item, index) => ({ ...item, eventId: `E${index + 1}`, description: item.description.trim() }))
        .filter((item) => item.description);
      const isEventOnly = temporalSearchMode === 'events-only';
      if (!events.length || (!selectedTemporalVideo && !isEventOnly && !context)) return;
      const structuredQuery = [
        ...(!isEventOnly && context ? [context] : []),
        ...events.map((item) => `${item.eventId}: ${item.description}`),
      ].join('\n');
      const eventOptions = events.map((item) => ({
        eventId: item.eventId,
        textWeight: item.textWeightPercent / 100,
        visualWeight: (100 - item.textWeightPercent) / 100,
        useRerank: item.useRerank,
        requiresAfterPrevious: item.requiresAfterPrevious,
        verifyCameraMotion: item.verifyCameraMotion,
        motionWeight: item.motionWeightPercent / 100,
      }));

      if (selectedTemporalVideo) {
        onTemporalEventSearch?.(
          structuredQuery,
          selectedTemporalVideo.video_id,
          filters,
          eventOptions,
        );
      } else {
        onTemporalVideoSearch?.(
          structuredQuery,
          isEventOnly
            ? { summaryWeight: 0, kisWeight: 1 }
            : {
                summaryWeight: summaryWeightPercent / 100,
                kisWeight: (100 - summaryWeightPercent) / 100,
              },
          filters,
          eventOptions,
        );
      }
      return;
    }

    if (activeTab === 'asr') {
      const trimmed = query.trim();
      if (trimmed) onAsrSearch?.(trimmed, filters);
      return;
    }

    const trimmed = query.trim();
    if (!trimmed) return;
    const isKisAsrFusion = activeTab === 'kis-asr';
    const kisWeight = isKisAsrFusion ? kisFusionWeightPercent / 100 : 1;
    const asrWeight = isKisAsrFusion ? 1 - kisWeight : 0;
    onSearch?.(
      trimmed,
      useRerank,
      isKisAsrFusion
        ? {
            textWeight: (textWeightPercent / 100) * kisWeight,
            visualWeight: ((100 - textWeightPercent) / 100) * kisWeight,
            asrWeight,
          }
        : {
            textWeight: textWeightPercent / 100,
            visualWeight: (100 - textWeightPercent) / 100,
          },
      filters,
    );
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit(event);
    }
  };

  const updateTemporalEvent = (index, changes) => {
    setTemporalEvents((events) => events.map((item, position) => (
      position === index ? { ...item, ...changes } : item
    )));
  };

  const removeTemporalEvent = (index) => {
    setTemporalEvents((events) => events.filter((_, position) => position !== index));
  };

  return (
    <form className="search-bar" onSubmit={handleSubmit}>
      <div className="search-tabs" role="tablist">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.id}
            className={activeTab === tab.id ? 'search-tab active' : 'search-tab'}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {(activeTab !== 'temporal' && activeTab !== 'video') && (
        <label className="search-field-label">
          <span>
            {activeTab === 'vqa'
              ? 'Event description'
              : activeTab === 'asr'
                ? 'Nội dung lời thoại'
                : 'Search query'}
          </span>
          <textarea
            className="search-textarea"
            rows={activeTab === 'vqa' ? 3 : 4}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={handleKeyDown}
            aria-label={activeTab === 'asr' ? 'ASR query' : undefined}
            placeholder={activeTab === 'asr'
              ? 'Nhập lời thoại hoặc nội dung âm thanh cần tìm'
              : 'Mô tả cảnh cần tìm (VD: người đàn ông làm rơi ví)'}
          />
        </label>
      )}

      {activeTab === 'video' && (
        <section className="video-search-panel" aria-label="Direct video lookup">
          <label className="search-field-label">
            <span>Video batch</span>
            <input
              className="video-search-input"
              type="text"
              value={videoBatchInput}
              onChange={(event) => setVideoBatchInput(event.target.value)}
              placeholder="L25"
              aria-label="Video batch"
            />
          </label>
          <label className="search-field-label">
            <span>Video id</span>
            <input
              className="video-search-input"
              type="text"
              value={videoIdInput}
              onChange={(event) => setVideoIdInput(event.target.value)}
              placeholder="032"
              aria-label="Video id"
            />
          </label>
          <label className="search-field-label video-search-frame-field">
            <span>Frame (optional)</span>
            <input
              className="video-search-input"
              type="text"
              inputMode="numeric"
              value={videoFrameInput}
              onChange={(event) => setVideoFrameInput(event.target.value)}
              placeholder="e.g. 2450"
              aria-label="Video frame"
            />
          </label>
          <div className="video-search-preview" aria-live="polite">
            <span>Target</span>
            <strong>
              {normalizeBatchId(videoBatchInput) || 'L--'}_{normalizeVideoSuffix(videoIdInput) || 'V---'}
              {normalizeNativeFrame(videoFrameInput) === null
                ? ' · first frame'
                : ` · frame ${normalizeNativeFrame(videoFrameInput) ?? '—'}`}
            </strong>
          </div>
        </section>
      )}
      {activeTab === 'temporal' && (
        <section className="temporal-search" aria-label="Temporal event search form">
          <div className="temporal-workflow-steps" aria-label="TRAKE workflow">
            <div className={selectedTemporalVideo ? 'temporal-workflow-step complete' : 'temporal-workflow-step active'}>
              <span>1</span>
              <div>
                <strong>Tìm và kiểm tra video</strong>
                <small>Summary + KIS visual/caption</small>
              </div>
            </div>
            <div className={selectedTemporalVideo ? 'temporal-workflow-step active' : 'temporal-workflow-step'}>
              <span>2</span>
              <div>
                <strong>Tìm E1…En</strong>
                <small>Chỉ trong video đã chọn</small>
              </div>
            </div>
          </div>

          {selectedTemporalVideo && (
            <div className="temporal-selected-video" role="status">
              <div>
                <span>Video đang chọn</span>
                <strong>{selectedTemporalVideo.video_id}</strong>
              </div>
              <button type="button" onClick={() => onClearTemporalVideo?.()}>
                Đổi video
              </button>
            </div>
          )}

          {!selectedTemporalVideo && (
            <div className="temporal-search-mode" role="group" aria-label="Video selection mode">
              <button
                type="button"
                className={temporalSearchMode === 'summary-kis' ? 'active' : ''}
                onClick={() => setTemporalSearchMode('summary-kis')}
              >
                Summary + Events
              </button>
              <button
                type="button"
                className={temporalSearchMode === 'events-only' ? 'active' : ''}
                onClick={() => setTemporalSearchMode('events-only')}
              >
                Chỉ Events
              </button>
            </div>
          )}

          {!selectedTemporalVideo && temporalSearchMode === 'summary-kis' && (
          <label className="search-field-label">
            <span>Summary chung của video</span>
            <textarea
              className="search-textarea temporal-context-input"
              rows={3}
              value={temporalContext}
              onChange={(event) => setTemporalContext(event.target.value)}
              placeholder="Ví dụ: Video múa lân đen trắng biểu diễn trên hệ thống cột cao"
              aria-label="Video context"
              disabled={Boolean(selectedTemporalVideo)}
            />
            <small>
              {selectedTemporalVideo
                ? 'Summary được khóa để bước tìm sự kiện luôn bám đúng video đã kiểm tra.'
                : 'Dùng để xếp hạng video trước; chưa định vị khoảnh khắc ở bước này.'}
            </small>
          </label>
          )}

          <div className="temporal-events-header">
            <div>
              <strong>Events</strong>
              <span>
                {selectedTemporalVideo
                  ? 'Nhập các khoảnh khắc cần tìm trong video đã chọn'
                  : 'Tùy chọn: thêm gợi ý để hỗ trợ xếp hạng video'}
              </span>
            </div>
            <button
              type="button"
              className="temporal-event-add"
              onClick={() => setTemporalEvents((events) => [...events, createTemporalEvent()])}
              disabled={temporalEvents.length >= 8}
            >
              + Add event
            </button>
          </div>

          <div className="temporal-events-list">
            {temporalEvents.map((item, index) => (
              <div className="temporal-event-row" key={`temporal-event-${index}`}>
                <span className="temporal-event-label" aria-hidden="true">E{index + 1}</span>
                <textarea
                  className="search-textarea temporal-event-input"
                  rows={2}
                  value={item.description}
                  onChange={(event) => updateTemporalEvent(index, { description: event.target.value })}
                  placeholder="Mô tả khoảnh khắc hoặc hành động cần tìm"
                  aria-label={`Event ${index + 1}`}
                />
                <button
                  type="button"
                  className={item.settingsOpen ? 'temporal-event-settings-toggle active' : 'temporal-event-settings-toggle'}
                  onClick={() => updateTemporalEvent(index, { settingsOpen: !item.settingsOpen })}
                  aria-expanded={item.settingsOpen}
                  aria-label={`Advanced settings for event ${index + 1}`}
                  title="Event settings"
                >
                  Settings
                </button>
                {item.settingsOpen && (
                  <section className="temporal-event-advanced" aria-label={`Event ${index + 1} advanced settings`}>
                    <div className="fusion-weight-control temporal-event-weight-control">
                      <div className="fusion-weight-header">
                        <span>KIS fusion</span>
                        <span>
                          Text/caption {item.textWeightPercent}% / Visual {100 - item.textWeightPercent}%
                        </span>
                      </div>
                      <input
                        className="fusion-weight-slider"
                        type="range"
                        min="0"
                        max="100"
                        step="5"
                        value={item.textWeightPercent}
                        onChange={(event) => updateTemporalEvent(index, { textWeightPercent: Number(event.target.value) })}
                        aria-label={`Event ${index + 1} KIS text weight`}
                      />
                      <div className="fusion-weight-scale" aria-hidden="true">
                        <span>Text/caption</span>
                        <span>Visual Qwen</span>
                      </div>
                    </div>

                    <label className="search-rerank-toggle temporal-event-toggle">
                      <input
                        type="checkbox"
                        checked={item.useRerank}
                        onChange={(event) => updateTemporalEvent(index, { useRerank: event.target.checked })}
                      />
                      <span>Gemini re-rank this event (uses an extra request)</span>
                    </label>

                    {index > 0 && (
                      <label className="search-rerank-toggle temporal-event-toggle">
                        <input
                          type="checkbox"
                          checked={item.requiresAfterPrevious}
                          onChange={(event) => updateTemporalEvent(index, { requiresAfterPrevious: event.target.checked })}
                        />
                        <span>E{index + 1} must occur after E{index}</span>
                      </label>
                    )}

                    <label className="search-rerank-toggle temporal-event-toggle">
                      <input
                        type="checkbox"
                        checked={item.verifyCameraMotion}
                        onChange={(event) => updateTemporalEvent(index, { verifyCameraMotion: event.target.checked })}
                      />
                      <span>Verify shot transition / camera motion</span>
                    </label>

                    {item.verifyCameraMotion && (
                      <div className="fusion-weight-control temporal-event-weight-control temporal-motion-weight-control">
                        <div className="fusion-weight-header">
                          <span>KIS / Camera motion</span>
                          <span>
                            KIS {100 - item.motionWeightPercent}% / Motion {item.motionWeightPercent}%
                          </span>
                        </div>
                        <input
                          className="fusion-weight-slider"
                          type="range"
                          min="0"
                          max="100"
                          step="5"
                          value={item.motionWeightPercent}
                          onChange={(event) => updateTemporalEvent(index, { motionWeightPercent: Number(event.target.value) })}
                          aria-label={`Event ${index + 1} camera motion weight`}
                        />
                        <div className="fusion-weight-scale" aria-hidden="true">
                          <span>KIS</span>
                          <span>Camera motion</span>
                        </div>
                      </div>
                    )}
                  </section>
                )}
                <button
                  type="button"
                  className="temporal-event-remove"
                  onClick={() => removeTemporalEvent(index)}
                  aria-label={`Remove event ${index + 1}`}
                  title="Remove event"
                >
                  ×
                </button>
              </div>
            ))}
          </div>

          {!selectedTemporalVideo && temporalSearchMode === 'summary-kis' && (
            <div className="fusion-weight-control temporal-fusion-control">
              <div className="fusion-weight-header">
                <span>Video ranking weight</span>
                <span>
                  Summary {summaryWeightPercent}% · KIS {100 - summaryWeightPercent}%
                </span>
              </div>
              <input
                id="summary-weight-slider"
                className="fusion-weight-slider"
                type="range"
                min="0"
                max="100"
                step="5"
                value={summaryWeightPercent}
                onChange={(event) => setSummaryWeightPercent(Number(event.target.value))}
                aria-label="Summary video ranking weight"
              />
              <div className="fusion-weight-scale" aria-hidden="true">
                <span>Summary</span>
                <span>KIS visual + caption</span>
              </div>
            </div>
          )}

        </section>
      )}

      {activeTab === 'vqa' && (
        <label className="search-field-label">
          <span>Question</span>
          <textarea
            className="search-textarea"
            rows={3}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Câu hỏi cần trả lời từ ảnh"
          />
        </label>
      )}

      {(activeTab === 'text' || activeTab === 'kis-asr') && (
        <div className="fusion-weight-control">
          <div className="fusion-weight-header">
            <span>{activeTab === 'kis-asr' ? 'KIS split' : 'Fusion weight'}</span>
            <span>
              {activeTab === 'kis-asr'
                ? `Caption ${textWeightPercent}% · Visual ${100 - textWeightPercent}%`
                : `Text ${textWeightPercent}% · Visual ${100 - textWeightPercent}%`}
            </span>
          </div>
          <input
            id={activeTab === 'kis-asr' ? 'kis-asr-text-weight-slider' : 'text-weight-slider'}
            className="fusion-weight-slider"
            type="range"
            min="0"
            max="100"
            step="5"
            value={textWeightPercent}
            onChange={(event) => setTextWeightPercent(Number(event.target.value))}
            aria-label={activeTab === 'kis-asr' ? 'KIS caption and visual weight' : 'Text search weight'}
          />
          <div className="fusion-weight-scale" aria-hidden="true">
            <span>{activeTab === 'kis-asr' ? 'Caption' : 'Text'}</span>
            <span>Visual Qwen</span>
          </div>
          {activeTab === 'kis-asr' && (
            <>
              <div className="fusion-weight-header kis-asr-weight-header">
                <span>KIS / ASR evidence</span>
                <span>KIS {kisFusionWeightPercent}% / ASR {100 - kisFusionWeightPercent}%</span>
              </div>
              <input
                id="kis-asr-weight-slider"
                className="fusion-weight-slider"
                type="range"
                min="0"
                max="100"
                step="5"
                value={kisFusionWeightPercent}
                onChange={(event) => setKisFusionWeightPercent(Number(event.target.value))}
                aria-label="KIS and ASR fusion weight"
              />
              <div className="fusion-weight-scale" aria-hidden="true">
                <span>KIS visual + caption</span>
                <span>ASR</span>
              </div>
            </>
          )}
        </div>
      )}

      {(activeTab === 'text' || activeTab === 'kis-asr' || activeTab === 'vqa') && (
        <label className="search-rerank-toggle">
          <input
            type="checkbox"
            checked={useRerank}
            onChange={(event) => setUseRerank(event.target.checked)}
          />
          <span>Gemini re-rank (tốn 1 request)</span>
        </label>
      )}

      <div className="search-actions">
        <button type="submit" className="btn btn-primary">
          {activeTab === 'asr'
            ? 'Tìm lời thoại'
            : activeTab === 'vqa'
            ? 'Trả lời'
            : activeTab === 'temporal'
              ? selectedTemporalVideo
                ? `Tìm sự kiện trong ${selectedTemporalVideo.video_id}`
                : 'Tìm video'
              : activeTab === 'video'
                ? 'Mở video'
                : 'Tìm kiếm'}
        </button>
      </div>
    </form>
  );
}

export default SearchBar;



