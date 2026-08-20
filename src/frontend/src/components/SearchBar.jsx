import { useState } from 'react';

const TABS = [
  { id: 'text', label: 'Text Search' },
  { id: 'asr', label: 'ASR Search' },
  { id: 'vqa', label: 'VQA' },
  { id: 'image', label: 'Image Search' },
  { id: 'temporal', label: 'Temporal Events' },
];

function SearchBar({
  onSearch,
  onImageSearch,
  onVqaSearch,
  onAsrSearch,
  onTemporalVideoSearch,
  onTemporalEventSearch,
  selectedTemporalVideo = null,
  onClearTemporalVideo,
  filters = { batchIds: [], videoIds: [] },
}) {
  const [activeTab, setActiveTab] = useState('text');
  const [query, setQuery] = useState('');
  const [question, setQuestion] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [useRerank, setUseRerank] = useState(false);
  const [textWeightPercent, setTextWeightPercent] = useState(50);
  const [temporalContext, setTemporalContext] = useState('');
  const [temporalEvents, setTemporalEvents] = useState(['']);
  const [summaryWeightPercent, setSummaryWeightPercent] = useState(75);

  const handleImageChange = (event) => {
    setImageFile(event.target.files?.[0] ?? null);
  };

  const handleSubmit = (event) => {
    event?.preventDefault();
    if (activeTab === 'image') {
      if (imageFile) onImageSearch?.(imageFile);
      return;
    }
    if (activeTab === 'vqa') {
      const trimmed = query.trim();
      const trimmedQuestion = question.trim();
      if (trimmed && trimmedQuestion) {
        onVqaSearch?.({ query: trimmed, question: trimmedQuestion, useRerank });
      }
      return;
    }
    if (activeTab === 'temporal') {
      const context = temporalContext.trim();
      if (!context) return;
      const events = temporalEvents.map((item) => item.trim()).filter(Boolean);
      const structuredQuery = [
        context,
        ...events.map((item, index) => `E${index + 1}: ${item}`),
      ].join('\n');

      if (selectedTemporalVideo) {
        if (!events.length) return;
        onTemporalEventSearch?.(structuredQuery, selectedTemporalVideo.video_id, filters);
      } else {
        onTemporalVideoSearch?.(
          structuredQuery,
          {
            summaryWeight: summaryWeightPercent / 100,
            kisWeight: (100 - summaryWeightPercent) / 100,
          },
          filters,
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
    onSearch?.(
      trimmed,
      useRerank,
      {
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

  const updateTemporalEvent = (index, value) => {
    setTemporalEvents((events) => events.map((item, position) => (
      position === index ? value : item
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

      {activeTab !== 'image' && activeTab !== 'temporal' && (
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
              onClick={() => setTemporalEvents((events) => [...events, ''])}
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
                  value={item}
                  onChange={(event) => updateTemporalEvent(index, event.target.value)}
                  placeholder="Mô tả khoảnh khắc hoặc hành động cần tìm"
                  aria-label={`Event ${index + 1}`}
                />
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

          {!selectedTemporalVideo && (
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

      {activeTab === 'image' && (
        <label className="search-file-label">
          <span>{imageFile ? imageFile.name : 'Chọn ảnh truy vấn'}</span>
          <input
            type="file"
            accept="image/jpeg,image/png"
            onChange={handleImageChange}
          />
        </label>
      )}

      {activeTab === 'text' && (
        <div className="fusion-weight-control">
          <div className="fusion-weight-header">
            <span>Fusion weight</span>
            <span>Text {textWeightPercent}% · Visual {100 - textWeightPercent}%</span>
          </div>
          <input
            id="text-weight-slider"
            className="fusion-weight-slider"
            type="range"
            min="0"
            max="100"
            step="5"
            value={textWeightPercent}
            onChange={(event) => setTextWeightPercent(Number(event.target.value))}
            aria-label="Text search weight"
          />
          <div className="fusion-weight-scale" aria-hidden="true">
            <span>Text</span>
            <span>Visual</span>
          </div>
        </div>
      )}

      {(activeTab === 'text' || activeTab === 'vqa') && (
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
              : 'Tìm kiếm'}
        </button>
      </div>
    </form>
  );
}

export default SearchBar;
