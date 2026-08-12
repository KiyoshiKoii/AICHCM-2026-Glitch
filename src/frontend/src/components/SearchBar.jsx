import { useState } from 'react';

const TABS = [
  { id: 'text', label: 'Text Search' },
  { id: 'vqa', label: 'VQA' },
  { id: 'image', label: 'Image Search' },
];

function SearchBar({ onSearch, onImageSearch, onVqaSearch, filters = { batchIds: [], videoIds: [] } }) {
  const [activeTab, setActiveTab] = useState('text');
  const [query, setQuery] = useState('');
  const [question, setQuestion] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [useRerank, setUseRerank] = useState(false);
  const [textWeightPercent, setTextWeightPercent] = useState(50);

  const handleImageChange = (e) => {
    setImageFile(e.target.files?.[0] ?? null);
  };

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    if (activeTab === 'image') {
      if (!imageFile) return;
      onImageSearch?.(imageFile);
      return;
    }
    const trimmed = query.trim();
    if (!trimmed) return;
    if (activeTab === 'vqa') {
      const trimmedQuestion = question.trim();
      if (!trimmedQuestion) return;
      onVqaSearch?.({ query: trimmed, question: trimmedQuestion, useRerank });
      return;
    }
    onSearch?.(trimmed, useRerank, {
      textWeight: textWeightPercent / 100,
      visualWeight: (100 - textWeightPercent) / 100,
    }, filters);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault(); // Prevent default new line
      handleSubmit(e);
    }
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

      {activeTab !== 'image' && (
        <label className="search-field-label">
          <span>{activeTab === 'vqa' ? 'Event description' : 'Search query'}</span>
          <textarea
            className="search-textarea"
            rows={activeTab === 'vqa' ? 3 : 4}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Mô tả cảnh cần tìm (VD: người đàn ông làm rơi ví)"
          />
        </label>
      )}

      {activeTab === 'vqa' && (
        <label className="search-field-label">
          <span>Question</span>
          <textarea
            className="search-textarea"
            rows={3}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
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
        <>
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
              onChange={(e) => setTextWeightPercent(Number(e.target.value))}
              aria-label="Text search weight"
            />
            <div className="fusion-weight-scale" aria-hidden="true">
              <span>Text</span>
              <span>Visual</span>
            </div>
          </div>
        </>
      )}

      {activeTab !== 'image' && (
        <label className="search-rerank-toggle">
          <input
            type="checkbox"
            checked={useRerank}
            onChange={(e) => setUseRerank(e.target.checked)}
          />
          <span>Gemini re-rank (tốn 1 request)</span>
        </label>
      )}

      <div className="search-actions">
        <button type="submit" className="btn btn-primary">
          {activeTab === 'vqa' ? 'Trả lời' : 'Tìm kiếm'}
        </button>
      </div>
    </form>
  );
}

export default SearchBar;
