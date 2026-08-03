import { useState } from 'react';

const TABS = [
  { id: 'text', label: 'Text Search' },
  { id: 'image', label: 'Image Search' },
];

function SearchBar({ onSearch, onImageSearch }) {
  const [activeTab, setActiveTab] = useState('text');
  const [query, setQuery] = useState('');
  const [imageFile, setImageFile] = useState(null);

  const handleImageChange = (e) => {
    setImageFile(e.target.files?.[0] ?? null);
  };

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    // Video KIS (image) takes priority over Textual KIS when both are set,
    // matching the "Luồng 2" flow which bypasses the LLM/text pipeline entirely.
    if (imageFile) {
      onImageSearch?.(imageFile);
      return;
    }
    const trimmed = query.trim();
    if (!trimmed) return;
    onSearch?.(trimmed);
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

      <textarea
        className="search-textarea"
        rows={4}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Mô tả cảnh cần tìm (VD: người đàn ông làm rơi ví)"
      />

      <label className="search-file-label">
        <span>{imageFile ? imageFile.name : 'Chọn ảnh truy vấn (tùy chọn)'}</span>
        <input
          type="file"
          accept="image/jpeg,image/png"
          onChange={handleImageChange}
        />
      </label>

      <div className="search-actions">
        <button
          type="button"
          className="btn btn-outline"
          disabled
          title="Multi-query search — coming soon"
        >
          Add
        </button>
        <button type="submit" className="btn btn-primary">
          Tìm kiếm
        </button>
      </div>
    </form>
  );
}

export default SearchBar;
