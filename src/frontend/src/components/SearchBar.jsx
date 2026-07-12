import { useState } from 'react';

function SearchBar({ onSearch, onImageSearch }) {
  const [query, setQuery] = useState('');
  const [imageFile, setImageFile] = useState(null);

  const handleImageChange = (e) => {
    setImageFile(e.target.files?.[0] ?? null);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
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

  return (
    <form className="search-bar" onSubmit={handleSubmit}>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Mô tả cảnh cần tìm (VD: người đàn ông làm rơi ví)"
      />
      <input
        type="file"
        accept="image/jpeg,image/png"
        onChange={handleImageChange}
      />
      <button type="submit">Tìm kiếm</button>
    </form>
  );
}

export default SearchBar;
