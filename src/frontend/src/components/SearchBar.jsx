import { useState } from 'react';

function SearchBar({ onSearch }) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
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
      <button type="submit">Tìm kiếm</button>
    </form>
  );
}

export default SearchBar;
