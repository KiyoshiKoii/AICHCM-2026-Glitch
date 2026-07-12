import { useState } from 'react';
import SearchBar from './components/SearchBar.jsx';
import ResultGrid from './components/ResultGrid.jsx';
import { fetchSearchResults, getFrameContext } from './api/mockClient.js';
import './App.css';

function App() {
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [frameContext, setFrameContext] = useState(null);
  const [isContextLoading, setIsContextLoading] = useState(false);

  const runSearch = async (input) => {
    setIsLoading(true);
    try {
      const response = await fetchSearchResults(input);
      setResults(response.data.results);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCardDoubleClick = async (frameId) => {
    setIsContextLoading(true);
    try {
      const response = await getFrameContext(frameId);
      setFrameContext(response.data);
    } finally {
      setIsContextLoading(false);
    }
  };

  return (
    <div className="app">
      <SearchBar onSearch={runSearch} onImageSearch={runSearch} />
      {isLoading ? (
        <p>Đang tìm kiếm...</p>
      ) : (
        <ResultGrid results={results} onCardDoubleClick={handleCardDoubleClick} />
      )}
      {isContextLoading && <p>Đang tải ngữ cảnh...</p>}
      {frameContext && !isContextLoading && (
        <ul className="frame-context-preview">
          {frameContext.before_frames.map((frame) => (
            <li key={frame.frame_id}>{frame.frame_id}</li>
          ))}
          <li>
            <strong>{frameContext.center_frame.frame_id}</strong>
          </li>
          {frameContext.after_frames.map((frame) => (
            <li key={frame.frame_id}>{frame.frame_id}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default App;
