import { useState } from 'react';
import SearchBar from './components/SearchBar.jsx';
import ResultGrid from './components/ResultGrid.jsx';
import TimelineViewer from './components/TimelineViewer.jsx';
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
      {!isContextLoading && <TimelineViewer frameContext={frameContext} />}
    </div>
  );
}

export default App;
