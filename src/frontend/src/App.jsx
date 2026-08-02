import { useState } from 'react';
import Navbar from './components/Navbar.jsx';
import Sidebar from './components/Sidebar.jsx';
import ResultGrid from './components/ResultGrid.jsx';
import Pagination from './components/Pagination.jsx';
import TimelineViewer from './components/TimelineViewer.jsx';
import LoadingSpinner from './components/LoadingSpinner.jsx';
import { fetchSearchResults, getFrameContext } from './api/mockClient.js';
import './App.css';

const PAGE_SIZE = 12;

function App() {
  const [dataset, setDataset] = useState('V3C1');
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [frameContext, setFrameContext] = useState(null);
  const [isContextLoading, setIsContextLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [lastQuery, setLastQuery] = useState(null);

  const runSearch = async (input) => {
    setIsLoading(true);
    setCurrentPage(1);
    setLastQuery(input);
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

  const handleNewSearch = () => {
    setResults([]);
    setFrameContext(null);
    setCurrentPage(1);
    setLastQuery(null);
  };

  const handleRepeatSearch = () => {
    if (lastQuery) runSearch(lastQuery);
  };

  const totalPages = Math.ceil(results.length / PAGE_SIZE) || 1;
  const pageResults = results.slice(
    (currentPage - 1) * PAGE_SIZE,
    currentPage * PAGE_SIZE
  );

  return (
    <div className="app">
      <Navbar />
      <div className="app-body">
        <Sidebar
          dataset={dataset}
          onDatasetChange={setDataset}
          onNewSearch={handleNewSearch}
          onSearch={runSearch}
          onImageSearch={runSearch}
          onRepeatSearch={handleRepeatSearch}
          canRepeatSearch={Boolean(lastQuery)}
        />
        <main className="main-content">
          {isLoading ? (
            <LoadingSpinner label="Đang tìm kiếm..." />
          ) : (
            <>
              <ResultGrid results={pageResults} onCardDoubleClick={handleCardDoubleClick} />
              <Pagination
                currentPage={currentPage}
                totalPages={totalPages}
                onPageChange={setCurrentPage}
              />
            </>
          )}
          {isContextLoading ? (
            <LoadingSpinner label="Đang tải ngữ cảnh..." />
          ) : (
            <TimelineViewer frameContext={frameContext} />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
