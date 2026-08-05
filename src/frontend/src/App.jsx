import { useState } from 'react';
import Navbar from './components/Navbar.jsx';
import Sidebar from './components/Sidebar.jsx';
import ResultGrid from './components/ResultGrid.jsx';
import Pagination from './components/Pagination.jsx';
import TimelineViewer from './components/TimelineViewer.jsx';
import LoadingSpinner from './components/LoadingSpinner.jsx';
import { searchByText, searchByImage, getFrameContext } from './api/apiClient.js';
import './App.css';

const PAGE_SIZE = 12;

function App() {
  const [dataset, setDataset] = useState('V3C1');
  const [results, setResults] = useState([]);
  const [llmResults, setLlmResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [frameContext, setFrameContext] = useState(null);
  const [isContextLoading, setIsContextLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [lastQuery, setLastQuery] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);

  const runSearch = async (input) => {
    setIsLoading(true);
    setCurrentPage(1);
    setLastQuery(input);
    try {
      let response;
      if (input instanceof File) {
        response = await searchByImage(input);
      } else {
        response = await searchByText(input);
      }
      setResults(response.data.results || []);
      setLlmResults(response.data.llm_reranked_results || []);
    } catch (err) {
      console.error(err);
      alert('Lỗi trong quá trình tìm kiếm! Xem console để biết thêm chi tiết.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCardDoubleClick = async (frameId) => {
    setIsContextLoading(true);
    try {
      const response = await getFrameContext(frameId);
      setFrameContext(response.data);
    } catch (err) {
      console.error(err);
      alert('Lỗi tải timeline!');
    } finally {
      setIsContextLoading(false);
    }
  };

  const handleNewSearch = () => {
    setResults([]);
    setLlmResults([]);
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
  const pageLlmResults = llmResults.slice(
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
              {llmResults.length > 0 && (
                <div className="results-section">
                  <h3 style={{ marginLeft: '1rem', marginTop: '1rem', color: '#888' }}>✨ Kết quả LLM Re-ranking (Gemini 3.1 Flash Lite)</h3>
                  <ResultGrid 
                    results={pageLlmResults} 
                    onCardDoubleClick={handleCardDoubleClick} 
                    onCardClick={(url) => setSelectedImage(url)} 
                  />
                  <hr style={{ margin: '2rem 1rem', borderColor: '#333' }} />
                  <h3 style={{ marginLeft: '1rem', color: '#888' }}>🔍 Kết quả RRF (Khoảng cách Vector)</h3>
                </div>
              )}
              <ResultGrid 
                results={pageResults} 
                onCardDoubleClick={handleCardDoubleClick} 
                onCardClick={(url) => setSelectedImage(url)} 
              />
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

          {selectedImage && (
            <div className="image-modal-overlay" onClick={() => setSelectedImage(null)}>
              <div className="image-modal-content" onClick={(e) => e.stopPropagation()}>
                <button className="image-modal-close" onClick={() => setSelectedImage(null)}>&times;</button>
                <img src={selectedImage} alt="Enlarged view" className="image-modal-img" />
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
