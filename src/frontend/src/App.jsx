import { useState } from 'react';
import Navbar from './components/Navbar.jsx';
import Sidebar from './components/Sidebar.jsx';
import ResultGrid from './components/ResultGrid.jsx';
import Pagination from './components/Pagination.jsx';
import TimelineViewer from './components/TimelineViewer.jsx';
import LoadingSpinner from './components/LoadingSpinner.jsx';
import { answerVqa, searchByText, searchByImage, searchTemporalEvents, getFrameContext } from './api/apiClient.js';
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
  const [vqaQuestion, setVqaQuestion] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [searchFilters, setSearchFilters] = useState({ batchIds: [], videoIds: [] });
  const [filterResetKey, setFilterResetKey] = useState(0);

  const runSearch = async (
    input,
    useRerank = false,
    weights = { textWeight: 0.5, visualWeight: 0.5 },
    filters = searchFilters,
  ) => {
    setIsLoading(true);
    setCurrentPage(1);
    setLastQuery(
      input instanceof File
        ? input
        : { mode: 'text', query: input, useRerank, weights, filters },
    );
    setVqaQuestion(null);
    try {
      let response;
      if (input instanceof File) {
        response = await searchByImage(input);
      } else {
        response = await searchByText(input, 100, useRerank, weights, filters);
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

  const runVqaSearch = async ({ query, question }) => {
    setIsLoading(true);
    setCurrentPage(1);
    setLastQuery({ mode: 'vqa', query, question });
    setVqaQuestion(question);
    setLlmResults([]);
    setResults([]);
    try {
      const response = await answerVqa(query, question);
      setResults(response.data.candidates || []);
    } catch (err) {
      console.error(err);
      alert('Lỗi trong quá trình trả lời VQA! Xem console để biết thêm chi tiết.');
    } finally {
      setIsLoading(false);
    }
  };

  const runTemporalSearch = async (query, filters = searchFilters) => {
    setIsLoading(true);
    setCurrentPage(1);
    setLastQuery({ mode: 'temporal', query, filters });
    setVqaQuestion(null);
    setLlmResults([]);
    setResults([]);
    try {
      const response = await searchTemporalEvents(query, filters);
      const data = response.data || {};
      const events = (data.events || []).map((event) => ({
        ...event,
        video_name: data.selected_video?.video_id || event.video_id,
        frame_index: event.native_frame_idx,
        thumbnail_url: event.thumbnail_url || `/media/thumbnails/${event.frame_id}.jpg`,
        metadata: {
          timestamp_ms: event.timestamp_ms,
          anchor_type: event.anchor_type,
          confidence: event.confidence,
          reason_vi: event.reason_vi,
        },
      }));
      setResults(events);
    } catch (err) {
      console.error(err);
      alert('Temporal event search failed. Check the console for details.');
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
    setVqaQuestion(null);
    setSearchFilters({ batchIds: [], videoIds: [] });
    setFilterResetKey((current) => current + 1);
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
          onVqaSearch={runVqaSearch}
          onTemporalSearch={runTemporalSearch}
          filters={searchFilters}
          onFiltersChange={setSearchFilters}
          filterResetKey={filterResetKey}
        />
        <main className="main-content">
          {isLoading ? (
            <LoadingSpinner label="Đang tìm kiếm..." />
          ) : (
            <>
              {vqaQuestion && (
                <div className="vqa-results-heading">
                  <h3>🤖 VQA answers from Gemini</h3>
                  <p>{vqaQuestion}</p>
                </div>
              )}
              {results.length > 0 && (
                <div className="results-toolbar">
                  <span className="results-count">
                    {results.length} kết quả · trang {currentPage}/{totalPages}
                  </span>
                  <Pagination
                    currentPage={currentPage}
                    totalPages={totalPages}
                    onPageChange={setCurrentPage}
                  />
                </div>
              )}
              {llmResults.length > 0 && (
                <div className="results-section">
                  <h3 style={{ marginLeft: '1rem', marginTop: '1rem', color: '#888' }}>✨ Kết quả LLM Re-ranking (Gemini)</h3>
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
