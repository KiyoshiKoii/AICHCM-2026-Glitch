import { useState } from 'react';
import Navbar from './components/Navbar.jsx';
import Sidebar from './components/Sidebar.jsx';
import ResultGrid from './components/ResultGrid.jsx';
import Pagination from './components/Pagination.jsx';
import TimelineViewer from './components/TimelineViewer.jsx';
import TemporalVideoCandidates from './components/TemporalVideoCandidates.jsx';
import VideoFrameModal from './components/VideoFrameModal.jsx';
import LoadingSpinner from './components/LoadingSpinner.jsx';
import {
  answerVqa,
  searchByText,
  searchAsr,
  searchByImage,
  searchTemporalEvents,
  searchTemporalVideos,
  getFrameContext,
} from './api/apiClient.js';
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
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [searchFilters, setSearchFilters] = useState({ batchIds: [], videoIds: [] });
  const [filterResetKey, setFilterResetKey] = useState(0);
  const [temporalVideoCandidates, setTemporalVideoCandidates] = useState([]);
  const [selectedTemporalVideo, setSelectedTemporalVideo] = useState(null);

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
    setSelectedFrame(null);
    setTemporalVideoCandidates([]);
    setSelectedTemporalVideo(null);
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

  const runVqaSearch = async ({ query, question, useRerank = false }) => {
    setIsLoading(true);
    setCurrentPage(1);
    setLastQuery({ mode: 'vqa', query, question, useRerank });
    setVqaQuestion(question);
    setSelectedFrame(null);
    setTemporalVideoCandidates([]);
    setSelectedTemporalVideo(null);
    setLlmResults([]);
    setResults([]);
    try {
      const response = await answerVqa(query, question, 50, 10, useRerank);
      const data = response.data || {};
      const answeredCandidates = data.candidates || [];
      const hasRerankedResults = Boolean(
        useRerank && data.llm_reranked_results?.length,
      );

      // When reranking is enabled, keep the answered candidates in the upper
      // Gemini section and show the complete RRF baseline underneath.  With
      // reranking disabled, the answered RRF candidates are the only section.
      setLlmResults(hasRerankedResults ? answeredCandidates : []);
      setResults(hasRerankedResults ? (data.results || []) : answeredCandidates);
    } catch (err) {
      console.error(err);
      alert('Lỗi trong quá trình trả lời VQA! Xem console để biết thêm chi tiết.');
    } finally {
      setIsLoading(false);
    }
  };

  const runTemporalVideoSearch = async (
    query,
    weights = { summaryWeight: 0.75, kisWeight: 0.25 },
    filters = searchFilters,
  ) => {
    setIsLoading(true);
    setCurrentPage(1);
    setLastQuery({ mode: 'temporal-video', query, weights, filters });
    setVqaQuestion(null);
    setSelectedFrame(null);
    setLlmResults([]);
    setResults([]);
    setFrameContext(null);
    setSelectedTemporalVideo(null);
    setTemporalVideoCandidates([]);
    try {
      const response = await searchTemporalVideos(query, weights, filters);
      const data = response.data || {};
      setTemporalVideoCandidates(data.candidates || data.video_selection?.candidates || []);
    } catch (err) {
      console.error(err);
      alert('Không thể tìm video TRAKE. Kiểm tra console để biết thêm chi tiết.');
    } finally {
      setIsLoading(false);
    }
  };

  const runAsrSearch = async (query, filters = searchFilters) => {
    setIsLoading(true);
    setCurrentPage(1);
    setLastQuery({ mode: 'asr', query, filters });
    setVqaQuestion(null);
    setSelectedFrame(null);
    setTemporalVideoCandidates([]);
    setSelectedTemporalVideo(null);
    setLlmResults([]);
    setResults([]);
    setFrameContext(null);
    try {
      const response = await searchAsr(query, 50, filters);
      setResults(response.data?.results || []);
    } catch (err) {
      console.error(err);
      alert('Không thể tìm kiếm ASR. Hãy kiểm tra Elasticsearch và ASR index.');
    } finally {
      setIsLoading(false);
    }
  };

  const runTemporalEventSearch = async (query, videoId, filters = searchFilters) => {
    setIsLoading(true);
    setCurrentPage(1);
    const scopedFilters = { ...filters, videoIds: [videoId] };
    setLastQuery({ mode: 'temporal-event', query, filters: scopedFilters });
    setVqaQuestion(null);
    setSelectedFrame(null);
    setLlmResults([]);
    setResults([]);
    try {
      const response = await searchTemporalEvents(query, scopedFilters);
      const data = response.data || {};
      const temporalResults = (data.events || []).map((item) => ({
        ...item,
        video_name: item.video_id || data.selected_video?.video_id || videoId,
        frame_index: item.native_frame_idx,
        score: item.video_score ?? item.score,
        thumbnail_url: item.thumbnail_url || `/media/thumbnails/${item.frame_id}.jpg`,
        metadata: {
          rank: item.rank,
          timestamp_ms: item.timestamp_ms,
          anchor_type: item.anchor_type,
          confidence: item.confidence,
          event_score: item.score,
          reason_vi: item.reason_vi,
          matched_context_entities: item.matched_context_entities,
        },
      }));
      setResults(temporalResults);
    } catch (err) {
      console.error(err);
      alert(`Không thể tìm sự kiện trong ${videoId}. Kiểm tra console để biết thêm chi tiết.`);
    } finally {
      setIsLoading(false);
    }
  };

  const selectTemporalVideo = (candidate) => {
    setSelectedTemporalVideo(candidate);
    setResults([]);
    setLlmResults([]);
    setFrameContext(null);
    setCurrentPage(1);
  };

  const clearTemporalVideo = () => {
    setSelectedTemporalVideo(null);
    setResults([]);
    setFrameContext(null);
    setCurrentPage(1);
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
    setSelectedFrame(null);
    setSearchFilters({ batchIds: [], videoIds: [] });
    setFilterResetKey((current) => current + 1);
    setTemporalVideoCandidates([]);
    setSelectedTemporalVideo(null);
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
          onAsrSearch={runAsrSearch}
          onTemporalVideoSearch={runTemporalVideoSearch}
          onTemporalEventSearch={runTemporalEventSearch}
          selectedTemporalVideo={selectedTemporalVideo}
          onClearTemporalVideo={clearTemporalVideo}
          filters={searchFilters}
          onFiltersChange={setSearchFilters}
          filterResetKey={filterResetKey}
        />
        <main className="main-content">
          {isLoading ? (
            <LoadingSpinner label="Đang tìm kiếm..." />
          ) : (
            <>
              <TemporalVideoCandidates
                candidates={temporalVideoCandidates}
                selectedVideoId={selectedTemporalVideo?.video_id}
                onSelect={selectTemporalVideo}
              />
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
              {llmResults.length > 0 && pageLlmResults.length > 0 && (
                <div className="results-section">
                  <h3 style={{ marginLeft: '1rem', marginTop: '1rem', color: '#888' }}>✨ Kết quả LLM Re-ranking (Gemini)</h3>
                  <ResultGrid 
                    results={pageLlmResults} 
                    onCardDoubleClick={handleCardDoubleClick} 
                    onCardClick={setSelectedFrame}
                  />
                  <hr style={{ margin: '2rem 1rem', borderColor: '#333' }} />
                  <h3 style={{ marginLeft: '1rem', color: '#888' }}>🔍 Kết quả RRF (Khoảng cách Vector)</h3>
                </div>
              )}
              <ResultGrid 
                results={pageResults} 
                onCardDoubleClick={handleCardDoubleClick} 
                onCardClick={setSelectedFrame}
              />
            </>
          )}
          {isContextLoading ? (
            <LoadingSpinner label="Đang tải ngữ cảnh..." />
          ) : (
            <TimelineViewer frameContext={frameContext} />
          )}

          <VideoFrameModal result={selectedFrame} onClose={() => setSelectedFrame(null)} />
        </main>
      </div>
    </div>
  );
}

export default App;
