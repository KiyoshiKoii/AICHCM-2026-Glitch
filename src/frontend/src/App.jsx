import { useRef, useState } from 'react';
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
  searchTemporalEvents,
  searchTemporalVideos,
  getFrameContext,
} from './api/apiClient.js';
import './App.css';

const PAGE_SIZE = 15;

function App() {
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
  const [temporalVideoCandidates, setTemporalVideoCandidates] = useState([]);
  const [selectedTemporalVideo, setSelectedTemporalVideo] = useState(null);
  const activeSearchController = useRef(null);

  const beginSearch = () => {
    activeSearchController.current?.abort();
    const controller = new AbortController();
    activeSearchController.current = controller;
    setIsLoading(true);
    return controller;
  };

  const finishSearch = (controller) => {
    if (activeSearchController.current === controller) {
      activeSearchController.current = null;
      setIsLoading(false);
    }
  };

  const isAborted = (error) => error?.name === 'AbortError';

  const cancelSearch = () => {
    activeSearchController.current?.abort();
    activeSearchController.current = null;
    setIsLoading(false);
  };

  const runSearch = async (
    input,
    useRerank = false,
    weights = { textWeight: 0.5, visualWeight: 0.5 },
    filters = searchFilters,
  ) => {
    const controller = beginSearch();
    setCurrentPage(1);
    setLastQuery({ mode: 'text', query: input, useRerank, weights, filters });
    setVqaQuestion(null);
    setSelectedFrame(null);
    setTemporalVideoCandidates([]);
    setSelectedTemporalVideo(null);
    try {
      const response = await searchByText(
        input,
        100,
        useRerank,
        weights,
        filters,
        controller.signal,
      );
      if (activeSearchController.current !== controller) return;
      setResults(response.data.results || []);
      setLlmResults(response.data.llm_reranked_results || []);
    } catch (err) {
      if (isAborted(err)) return;
      console.error(err);
      alert('Lỗi trong quá trình tìm kiếm! Xem console để biết thêm chi tiết.');
    } finally {
      finishSearch(controller);
    }
  };

  const runVqaSearch = async ({ query, question, useRerank = false }) => {
    const controller = beginSearch();
    setCurrentPage(1);
    setLastQuery({ mode: 'vqa', query, question, useRerank });
    setVqaQuestion(question);
    setSelectedFrame(null);
    setTemporalVideoCandidates([]);
    setSelectedTemporalVideo(null);
    setLlmResults([]);
    setResults([]);
    try {
      const response = await answerVqa(query, question, 50, 10, useRerank, controller.signal);
      if (activeSearchController.current !== controller) return;
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
      if (isAborted(err)) return;
      console.error(err);
      alert('Lỗi trong quá trình trả lời VQA! Xem console để biết thêm chi tiết.');
    } finally {
      finishSearch(controller);
    }
  };

  const runTemporalVideoSearch = async (
    query,
    weights = {
      summaryWeight: 0.75,
      kisWeight: 0.25,
      textWeight: 0.5,
      visualWeight: 0.5,
    },
    filters = searchFilters,
    eventOptions = [],
  ) => {
    const controller = beginSearch();
    setCurrentPage(1);
    setLastQuery({ mode: 'temporal-video', query, weights, filters, eventOptions });
    setVqaQuestion(null);
    setSelectedFrame(null);
    setLlmResults([]);
    setResults([]);
    setFrameContext(null);
    setSelectedTemporalVideo(null);
    setTemporalVideoCandidates([]);
    try {
      const response = await searchTemporalVideos(
        query,
        weights,
        filters,
        eventOptions,
        controller.signal,
      );
      if (activeSearchController.current !== controller) return;
      const data = response.data || {};
      setTemporalVideoCandidates(data.candidates || data.video_selection?.candidates || []);
    } catch (err) {
      if (isAborted(err)) return;
      console.error(err);
      alert('Không thể tìm video TRAKE. Kiểm tra console để biết thêm chi tiết.');
    } finally {
      finishSearch(controller);
    }
  };

  const runAsrSearch = async (query, filters = searchFilters) => {
    const controller = beginSearch();
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
      const response = await searchAsr(query, 50, filters, controller.signal);
      if (activeSearchController.current !== controller) return;
      setResults(response.data?.results || []);
    } catch (err) {
      if (isAborted(err)) return;
      console.error(err);
      alert('Không thể tìm kiếm ASR. Hãy kiểm tra Elasticsearch và ASR index.');
    } finally {
      finishSearch(controller);
    }
  };

  const openVideoSearch = ({ batchId, videoId, frameIndex = null }) => {
    const normalizedVideoId = `${batchId}_${videoId}`;
    const timelineAnchorId = `${normalizedVideoId}_f0001`;
    const hasRequestedFrame = Number.isInteger(frameIndex) && frameIndex >= 0;
    activeSearchController.current?.abort();
    activeSearchController.current = null;
    setIsLoading(false);
    setCurrentPage(1);
    setLastQuery({
      mode: 'video-search',
      videoId: normalizedVideoId,
      frameIndex: hasRequestedFrame ? frameIndex : null,
    });
    setVqaQuestion(null);
    setFrameContext(null);
    setTemporalVideoCandidates([]);
    setSelectedTemporalVideo(null);
    setLlmResults([]);
    setResults([]);
    setSelectedFrame({
      frame_id: timelineAnchorId,
      video_name: normalizedVideoId,
      frame_index: hasRequestedFrame ? frameIndex : 0,
      thumbnail_url: `/media/thumbnails/${timelineAnchorId}.jpg`,
      metadata: hasRequestedFrame
        ? { seek_frame_index: frameIndex }
        : { timestamp_ms: 0, seek_timestamp_ms: 0 },
    });
  };

  const runTemporalEventSearch = async (
    query,
    videoId,
    filters = searchFilters,
    eventOptions = [],
  ) => {
    const controller = beginSearch();
    setCurrentPage(1);
    const scopedFilters = { ...filters, videoIds: [videoId] };
    setLastQuery({ mode: 'temporal-event', query, filters: scopedFilters, eventOptions });
    setVqaQuestion(null);
    setSelectedFrame(null);
    setLlmResults([]);
    setResults([]);
    try {
      const response = await searchTemporalEvents(
        query,
        scopedFilters,
        { textWeight: 0.5, visualWeight: 0.5 },
        eventOptions,
        controller.signal,
      );
      if (activeSearchController.current !== controller) return;
      const data = response.data || {};
      const temporalResults = (data.events || []).map((item) => ({
        ...item,
        video_name: item.video_id || data.selected_video?.video_id || videoId,
        frame_index: item.native_frame_idx,
        score: item.video_score ?? item.score,
        thumbnail_url: item.thumbnail_url || `/media/thumbnails/${item.frame_id}.jpg`,
        metadata: {
          event_id: item.event_id,
          event_description: item.description,
          rank: item.rank,
          timestamp_ms: item.timestamp_ms,
          anchor_type: item.anchor_type,
          confidence: item.confidence,
          event_score: item.score,
          reason_vi: item.reason_vi,
          matched_context_entities: item.matched_context_entities,
          caption: item.caption,
           source_ranks: item.source_ranks,
           camera_motion: item.camera_motion,
           event_order: item.event_order,
         },
      }));
      setResults(temporalResults);
    } catch (err) {
      if (isAborted(err)) return;
      console.error(err);
      alert(`Không thể tìm sự kiện trong ${videoId}. Kiểm tra console để biết thêm chi tiết.`);
    } finally {
      finishSearch(controller);
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

  const totalPages = Math.ceil(results.length / PAGE_SIZE) || 1;
  const pageResults = results.slice(
    (currentPage - 1) * PAGE_SIZE,
    currentPage * PAGE_SIZE
  );
  const pageLlmResults = llmResults.slice(
    (currentPage - 1) * PAGE_SIZE,
    currentPage * PAGE_SIZE
  );
  const isTemporalEventWorkspace = Boolean(selectedTemporalVideo);

  return (
    <div className="app">
      <Navbar />
      <div className="app-body">
        <Sidebar
          onSearch={runSearch}
          onVqaSearch={runVqaSearch}
          onAsrSearch={runAsrSearch}
          onTemporalVideoSearch={runTemporalVideoSearch}
          onTemporalEventSearch={runTemporalEventSearch}
          onVideoSearch={openVideoSearch}
          selectedTemporalVideo={selectedTemporalVideo}
          onClearTemporalVideo={clearTemporalVideo}
          filters={searchFilters}
          onFiltersChange={setSearchFilters}
        />
        <main className="main-content">
          {isTemporalEventWorkspace ? (
            <section className="temporal-event-workspace" aria-label="Temporal event workspace">
              <header className="temporal-event-workspace-header">
                <button
                  type="button"
                  className="temporal-event-back"
                  onClick={clearTemporalVideo}
                >
                  ← Quay lại chọn video
                </button>
                <div>
                  <span className="temporal-step-kicker">Bước 2 · Tìm sự kiện</span>
                  <h1>{selectedTemporalVideo.video_id}</h1>
                  <p>Nhập E1…En ở thanh bên trái, rồi tìm KIS chỉ trong video này.</p>
                </div>
              </header>
              {isLoading ? (
                <LoadingSpinner label="Đang tìm frame KIS trong video..." onCancel={cancelSearch} />
              ) : results.length > 0 ? (
                <>
                  <section className="temporal-event-results-heading" aria-label="Temporal KIS results">
                    <span className="temporal-step-kicker">KIS visual + caption</span>
                    <h2>Ứng viên theo từng sự kiện</h2>
                    <p>Click để xem thông tin frame, double-click để mở video đúng vị trí.</p>
                  </section>
                  <ResultGrid
                    results={results}
                    onCardDoubleClick={handleCardDoubleClick}
                    onCardClick={setSelectedFrame}
                    groupTemporalEvents
                  />
                </>
              ) : (
                <div className="temporal-event-empty">
                  <strong>Sẵn sàng tìm event</strong>
                  <span>Thêm mô tả E1…En ở thanh bên trái rồi bấm nút tìm sự kiện.</span>
                </div>
              )}
            </section>
          ) : isLoading ? (
            <LoadingSpinner label="Đang tìm kiếm..." onCancel={cancelSearch} />
          ) : (
            <>
              {!selectedTemporalVideo && (
                <TemporalVideoCandidates
                  candidates={temporalVideoCandidates}
                  selectedVideoId={selectedTemporalVideo?.video_id}
                  onSelect={selectTemporalVideo}
                />
              )}
              {lastQuery?.mode === 'temporal-event' && results.length > 0 && (
                <section className="temporal-event-results-heading" aria-label="Temporal KIS results">
                  <span className="temporal-step-kicker">Bước 2 · KIS trong video đã chọn</span>
                  <h2>Ứng viên cho từng sự kiện trong {selectedTemporalVideo?.video_id}</h2>
                  <p>Click để xem thông tin frame, double-click để mở video đúng vị trí.</p>
                </section>
              )}
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
                groupTemporalEvents={lastQuery?.mode === 'temporal-event'}
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


