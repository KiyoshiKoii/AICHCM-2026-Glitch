import { useEffect, useState } from 'react';

const BATCHES = [
  { id: 'L21', label: 'Tin sáng', detail: '60 Giây Sáng – HTV' },
  { id: 'L22', label: 'Tin chiều', detail: '60 Giây Chiều – HTV' },
  { id: 'L23', label: 'Đua xe đạp', detail: 'Cúp Truyền Hình TP.HCM 2024' },
  { id: 'L24', label: 'Lân Sư Rồng', detail: 'Cúp Chợ Lớn HTV 2024' },
  { id: 'L25', label: 'Ôn thi THPT', detail: 'Bí quyết ôn thi THPT 2024' },
  { id: 'L26', label: 'Ẩm thực', detail: 'ViVU TV – Món ngon mỗi ngày' },
  { id: 'L27', label: 'Du lịch & văn hóa', detail: 'Việt Nam Đi Là Ghiền' },
  { id: 'L28', label: 'Khám phá Mê Kông', detail: 'Tản Mạn Mê Kông' },
  { id: 'L29', label: 'Đời sống miền Tây', detail: 'Đôi Mắt Mê Kông' },
  { id: 'L30', label: 'Phóng sự cộng đồng', detail: 'Lan tỏa năng lượng tích cực 2024' },
];

function parseVideoIds(value) {
  return value
    .split(/[,;\s]+/)
    .map((item) => item.trim().toUpperCase())
    .map((item) => (/^\d{1,3}$/.test(item) ? `V${Number(item).toString().padStart(3, '0')}` : item))
    .filter(Boolean);
}

function FilterPanel({ onFiltersChange, resetKey = 0 }) {
  const [selectedBatches, setSelectedBatches] = useState([]);
  const [videoInput, setVideoInput] = useState('');

  useEffect(() => {
    setSelectedBatches([]);
    setVideoInput('');
  }, [resetKey]);

  const toggleBatch = (batchId) => {
    setSelectedBatches((current) => {
      const next = current.includes(batchId)
        ? current.filter((item) => item !== batchId)
        : [...current, batchId];
      onFiltersChange?.({ batchIds: next, videoIds: parseVideoIds(videoInput) });
      return next;
    });
  };

  const clearFilters = () => {
    setSelectedBatches([]);
    setVideoInput('');
    onFiltersChange?.({ batchIds: [], videoIds: [] });
  };

  return (
    <section className="filter-panel">
      <div className="filter-panel-heading">
        <h2 className="filter-panel-title">Filter</h2>
        <button type="button" className="filter-clear" onClick={clearFilters}>
          Clear
        </button>
      </div>

      <div className="batch-filter-grid" aria-label="Filter by video batch">
        {BATCHES.map((batch) => (
          <label
            className={`batch-filter-chip${selectedBatches.includes(batch.id) ? ' selected' : ''}`}
            key={batch.id}
            title={`${batch.id}: ${batch.label} – ${batch.detail}`}
          >
            <input
              type="checkbox"
              checked={selectedBatches.includes(batch.id)}
              onChange={() => toggleBatch(batch.id)}
            />
            <span className="batch-filter-code">{batch.id}</span>
            <span className="batch-filter-label">{batch.label}</span>
          </label>
        ))}
      </div>

      <label className="video-filter-field">
        <span>Video ID</span>
        <input
          type="text"
          value={videoInput}
          onChange={(e) => {
            const value = e.target.value;
            setVideoInput(value);
            onFiltersChange?.({ batchIds: selectedBatches, videoIds: parseVideoIds(value) });
          }}
          placeholder="Nhập số video, ví dụ: 6, 40"
          aria-label="Video ID filter"
        />
      </label>
      <p className="filter-hint">Nhập một hoặc nhiều số video, cách nhau bằng dấu phẩy.</p>

    </section>
  );
}

export default FilterPanel;
