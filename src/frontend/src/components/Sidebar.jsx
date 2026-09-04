import SearchBar from './SearchBar.jsx';
import FilterPanel from './FilterPanel.jsx';

function Sidebar({
  onSearch,
  onVqaSearch,
  onAsrSearch,
  onTemporalVideoSearch,
  onTemporalEventSearch,
  onVideoSearch,
  selectedTemporalVideo,
  onClearTemporalVideo,
  onFiltersChange,
  filters,
}) {
  return (
    <aside className="sidebar">
      <SearchBar
        onSearch={onSearch}
        onVqaSearch={onVqaSearch}
        onAsrSearch={onAsrSearch}
        onTemporalVideoSearch={onTemporalVideoSearch}
        onTemporalEventSearch={onTemporalEventSearch}
        onVideoSearch={onVideoSearch}
        selectedTemporalVideo={selectedTemporalVideo}
        onClearTemporalVideo={onClearTemporalVideo}
        filters={filters}
      />

      <FilterPanel onFiltersChange={onFiltersChange} />
    </aside>
  );
}

export default Sidebar;


