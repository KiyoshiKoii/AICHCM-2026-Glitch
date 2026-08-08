import SearchBar from './SearchBar.jsx';
import FilterPanel from './FilterPanel.jsx';

const DATASETS = ['V3C1'];

function Sidebar({
  dataset,
  onDatasetChange,
  onNewSearch,
  onSearch,
  onImageSearch,
  onVqaSearch,
  onRepeatSearch,
  canRepeatSearch,
}) {
  return (
    <aside className="sidebar">
      <button type="button" className="btn btn-outline new-search-btn" onClick={onNewSearch}>
        New search
      </button>

      <div className="sidebar-field">
        <label htmlFor="dataset-select">Dataset</label>
        <select
          id="dataset-select"
          value={dataset}
          onChange={(e) => onDatasetChange?.(e.target.value)}
        >
          {DATASETS.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </div>

      <SearchBar
        onSearch={onSearch}
        onImageSearch={onImageSearch}
        onVqaSearch={onVqaSearch}
      />

      <FilterPanel onSearch={onRepeatSearch} canSearch={canRepeatSearch} />
    </aside>
  );
}

export default Sidebar;
