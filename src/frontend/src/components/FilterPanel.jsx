import { useState } from 'react';

const OBJECT_OPTIONS = ['Object 1', 'Object 2', 'Object 3'];

let nextRowId = 1;

function FilterPanel({ onSearch, canSearch }) {
  const [rows, setRows] = useState([{ id: nextRowId++, object: OBJECT_OPTIONS[0], value: '' }]);

  const addRow = () => {
    setRows((current) => [
      ...current,
      { id: nextRowId++, object: OBJECT_OPTIONS[0], value: '' },
    ]);
  };

  const removeRow = (id) => {
    setRows((current) => current.filter((row) => row.id !== id));
  };

  const updateRow = (id, changes) => {
    setRows((current) =>
      current.map((row) => (row.id === id ? { ...row, ...changes } : row))
    );
  };

  return (
    <section className="filter-panel">
      <h2 className="filter-panel-title">Filter</h2>
      <div className="filter-rows">
        {rows.map((row) => (
          <div className="filter-row" key={row.id}>
            <select
              className="filter-object"
              value={row.object}
              onChange={(e) => updateRow(row.id, { object: e.target.value })}
            >
              {OBJECT_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
            <input
              className="filter-value"
              type="text"
              placeholder="Enter value"
              value={row.value}
              onChange={(e) => updateRow(row.id, { value: e.target.value })}
            />
            <button
              type="button"
              className="filter-remove"
              aria-label="Remove filter"
              onClick={() => removeRow(row.id)}
            >
              ×
            </button>
          </div>
        ))}
      </div>
      <div className="filter-actions">
        <button type="button" className="btn btn-outline" onClick={addRow}>
          Add
        </button>
        <button
          type="button"
          className="btn btn-primary"
          disabled={!canSearch}
          onClick={() => onSearch?.()}
        >
          Search
        </button>
      </div>
    </section>
  );
}

export default FilterPanel;
