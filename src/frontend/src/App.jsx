import { useState } from 'react';
import SearchBar from './components/SearchBar.jsx';
import ResultGrid from './components/ResultGrid.jsx';
import { fetchSearchResults } from './api/mockClient.js';
import './App.css';

function App() {
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const runSearch = async (input) => {
    setIsLoading(true);
    try {
      const response = await fetchSearchResults(input);
      setResults(response.data.results);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <SearchBar onSearch={runSearch} onImageSearch={runSearch} />
      {isLoading ? <p>Đang tìm kiếm...</p> : <ResultGrid results={results} />}
    </div>
  );
}

export default App;
