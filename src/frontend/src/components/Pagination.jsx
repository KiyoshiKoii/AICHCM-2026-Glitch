function getPageItems(currentPage, totalPages) {
  if (totalPages <= 7) {
    return Array.from({ length: totalPages }, (_, i) => i + 1);
  }

  const pages = new Set([
    1,
    2,
    totalPages - 1,
    totalPages,
    currentPage - 1,
    currentPage,
    currentPage + 1,
  ]);
  const sorted = [...pages].filter((p) => p >= 1 && p <= totalPages).sort((a, b) => a - b);

  const items = [];
  let previous = 0;
  for (const page of sorted) {
    if (page - previous > 1) {
      items.push(`ellipsis-${page}`);
    }
    items.push(page);
    previous = page;
  }
  return items;
}

function Pagination({ currentPage, totalPages, onPageChange }) {
  if (!totalPages || totalPages <= 1) return null;

  const items = getPageItems(currentPage, totalPages);

  return (
    <nav className="pagination" aria-label="Pagination">
      <button
        type="button"
        className="pagination-nav"
        disabled={currentPage === 1}
        onClick={() => onPageChange(currentPage - 1)}
      >
        Previous
      </button>
      {items.map((item) =>
        typeof item === 'number' ? (
          <button
            key={item}
            type="button"
            className={
              item === currentPage ? 'pagination-page active' : 'pagination-page'
            }
            aria-current={item === currentPage ? 'page' : undefined}
            onClick={() => onPageChange(item)}
          >
            {item}
          </button>
        ) : (
          <span key={item} className="pagination-ellipsis">
            …
          </span>
        )
      )}
      <button
        type="button"
        className="pagination-nav"
        disabled={currentPage === totalPages}
        onClick={() => onPageChange(currentPage + 1)}
      >
        Next
      </button>
    </nav>
  );
}

export default Pagination;
