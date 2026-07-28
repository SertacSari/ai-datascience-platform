export default function Pager({ currentPage, onPage, pageSize, total, totalPages }) {
  const first = total === 0 ? 0 : (currentPage - 1) * pageSize + 1;
  const last = Math.min(currentPage * pageSize, total);
  return (
    <div className="pager">
      <span>{first}-{last} of {total} reports</span>
      <div>
        <button disabled={currentPage === 1} onClick={() => onPage(currentPage - 1)} type="button">&lt;</button>
        {Array.from({ length: totalPages }, (_, index) => index + 1).map((pageNumber) => (
          <button className={pageNumber === currentPage ? "active" : ""} key={pageNumber} onClick={() => onPage(pageNumber)} type="button">
            {pageNumber}
          </button>
        ))}
        <button disabled={currentPage === totalPages} onClick={() => onPage(currentPage + 1)} type="button">&gt;</button>
      </div>
    </div>
  );
}
