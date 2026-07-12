function LoadingSpinner({ label }) {
  return (
    <div className="loading-spinner" role="status">
      <div className="loading-spinner-icon" />
      {label && <span>{label}</span>}
    </div>
  );
}

export default LoadingSpinner;
