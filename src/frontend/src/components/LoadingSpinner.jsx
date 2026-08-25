function LoadingSpinner({ label, onCancel }) {
  return (
    <div className="loading-spinner" role="status">
      <div className="loading-spinner-icon" />
      {label && <span>{label}</span>}
      {onCancel && (
        <button type="button" onClick={onCancel}>Hủy</button>
      )}
    </div>
  );
}

export default LoadingSpinner;
