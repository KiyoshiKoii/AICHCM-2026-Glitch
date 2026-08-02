function Navbar() {
  return (
    <header className="navbar">
      <div className="navbar-brand">
        <span className="navbar-logo" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
            <path d="M12 21s-6.7-4.35-9.3-8.1C.86 10.13 1.3 6.6 4.1 4.9c2.3-1.4 4.9-.7 6.6 1.2L12 7.5l1.3-1.4c1.7-1.9 4.3-2.6 6.6-1.2 2.8 1.7 3.24 5.23 1.4 8-2.6 3.75-9.3 8.1-9.3 8.1z" />
          </svg>
        </span>
        <span className="navbar-title">AIC Glitch</span>
      </div>
      <button type="button" className="navbar-login">
        Login
      </button>
    </header>
  );
}

export default Navbar;
