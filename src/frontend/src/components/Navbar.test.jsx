import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Navbar from './Navbar.jsx';

describe('Navbar', () => {
  it('renders the brand title and a login button', () => {
    render(<Navbar />);
    expect(screen.getByText('AIC Glitch')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /login/i })).toBeInTheDocument();
  });
});
