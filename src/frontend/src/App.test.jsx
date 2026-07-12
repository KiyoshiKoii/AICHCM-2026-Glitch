import { describe, it, expect } from 'vitest';
import { render, screen, waitForElementToBeRemoved } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App.jsx';

describe('App loading states', () => {
  it('shows a spinner while searching, then reveals the results grid', async () => {
    const user = userEvent.setup();
    render(<App />);

    const searchInput = screen.getByPlaceholderText(/mô tả cảnh cần tìm/i);
    await user.type(searchInput, 'người đàn ông làm rơi ví');
    await user.click(screen.getByRole('button', { name: /tìm kiếm/i }));

    expect(screen.getByRole('status')).toBeInTheDocument();

    await waitForElementToBeRemoved(() => screen.queryByRole('status'), {
      timeout: 3000,
    });

    expect(screen.getAllByRole('img').length).toBeGreaterThan(0);
  });
});
