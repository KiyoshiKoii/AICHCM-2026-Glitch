import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitForElementToBeRemoved } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

vi.mock('./api/apiClient.js', () => ({
  searchByText: vi.fn(() => new Promise((resolve) => {
    setTimeout(() => resolve({
      data: {
        results: [{
          frame_id: 'L21_V001_f0001',
          video_name: 'L21_V001',
          frame_index: 1,
          score: 0.9,
          thumbnail_url: '/media/thumbnails/L21_V001_f0001.jpg',
        }],
        llm_reranked_results: [],
      },
    }), 25);
  })),
  searchByImage: vi.fn(),
  answerVqa: vi.fn(),
  getFrameContext: vi.fn(),
}));

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
