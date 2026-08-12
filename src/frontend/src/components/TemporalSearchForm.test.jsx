import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SearchBar from './SearchBar.jsx';


describe('Temporal Events form', () => {
  it('serializes a video context and dynamically added events for the API', async () => {
    const user = userEvent.setup();
    const onTemporalSearch = vi.fn();
    const { container } = render(<SearchBar onTemporalSearch={onTemporalSearch} />);

    await user.click(screen.getByRole('tab', { name: /temporal events/i }));
    await user.type(
      screen.getByLabelText('Video context'),
      'Bản tin về thời tiết nóng tại Barcelona',
    );
    await user.type(
      screen.getByLabelText('Event 1'),
      'Khoảnh khắc đầu tiên thấy nhiệt độ kỷ lục',
    );
    await user.click(screen.getByRole('button', { name: /add event/i }));
    await user.type(
      screen.getByLabelText('Event 2'),
      'Khoảnh khắc thấy người dân tránh nóng',
    );
    await user.click(container.querySelector('button[type="submit"]'));

    expect(onTemporalSearch).toHaveBeenCalledWith(
      'Bản tin về thời tiết nóng tại Barcelona\nE1: Khoảnh khắc đầu tiên thấy nhiệt độ kỷ lục\nE2: Khoảnh khắc thấy người dân tránh nóng',
      { batchIds: [], videoIds: [] },
    );
  });

  it('renumbers remaining events after one is removed', async () => {
    const user = userEvent.setup();
    const { container } = render(<SearchBar />);

    await user.click(screen.getByRole('tab', { name: /temporal events/i }));
    await user.click(screen.getByRole('button', { name: /add event/i }));
    await user.click(screen.getByRole('button', { name: 'Remove event 1' }));

    expect(screen.getByLabelText('Event 1')).toBeInTheDocument();
    expect(screen.queryByLabelText('Event 2')).not.toBeInTheDocument();
    expect(container.querySelectorAll('.temporal-event-row')).toHaveLength(1);
  });
});
