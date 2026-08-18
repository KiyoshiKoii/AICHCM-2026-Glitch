import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SearchBar from './SearchBar.jsx';


describe('Temporal Events form', () => {
  it('searches candidate videos first with configurable summary/event weights', async () => {
    const user = userEvent.setup();
    const onTemporalVideoSearch = vi.fn();
    const { container } = render(
      <SearchBar onTemporalVideoSearch={onTemporalVideoSearch} />,
    );

    await user.click(screen.getByRole('tab', { name: /temporal events/i }));
    await user.type(screen.getByLabelText('Video context'), 'Video múa lân trên cột cao');
    await user.type(screen.getByLabelText('Event 1'), 'Lân chào ban giám khảo');
    fireEvent.change(
      screen.getByRole('slider', { name: /summary video ranking weight/i }),
      { target: { value: '60' } },
    );
    await user.click(container.querySelector('button[type="submit"]'));

    expect(onTemporalVideoSearch).toHaveBeenCalledWith(
      'Video múa lân trên cột cao\nE1: Lân chào ban giám khảo',
      { summaryWeight: 0.6, eventWeight: 0.4 },
      { batchIds: [], videoIds: [] },
    );
  });

  it('scopes event search to the video selected by the user', async () => {
    const user = userEvent.setup();
    const onTemporalEventSearch = vi.fn();
    const { rerender } = render(
      <SearchBar
        onTemporalEventSearch={onTemporalEventSearch}
      />,
    );

    await user.click(screen.getByRole('tab', { name: /temporal events/i }));
    await user.type(screen.getByLabelText('Video context'), 'Video múa lân trên cột cao');
    rerender(
      <SearchBar
        selectedTemporalVideo={{ video_id: 'L24_V033' }}
        onTemporalEventSearch={onTemporalEventSearch}
      />,
    );
    expect(screen.getByText('L24_V033')).toBeInTheDocument();
    await user.type(screen.getByLabelText('Event 1'), 'Khoảnh khắc lân bắt đầu xoay');
    await user.click(screen.getByRole('button', { name: /tìm sự kiện trong L24_V033/i }));

    expect(onTemporalEventSearch).toHaveBeenCalledWith(
      'Video múa lân trên cột cao\nE1: Khoảnh khắc lân bắt đầu xoay',
      'L24_V033',
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
