import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SearchBar from './SearchBar.jsx';


describe('Temporal Events form', () => {
  it('searches candidate videos first with configurable summary/KIS weights', async () => {
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
    fireEvent.change(
      screen.getByRole('slider', { name: /temporal kis text weight/i }),
      { target: { value: '35' } },
    );
    await user.click(container.querySelector('button[type="submit"]'));

    expect(onTemporalVideoSearch).toHaveBeenCalledWith(
      'Video múa lân trên cột cao\nE1: Lân chào ban giám khảo',
      { summaryWeight: 0.6, kisWeight: 0.4, textWeight: 0.35, visualWeight: 0.65 },
      { batchIds: [], videoIds: [] },
      false,
    );
  });

  it('searches videos from aggregate event evidence without a summary', async () => {
    const user = userEvent.setup();
    const onTemporalVideoSearch = vi.fn();
    const { container } = render(
      <SearchBar onTemporalVideoSearch={onTemporalVideoSearch} />,
    );

    await user.click(screen.getByRole('tab', { name: /temporal events/i }));
    await user.click(screen.getByRole('button', { name: /chỉ events/i }));
    expect(screen.queryByLabelText('Video context')).not.toBeInTheDocument();
    await user.type(screen.getByLabelText('Event 1'), 'người đầu bếp cho dầu vào chảo');
    await user.click(container.querySelector('button[type="submit"]'));

    expect(onTemporalVideoSearch).toHaveBeenCalledWith(
      'E1: người đầu bếp cho dầu vào chảo',
      { summaryWeight: 0, kisWeight: 1, textWeight: 0.5, visualWeight: 0.5 },
      { batchIds: [], videoIds: [] },
      false,
    );
  });

  it('lets the user opt in to Gemini reranking for every event', async () => {
    const user = userEvent.setup();
    const onTemporalVideoSearch = vi.fn();
    const { container } = render(
      <SearchBar onTemporalVideoSearch={onTemporalVideoSearch} />,
    );

    await user.click(screen.getByRole('tab', { name: /temporal events/i }));
    await user.type(screen.getByLabelText('Video context'), 'Video múa lân trên cột cao');
    await user.type(screen.getByLabelText('Event 1'), 'Lân chào ban giám khảo');
    await user.click(screen.getByRole('checkbox', { name: /gemini re-rank từng event/i }));
    await user.click(container.querySelector('button[type="submit"]'));

    expect(onTemporalVideoSearch).toHaveBeenCalledWith(
      'Video múa lân trên cột cao\nE1: Lân chào ban giám khảo',
      { summaryWeight: 0.75, kisWeight: 0.25, textWeight: 0.5, visualWeight: 0.5 },
      { batchIds: [], videoIds: [] },
      true,
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
    fireEvent.change(
      screen.getByRole('slider', { name: /temporal kis text weight/i }),
      { target: { value: '25' } },
    );
    await user.click(screen.getByRole('button', { name: /tìm sự kiện trong L24_V033/i }));

    expect(onTemporalEventSearch).toHaveBeenCalledWith(
      'Video múa lân trên cột cao\nE1: Khoảnh khắc lân bắt đầu xoay',
      'L24_V033',
      { batchIds: [], videoIds: [] },
      { textWeight: 0.25, visualWeight: 0.75 },
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
