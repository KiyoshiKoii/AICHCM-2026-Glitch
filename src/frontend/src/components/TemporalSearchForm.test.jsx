import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SearchBar from './SearchBar.jsx';


const defaultEvent = {
  eventId: 'E1',
  textWeight: 0.5,
  visualWeight: 0.5,
  useRerank: false,
  requiresAfterPrevious: false,
  verifyCameraMotion: false,
  motionWeight: 0.7,
};

describe('Temporal Events form', () => {
  it('sends summary/KIS video weights and per-event KIS controls', async () => {
    const user = userEvent.setup();
    const onTemporalVideoSearch = vi.fn();
    const { container } = render(
      <SearchBar onTemporalVideoSearch={onTemporalVideoSearch} />,
    );

    await user.click(screen.getByRole('tab', { name: /temporal events/i }));
    await user.type(screen.getByLabelText('Video context'), 'Video mua lan tren cot cao');
    await user.type(screen.getByLabelText('Event 1'), 'Lan chao ban giam khao');
    fireEvent.change(
      screen.getByRole('slider', { name: /summary video ranking weight/i }),
      { target: { value: '60' } },
    );
    await user.click(screen.getByRole('button', { name: /advanced settings for event 1/i }));
    fireEvent.change(
      screen.getByRole('slider', { name: /event 1 kis text weight/i }),
      { target: { value: '35' } },
    );
    await user.click(screen.getByRole('checkbox', { name: /gemini re-rank this event/i }));
    await user.click(container.querySelector('button[type="submit"]'));

    expect(onTemporalVideoSearch).toHaveBeenCalledWith(
      'Video mua lan tren cot cao\nE1: Lan chao ban giam khao',
      { summaryWeight: 0.6, kisWeight: 0.4 },
      { batchIds: [], videoIds: [] },
      [{
        ...defaultEvent,
        textWeight: 0.35,
        visualWeight: 0.65,
        useRerank: true,
      }],
    );
  });

  it('searches videos from aggregate event evidence without a summary', async () => {
    const user = userEvent.setup();
    const onTemporalVideoSearch = vi.fn();
    const { container } = render(
      <SearchBar onTemporalVideoSearch={onTemporalVideoSearch} />,
    );

    await user.click(screen.getByRole('tab', { name: /temporal events/i }));
    await user.click(screen.getByText(/Chỉ Events/i));
    expect(screen.queryByLabelText('Video context')).not.toBeInTheDocument();
    await user.type(screen.getByLabelText('Event 1'), 'nguoi dau bep cho dau vao chao');
    await user.click(container.querySelector('button[type="submit"]'));

    expect(onTemporalVideoSearch).toHaveBeenCalledWith(
      'E1: nguoi dau bep cho dau vao chao',
      { summaryWeight: 0, kisWeight: 1 },
      { batchIds: [], videoIds: [] },
      [defaultEvent],
    );
  });

  it('scopes event search to the selected video and keeps each event option', async () => {
    const user = userEvent.setup();
    const onTemporalEventSearch = vi.fn();
    const { container } = render(
      <SearchBar
        selectedTemporalVideo={{ video_id: 'L24_V033' }}
        onTemporalEventSearch={onTemporalEventSearch}
      />,
    );

    await user.click(screen.getByRole('tab', { name: /temporal events/i }));
    await user.type(screen.getByLabelText('Event 1'), 'Lan bat dau xoay');
    await user.click(screen.getByRole('button', { name: /add event/i }));
    await user.type(screen.getByLabelText('Event 2'), 'Lan tiep dat');
    await user.click(screen.getByRole('button', { name: /advanced settings for event 2/i }));
    await user.click(screen.getByRole('checkbox', { name: /e2 must occur after e1/i }));
    await user.click(screen.getByRole('checkbox', { name: /verify shot transition/i }));
    fireEvent.change(
      screen.getByRole('slider', { name: /event 2 camera motion weight/i }),
      { target: { value: '65' } },
    );
    await user.click(container.querySelector('button[type="submit"]'));

    expect(onTemporalEventSearch).toHaveBeenCalledWith(
      'E1: Lan bat dau xoay\nE2: Lan tiep dat',
      'L24_V033',
      { batchIds: [], videoIds: [] },
      [
        defaultEvent,
        {
          ...defaultEvent,
          eventId: 'E2',
          requiresAfterPrevious: true,
          verifyCameraMotion: true,
          motionWeight: 0.65,
        },
      ],
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
