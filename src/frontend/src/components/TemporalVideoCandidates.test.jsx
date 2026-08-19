import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import TemporalVideoCandidates from './TemporalVideoCandidates.jsx';


describe('TemporalVideoCandidates', () => {
  it('previews and selects a candidate video', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    const candidate = {
      video_id: 'L24_V033',
      score: 0.91,
      summary_score: 0.95,
      kis_score: 0.72,
      summary_vi: 'Múa lân trên hệ thống cột cao.',
    };

    render(<TemporalVideoCandidates candidates={[candidate]} onSelect={onSelect} />);

    expect(screen.getByLabelText('Preview L24_V033')).toHaveAttribute(
      'src',
      '/media/videos/L24_V033.mp4',
    );
    await user.click(screen.getByRole('button', { name: /chọn để tìm sự kiện/i }));
    expect(onSelect).toHaveBeenCalledWith(candidate);
  });
});
