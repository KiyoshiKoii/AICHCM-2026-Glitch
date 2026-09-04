import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import TemporalVideoCandidates from './TemporalVideoCandidates.jsx';

describe('TemporalVideoCandidates', () => {
  it('previews and selects a candidate video without frame evidence', async () => {
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

  it('uses the best E1 evidence by default', () => {
    render(<TemporalVideoCandidates candidates={[{
      video_id: 'L26_V001',
      score: 0.9,
      summary_score: 0,
      kis_score: 0.9,
      kis_evidence: [{ event_id: 'E1', frame_id: 'L26_V001_f0042' }],
    }]} />);

    expect(screen.getByAltText('E1 candidate for L26_V001')).toHaveAttribute(
      'src',
      '/media/thumbnails/L26_V001_f0042.jpg',
    );
  });

  it('keeps long summaries in a keyboard-focusable scroll region', () => {
    render(<TemporalVideoCandidates candidates={[{
      video_id: 'L26_V311',
      score: 0.9,
      summary_score: 0,
      kis_score: 0.9,
      summary_vi: 'Mô tả dài của video để người dùng có thể cuộn đọc toàn bộ nội dung.',
    }]} />);

    expect(screen.getByLabelText('Video summary for L26_V311')).toHaveAttribute('tabindex', '0');
  });

  it('cycles a video thumbnail through the best evidence for its events', async () => {
    const user = userEvent.setup();
    render(
      <TemporalVideoCandidates
        onSelect={vi.fn()}
        candidates={[{
          video_id: 'L24_V018',
          score: 0.9,
          summary_score: 0.5,
          kis_score: 1,
          kis_evidence: [
            {
              event_id: 'E1',
              frame_id: 'L24_V018_f0003',
              caption: 'Lân bắt đầu xoay.',
            },
            {
              event_id: 'E2',
              frame_id: 'L24_V018_f0012',
              caption: 'Lân tiếp đất.',
            },
          ],
        }]}
      />,
    );

    expect(screen.getByAltText('E1 candidate for L24_V018')).toBeInTheDocument();
    expect(screen.getByText('E1 · 1/2')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Next event evidence for L24_V018' }));

    expect(screen.getByAltText('E2 candidate for L24_V018')).toBeInTheDocument();
    expect(screen.getByText('E2 · 2/2')).toBeInTheDocument();
  });

  it('paginates a large candidate list while preserving its overall rank', async () => {
    const user = userEvent.setup();
    const candidates = Array.from({ length: 16 }, (_, index) => ({
      video_id: `L26_V${String(index + 1).padStart(3, '0')}`,
      score: 1 - index / 100,
      summary_score: 0,
      kis_score: 0,
    }));
    render(<TemporalVideoCandidates candidates={candidates} />);

    expect(screen.getByText('L26_V001')).toBeInTheDocument();
    expect(screen.queryByText('L26_V016')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Next' }));

    expect(screen.getByText('L26_V016')).toBeInTheDocument();
    expect(screen.getByText('#16')).toBeInTheDocument();
  });
});
