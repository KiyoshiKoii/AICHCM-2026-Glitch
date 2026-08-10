import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import FilterPanel from './FilterPanel.jsx';

describe('FilterPanel', () => {
  it('renders all L21-L30 batch filters with compact content labels', () => {
    render(<FilterPanel />);

    expect(screen.getAllByRole('checkbox')).toHaveLength(10);
    expect(screen.getByText('Tin sáng')).toBeInTheDocument();
    expect(screen.getByText('Phóng sự cộng đồng')).toBeInTheDocument();
  });

  it('submits selected batches and zero-padded numeric video IDs', async () => {
    const user = userEvent.setup();
    const onFiltersChange = vi.fn();
    render(<FilterPanel onFiltersChange={onFiltersChange} />);

    await user.click(screen.getByLabelText(/L21/));
    await user.type(screen.getByRole('textbox', { name: /video id filter/i }), '6, 30');

    expect(onFiltersChange).toHaveBeenLastCalledWith({
      batchIds: ['L21'],
      videoIds: ['V006', 'V030'],
    });
  });

  it('does not render a second search/apply button', () => {
    render(<FilterPanel />);

    expect(screen.queryByRole('button', { name: /apply filter/i })).not.toBeInTheDocument();
  });
});
