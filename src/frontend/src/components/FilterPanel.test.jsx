import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import FilterPanel from './FilterPanel.jsx';

describe('FilterPanel', () => {
  it('starts with a single filter row', () => {
    render(<FilterPanel canSearch={false} />);
    expect(screen.getAllByPlaceholderText('Enter value')).toHaveLength(1);
  });

  it('adds a new filter row when Add is clicked', async () => {
    const user = userEvent.setup();
    render(<FilterPanel canSearch={false} />);

    await user.click(screen.getByRole('button', { name: /add/i }));

    expect(screen.getAllByPlaceholderText('Enter value')).toHaveLength(2);
  });

  it('removes a filter row when its remove button is clicked', async () => {
    const user = userEvent.setup();
    render(<FilterPanel canSearch={false} />);

    await user.click(screen.getByRole('button', { name: /add/i }));
    await user.click(screen.getAllByRole('button', { name: /remove filter/i })[0]);

    expect(screen.getAllByPlaceholderText('Enter value')).toHaveLength(1);
  });

  it('disables Search when canSearch is false and enables/calls onSearch otherwise', async () => {
    const user = userEvent.setup();
    const onSearch = vi.fn();
    const { rerender } = render(<FilterPanel canSearch={false} onSearch={onSearch} />);

    expect(screen.getByRole('button', { name: /^search$/i })).toBeDisabled();

    rerender(<FilterPanel canSearch onSearch={onSearch} />);
    await user.click(screen.getByRole('button', { name: /^search$/i }));

    expect(onSearch).toHaveBeenCalled();
  });
});
