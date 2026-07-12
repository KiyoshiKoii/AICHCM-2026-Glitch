import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SearchBar from './SearchBar.jsx';

const getTextInput = () =>
  screen.getByPlaceholderText(/mô tả cảnh cần tìm/i);
const getFileInput = (container) => container.querySelector('input[type="file"]');
const getSubmitButton = () => screen.getByRole('button', { name: /tìm kiếm/i });

describe('SearchBar', () => {
  it('renders a text input, a file input, and a submit button', () => {
    const { container } = render(<SearchBar />);
    expect(getTextInput()).toBeInTheDocument();
    expect(getFileInput(container)).toBeInTheDocument();
    expect(getSubmitButton()).toBeInTheDocument();
  });

  it('calls onSearch with the trimmed query on submit', async () => {
    const user = userEvent.setup();
    const onSearch = vi.fn();
    const onImageSearch = vi.fn();
    render(<SearchBar onSearch={onSearch} onImageSearch={onImageSearch} />);

    await user.type(getTextInput(), '  người đàn ông làm rơi ví  ');
    await user.click(getSubmitButton());

    expect(onSearch).toHaveBeenCalledWith('người đàn ông làm rơi ví');
    expect(onImageSearch).not.toHaveBeenCalled();
  });

  it('does not call onSearch when the query is empty or whitespace-only', async () => {
    const user = userEvent.setup();
    const onSearch = vi.fn();
    render(<SearchBar onSearch={onSearch} />);

    await user.type(getTextInput(), '   ');
    await user.click(getSubmitButton());

    expect(onSearch).not.toHaveBeenCalled();
  });

  it('calls onImageSearch instead of onSearch when an image is selected, even with text present', async () => {
    const user = userEvent.setup();
    const onSearch = vi.fn();
    const onImageSearch = vi.fn();
    const { container } = render(
      <SearchBar onSearch={onSearch} onImageSearch={onImageSearch} />
    );
    const file = new File(['fake-image-bytes'], 'query.jpg', { type: 'image/jpeg' });

    await user.type(getTextInput(), 'người đàn ông làm rơi ví');
    await user.upload(getFileInput(container), file);
    await user.click(getSubmitButton());

    expect(onImageSearch).toHaveBeenCalledWith(file);
    expect(onSearch).not.toHaveBeenCalled();
  });
});
