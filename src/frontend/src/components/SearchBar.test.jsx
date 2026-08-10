import { describe, it, expect, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SearchBar from './SearchBar.jsx';

const getTextInput = () =>
  screen.getByPlaceholderText(/mô tả cảnh cần tìm/i);
const getFileInput = (container) => container.querySelector('input[type="file"]');
const getSubmitButton = () => screen.getByRole('button', { name: /tìm kiếm/i });

describe('SearchBar', () => {
  it('renders a text input and a submit button', () => {
    const { container } = render(<SearchBar />);
    expect(getTextInput()).toBeInTheDocument();
    expect(getFileInput(container)).not.toBeInTheDocument();
    expect(getSubmitButton()).toBeInTheDocument();
  });

  it('calls onSearch with the trimmed query on submit', async () => {
    const user = userEvent.setup();
    const onSearch = vi.fn();
    const onImageSearch = vi.fn();
    render(<SearchBar onSearch={onSearch} onImageSearch={onImageSearch} />);

    await user.type(getTextInput(), '  người đàn ông làm rơi ví  ');
    await user.click(getSubmitButton());

    expect(onSearch).toHaveBeenCalledWith(
      'người đàn ông làm rơi ví',
      false,
      { textWeight: 0.5, visualWeight: 0.5 },
      { batchIds: [], videoIds: [] },
    );
    expect(onImageSearch).not.toHaveBeenCalled();
  });

  it('passes useRerank when Gemini re-rank is enabled', async () => {
    const user = userEvent.setup();
    const onSearch = vi.fn();
    render(<SearchBar onSearch={onSearch} />);

    await user.type(getTextInput(), 'người đàn ông làm rơi ví');
    await user.click(screen.getByRole('checkbox', { name: /gemini re-rank/i }));
    await user.click(getSubmitButton());

    expect(onSearch).toHaveBeenCalledWith(
      'người đàn ông làm rơi ví',
      true,
      { textWeight: 0.5, visualWeight: 0.5 },
      { batchIds: [], videoIds: [] },
    );
  });

  it('passes the selected visual/text fusion weights', async () => {
    const user = userEvent.setup();
    const onSearch = vi.fn();
    render(<SearchBar onSearch={onSearch} />);

    await user.type(getTextInput(), 'người đàn ông làm rơi ví');
    fireEvent.change(screen.getByRole('slider', { name: /text search weight/i }), {
      target: { value: '70' },
    });
    await user.click(getSubmitButton());

    expect(onSearch).toHaveBeenCalledWith(
      'người đàn ông làm rơi ví',
      false,
      { textWeight: 0.7, visualWeight: 0.3 },
      { batchIds: [], videoIds: [] },
    );
  });

  it('does not call onSearch when the query is empty or whitespace-only', async () => {
    const user = userEvent.setup();
    const onSearch = vi.fn();
    render(<SearchBar onSearch={onSearch} />);

    await user.type(getTextInput(), '   ');
    await user.click(getSubmitButton());

    expect(onSearch).not.toHaveBeenCalled();
  });

  it('calls onImageSearch from Image Search mode', async () => {
    const user = userEvent.setup();
    const onSearch = vi.fn();
    const onImageSearch = vi.fn();
    const { container } = render(
      <SearchBar onSearch={onSearch} onImageSearch={onImageSearch} />
    );
    const file = new File(['fake-image-bytes'], 'query.jpg', { type: 'image/jpeg' });

    await user.click(screen.getByRole('tab', { name: /image search/i }));
    await user.upload(getFileInput(container), file);
    await user.click(getSubmitButton());

    expect(onImageSearch).toHaveBeenCalledWith(file);
    expect(onSearch).not.toHaveBeenCalled();
  });

  it('calls onVqaSearch with description and question in VQA mode', async () => {
    const user = userEvent.setup();
    const onVqaSearch = vi.fn();
    render(<SearchBar onVqaSearch={onVqaSearch} />);

    await user.click(screen.getByRole('tab', { name: 'VQA' }));
    await user.type(getTextInput(), 'lễ trao giải có nhiều người trên sân khấu');
    await user.type(screen.getByPlaceholderText(/câu hỏi cần trả lời/i), 'Có bao nhiêu người?');
    await user.click(screen.getByRole('button', { name: /trả lời/i }));

    expect(onVqaSearch).toHaveBeenCalledWith({
      query: 'lễ trao giải có nhiều người trên sân khấu',
      question: 'Có bao nhiêu người?',
    });
  });
});
