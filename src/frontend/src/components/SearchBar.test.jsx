import { describe, it, expect, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SearchBar from './SearchBar.jsx';

const getTextInput = () =>
  screen.getByPlaceholderText(/mô tả cảnh cần tìm/i);
const getSubmitButton = () => screen.getByRole('button', { name: /tìm kiếm/i });

describe('SearchBar', () => {
  it('renders a text input and a submit button', () => {
    render(<SearchBar />);
    expect(getTextInput()).toBeInTheDocument();
    expect(getSubmitButton()).toBeInTheDocument();
  });

  it('calls onSearch with the trimmed query on submit', async () => {
    const user = userEvent.setup();
    const onSearch = vi.fn();
    render(<SearchBar onSearch={onSearch} />);

    await user.type(getTextInput(), '  người đàn ông làm rơi ví  ');
    await user.click(getSubmitButton());

    expect(onSearch).toHaveBeenCalledWith(
      'người đàn ông làm rơi ví',
      false,
      { textWeight: 0.5, visualWeight: 0.5 },
      { batchIds: [], videoIds: [] },
    );
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

  it('passes the optional camera-motion verifier for a regular KIS search', async () => {
    const user = userEvent.setup();
    const onSearch = vi.fn();
    render(<SearchBar onSearch={onSearch} />);

    await user.type(getTextInput(), 'nguyên liệu hải sản');
    await user.click(screen.getByRole('checkbox', { name: /verify shot transition/i }));
    await user.type(
      screen.getByRole('textbox', { name: /camera motion query/i }),
      'máy quay chéo lên và kết thúc tại cảnh đích',
    );
    fireEvent.change(screen.getByRole('slider', { name: /kis camera motion weight/i }), {
      target: { value: '60' },
    });
    await user.click(getSubmitButton());

    expect(onSearch).toHaveBeenCalledWith(
      'nguyên liệu hải sản',
      false,
      {
        textWeight: 0.5,
        visualWeight: 0.5,
        verifyCameraMotion: true,
        cameraMotionQuery: 'máy quay chéo lên và kết thúc tại cảnh đích',
        motionWeight: 0.6,
      },
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

  it('searches one frame with visual, caption and ASR evidence in KIS + ASR mode', async () => {
    const user = userEvent.setup();
    const onSearch = vi.fn();
    render(<SearchBar onSearch={onSearch} />);

    await user.click(screen.getByRole('tab', { name: 'KIS + ASR' }));
    await user.type(getTextInput(), 'squid with white wine');
    fireEvent.change(screen.getByRole('slider', { name: /kis and asr fusion weight/i }), {
      target: { value: '60' },
    });
    await user.click(getSubmitButton());

    expect(onSearch).toHaveBeenCalledWith(
      'squid with white wine',
      false,
      { textWeight: 0.3, visualWeight: 0.3, asrWeight: 0.4 },
      { batchIds: [], videoIds: [] },
    );
  });

  it('searches timestamped transcripts in ASR mode', async () => {
    const user = userEvent.setup();
    const onAsrSearch = vi.fn();
    render(<SearchBar onAsrSearch={onAsrSearch} filters={{ batchIds: ['L26'], videoIds: [] }} />);

    await user.click(screen.getByRole('tab', { name: /asr search/i }));
    await user.type(screen.getByRole('textbox'), '  cho dầu vào chảo  ');
    await user.click(screen.getByRole('button', { name: /tìm lời thoại/i }));

    expect(onAsrSearch).toHaveBeenCalledWith(
      'cho dầu vào chảo',
      { batchIds: ['L26'], videoIds: [] },
    );
  });

  it('opens a requested video at the specified frame', async () => {
    const user = userEvent.setup();
    const onVideoSearch = vi.fn();
    render(<SearchBar onVideoSearch={onVideoSearch} />);

    await user.click(screen.getByRole('tab', { name: 'Video Search' }));
    await user.clear(screen.getByRole('textbox', { name: 'Video batch' }));
    await user.type(screen.getByRole('textbox', { name: 'Video batch' }), '26');
    await user.clear(screen.getByRole('textbox', { name: 'Video id' }));
    await user.type(screen.getByRole('textbox', { name: 'Video id' }), '74');
    await user.type(screen.getByRole('textbox', { name: 'Video frame' }), '103');
    await user.click(screen.getByRole('button', { name: /video/i }));

    expect(onVideoSearch).toHaveBeenCalledWith({
      batchId: 'L26',
      videoId: 'V074',
      frameIndex: 103,
    });
  });

  it('defaults direct video lookup to the first frame', async () => {
    const user = userEvent.setup();
    const onVideoSearch = vi.fn();
    render(<SearchBar onVideoSearch={onVideoSearch} />);

    await user.click(screen.getByRole('tab', { name: 'Video Search' }));
    await user.click(screen.getByRole('button', { name: /video/i }));

    expect(onVideoSearch).toHaveBeenCalledWith({
      batchId: 'L21',
      videoId: 'V001',
      frameIndex: null,
    });
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
      useRerank: false,
    });
  });

  it('passes the VQA re-rank toggle to the submit handler', async () => {
    const user = userEvent.setup();
    const onVqaSearch = vi.fn();
    render(<SearchBar onVqaSearch={onVqaSearch} />);

    await user.click(screen.getByRole('tab', { name: 'VQA' }));
    await user.type(getTextInput(), 'người đứng cạnh xe máy');
    await user.type(screen.getByPlaceholderText(/câu hỏi cần trả lời/i), 'Màu gì?');
    await user.click(screen.getByRole('checkbox', { name: /gemini re-rank/i }));
    await user.click(screen.getByRole('button', { name: /trả lời/i }));

    expect(onVqaSearch).toHaveBeenCalledWith({
      query: 'người đứng cạnh xe máy',
      question: 'Màu gì?',
      useRerank: true,
    });
  });
});
