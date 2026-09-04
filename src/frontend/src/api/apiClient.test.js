import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  getVideoSummaries,
  searchByText,
  searchTemporalEvents,
  searchTemporalVideos,
} from './apiClient.js';


const eventOptions = [{
  eventId: 'E2',
  textWeight: 0.3,
  visualWeight: 0.7,
  useRerank: true,
  requiresAfterPrevious: true,
  verifyCameraMotion: true,
  motionWeight: 0.65,
}];

const mockResponse = () => ({
  ok: true,
  json: vi.fn().mockResolvedValue({ status: 'success', data: {} }),
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('temporal API client', () => {
  it('serializes the optional ASR source for KIS + ASR frame search', async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockResponse());
    vi.stubGlobal('fetch', fetchMock);

    await searchByText(
      'squid with white wine',
      100,
      false,
      { textWeight: 0.3, visualWeight: 0.3, asrWeight: 0.4 },
      { batchIds: ['L26'] },
    );

    const payload = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(payload).toMatchObject({
      text_weight: 0.3,
      visual_weight: 0.3,
      asr_weight: 0.4,
      batch_ids: ['L26'],
    });
  });

  it('serializes per-event options for candidate-video ranking', async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockResponse());
    vi.stubGlobal('fetch', fetchMock);

    await searchTemporalVideos(
      'E1: riders enter the bridge\nE2: riders leave the bridge',
      { summaryWeight: 0.6, kisWeight: 0.4 },
      { batchIds: ['L23'] },
      eventOptions,
    );

    const payload = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(payload).toMatchObject({
      summary_weight: 0.6,
      kis_weight: 0.4,
      event_options: [{
        event_id: 'E2',
        text_weight: 0.3,
        visual_weight: 0.7,
        use_rerank: true,
        requires_after_previous: true,
        min_frame_gap: 30,
        verify_camera_motion: true,
        motion_weight: 0.65,
      }],
    });
  });

  it('serializes the same controls for selected-video event search', async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockResponse());
    vi.stubGlobal('fetch', fetchMock);

    await searchTemporalEvents(
      'E1: camera tilts up',
      { videoIds: ['L26_V074'] },
      { textWeight: 0.5, visualWeight: 0.5 },
      eventOptions,
    );

    const payload = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(payload.video_ids).toEqual(['L26_V074']);
    expect(payload.event_options[0]).toMatchObject({
      event_id: 'E2',
      requires_after_previous: true,
      verify_camera_motion: true,
    });
  });

  it('requests exact summaries for temporal result cards', async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockResponse());
    vi.stubGlobal('fetch', fetchMock);

    await getVideoSummaries(['L26_V311']);

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/search/video-summaries',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ video_ids: ['L26_V311'] }),
      }),
    );
  });
});
