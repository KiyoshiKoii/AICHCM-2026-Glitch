/**
 * apiClient.js
 * Giao tiếp trực tiếp với Backend (Dev 3) thông qua các endpoints của API Contract.
 */

export const searchByText = async (
  query,
  topK = 100,
  useRerank = false,
  { textWeight = 0.5, visualWeight = 0.5 } = {},
  { batchIds = [], videoIds = [] } = {},
) => {
  const response = await fetch('/api/v1/search/text', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      top_k: topK,
      use_rerank: useRerank,
      text_weight: textWeight,
      visual_weight: visualWeight,
      batch_ids: batchIds,
      video_ids: videoIds,
    }),
  });
  
  if (!response.ok) {
    throw new Error(`Text search failed with status ${response.status}`);
  }
  return response.json();
};

export const searchByImage = async (imageFile, topK = 100) => {
  const formData = new FormData();
  formData.append('image_file', imageFile);
  formData.append('top_k', topK.toString());

  const response = await fetch('/api/v1/search/image', {
    method: 'POST',
    body: formData, // FormData tự động set Content-Type multipart/form-data
  });

  if (!response.ok) {
    throw new Error(`Image search failed with status ${response.status}`);
  }
  return response.json();
};

export const answerVqa = async (
  query,
  question,
  retrievalTopK = 50,
  answerTopK = 10,
  useRerank = false,
) => {
  const response = await fetch('/api/v1/vqa', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      question,
      use_rerank: useRerank,
      retrieval_top_k: retrievalTopK,
      answer_top_k: answerTopK,
    }),
  });

  if (!response.ok) {
    throw new Error(`VQA failed with status ${response.status}`);
  }
  return response.json();
};

export const getFrameContext = async (frameId, window = 5) => {
  const response = await fetch(`/api/v1/frames/context/${frameId}?window=${window}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Frame context failed with status ${response.status}`);
  }
  return response.json();
};

export const searchTemporalEvents = async (
  query,
  { batchIds = [], videoIds = [], topKVideos = 20 } = {},
) => {
  const response = await fetch('/api/v1/search/temporal-events', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      batch_ids: batchIds,
      video_ids: videoIds,
      top_k_videos: topKVideos,
    }),
  });
  if (!response.ok) {
    throw new Error(`Temporal event search failed with status ${response.status}`);
  }
  return response.json();
};

export const getFrameTimeline = async (frameId) => {
  const response = await fetch(`/api/v1/frames/timeline/${frameId}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Frame timeline failed with status ${response.status}`);
  }
  return response.json();
};

export const searchTemporalVideos = async (
  query,
  { summaryWeight = 0.75, kisWeight = 0.25 } = {},
  { batchIds = [], videoIds = [], topKVideos = 20 } = {},
) => {
  const response = await fetch('/api/v1/search/temporal-videos', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      batch_ids: batchIds,
      video_ids: videoIds,
      top_k_videos: topKVideos,
      summary_weight: summaryWeight,
      kis_weight: kisWeight,
    }),
  });
  if (!response.ok) {
    throw new Error(`Temporal video search failed with status ${response.status}`);
  }
  return response.json();
};
