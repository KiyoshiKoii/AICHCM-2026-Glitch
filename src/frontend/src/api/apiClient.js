/**
 * apiClient.js
 * Giao tiếp trực tiếp với Backend (Dev 3) thông qua các endpoints của API Contract.
 */

export const searchByText = async (query, topK = 50) => {
  const response = await fetch('/api/v1/search/text', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query, top_k: topK }),
  });
  
  if (!response.ok) {
    throw new Error(`Text search failed with status ${response.status}`);
  }
  return response.json();
};

export const searchByImage = async (imageFile, topK = 50) => {
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
) => {
  const response = await fetch('/api/v1/vqa', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      question,
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
