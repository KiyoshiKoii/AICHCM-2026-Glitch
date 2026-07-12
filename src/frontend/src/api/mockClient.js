import searchMockData from '../../mock/search_response_v1.json';
import frameContextMockData from '../../mock/frame_context_response_v1.json';

export const fetchSearchResults = async (query) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(searchMockData);
    }, 1000);
  });
};

export const getFrameContext = async (frameId, window = 5) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(frameContextMockData);
    }, 1000);
  });
};
