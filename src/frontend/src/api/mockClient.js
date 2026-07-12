import searchMockData from '../../mock/search_response_v1.json';

export const fetchSearchResults = async (query) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(searchMockData);
    }, 1000);
  });
};
