// src/services/api.js

const API_BASE_URL = 'http://localhost:8000';

/**
 * Service for interacting with the RAG backend API
 */
const apiService = {
  /**
   * Send a query to the RAG backend
   * @param {string} query - The user's text query
   * @param {number} [topK=5] - Number of results to retrieve from the vector DB
   * @returns {Promise<Object>} - The response containing answer and sources
   */
  async sendQuery(query, type, topK = 5) {
    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          collection_name: type,
          top_k: topK
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.detail || `Server error: ${response.status}`
        );
      }

      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  },

  /**
   * Check the health of the API
   * @returns {Promise<Object>} - The health status
   */
  async checkHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }
};

export default apiService;