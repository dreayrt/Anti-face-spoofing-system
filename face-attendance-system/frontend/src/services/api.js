import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

// Create configured axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  // Check health status
  healthCheck: async () => {
    const response = await apiClient.get('/health');
    return response.data;
  },
  
  // Submit face image and bounding box for verification
  verifyFace: async (imageBase64, faceBox) => {
    // In a real implementation you might send a multipart/form-data with a blob
    // For simplicity, we'll send base64 encoded JSON here.
    const response = await apiClient.post('/face/recognize', { 
      image: imageBase64,
      box: faceBox
    });
    return response.data;
  },

  // Record manual attendance or query stats
  getStats: async () => {
    const response = await apiClient.get('/attendance/stats');
    return response.data;
  }
};
