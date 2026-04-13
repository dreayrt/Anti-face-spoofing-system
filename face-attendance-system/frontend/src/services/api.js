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
  
  // Submit face image, bounding box, and descriptor for verification
  verifyFace: async (imageBase64, faceBox, descriptor) => {
    const response = await apiClient.post('/face/recognize', { 
      image: imageBase64,
      box: faceBox,
      descriptor: descriptor
    });
    return response.data;
  },

  // Register a new employee with face image
  registerEmployee: async (employeeData) => {
    const response = await apiClient.post('/face/register', employeeData);
    return response.data;
  },

  // Record manual attendance or query stats
  getStats: async () => {
    const response = await apiClient.get('/attendance/stats');
    return response.data;
  }
};

