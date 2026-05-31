import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

export async function recognizeFace(payload) {
  const { data } = await api.post('/face/recognize', payload);
  return data;
}

export async function registerEmployee(payload) {
  const { data } = await api.post('/face/register', payload);
  return data;
}

export async function checkLiveness(payload) {
  const { data } = await api.post('/face/liveness', payload);
  return data;
}

export default api;
