import type { Artifact } from '../types/artifact';

// Base URL for the API - adjust if your API is hosted elsewhere
const API_BASE_URL = 'http://localhost:8000'; // Default FastAPI port

interface ApiError extends Error {
  status?: number;
  details?: unknown;
}

const handleApiError = async (response: Response): Promise<never> => {
  let errorMessage = `Request failed with status ${response.status}`;
  let errorDetails: unknown = null;

  try {
    const errorData = await response.json().catch(() => ({}));
    if (errorData?.message) {
      errorMessage = errorData.message;
      errorDetails = errorData.details;
    }
  } catch (e) {
    // Ignore JSON parse errors, use default error message
  }

  const error: ApiError = new Error(errorMessage);
  error.status = response.status;
  if (errorDetails) error.details = errorDetails;
  throw error;
};

export const getArtifacts = async (): Promise<Artifact[]> => {
  try {
    const response = await fetch(`${API_BASE_URL}/artifacts`);
    if (!response.ok) {
      return await handleApiError(response);
    }
    const data = await response.json();
    // The backend returns { artifacts: Artifact[] }
    return Array.isArray(data.artifacts) ? data.artifacts : [];
  } catch (error) {
    console.error('Error fetching artifacts:', error);
    throw new Error('Failed to fetch artifacts. Please check your connection and try again.');
  }
};

export const deleteArtifact = async (name: string): Promise<void> => {
  try {
    const response = await fetch(`${API_BASE_URL}/artifacts/delete/${encodeURIComponent(name)}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    if (!response.ok) {
      return await handleApiError(response);
    }
  } catch (error) {
    console.error(`Error deleting artifact ${name}:`, error);
    throw new Error(`Failed to delete artifact: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
};

export const getArtifactUrl = (filename: string): string => {
  return `${API_BASE_URL}/${filename}`;
};

export const downloadArtifact = async (name: string): Promise<void> => {
  try {
    // The backend serves files directly at /artifacts/{filename}
    const url = `${API_BASE_URL}/artifacts/${encodeURIComponent(name)}`;
    
    // Open in a new tab to trigger download
    window.open(url, '_blank');
  } catch (error) {
    console.error(`Error downloading artifact ${name}:`, error);
    throw new Error(`Failed to download artifact: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
};
