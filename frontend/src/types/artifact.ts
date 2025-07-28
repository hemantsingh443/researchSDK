export interface Artifact {
  name: string;
  size: number;
  created: string; // ISO date string
  type?: string;   // MIME type or file extension
  path?: string;   // Full path to the artifact
  lastModified?: string; // ISO date string
  url?: string;    // Download URL if available
  metadata?: Record<string, unknown>; // Additional metadata
}

export interface ArtifactListResponse {
  artifacts: Artifact[];
}
