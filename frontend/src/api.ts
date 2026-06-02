// api.ts

// Define response type (VERY IMPORTANT for TS)
export interface AnalyzeResponse {
  submission_id: string;
  timestamp_utc: string;
  file: {
    name: string;
    sha256: string;
    media_type: string;
    size_human: string;
    size_bytes: number;
  };
  deepfake_detection: {
    label: string;
    confidence: number;
    risk_level: string;
    is_deepfake: boolean;
    fake_probability: number;
    streams: Record<string, number>;
    metadata_flags: string[];
    gradcam_url: string | null;
    processing_ms: number;
  };
  fact_check: {
    available: boolean;
    note: string;
    label?: string;
    harmful_score?: number;
    flags?: string[];
  };
  integrity: {
    sha256: string;
    audit_entry: string;
  };
  legal_report_url: string | null;
}

// Function to call backend API with timeout
export async function analyzeFile(file: File, timeoutMs: number = 300000): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append("file", file);

  // Create abort controller for timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch("http://127.0.0.1:5001/analyze", {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: response.statusText }));
      throw new Error(errorData.error || `API error: ${response.status}`);
    }

    return response.json();
  } finally {
    clearTimeout(timeoutId);
  }
}