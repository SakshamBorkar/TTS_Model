import type { ChatMessage, LLMConfig, VoiceResponse, SystemMetrics } from './types';

// Default API URL (proxyable or fallback to localhost:8000)
export const getApiBaseUrl = (): string => {
  return localStorage.getItem('voiceai_api_url') || 'http://localhost:8000';
};

export const setApiBaseUrl = (url: string): void => {
  localStorage.setItem('voiceai_api_url', url);
};

export interface SendChatPayload {
  message: string;
  history: { role: string; content: string }[];
  provider: string;
  apiKey?: string;
  modelName?: string;
  baseUrl?: string;
  systemPrompt?: string;
  synthesizeVoice: boolean;
}

export interface ChatApiResponse {
  reply: string;
  audio_url?: string;
  metrics: {
    llm_time_seconds: number;
    tts_time_seconds: number;
    audio_duration_seconds: number;
    real_time_factor: number;
    device: string;
  };
}

export const sendChatMessage = async (
  message: string,
  history: ChatMessage[],
  config: LLMConfig,
  synthesizeVoice: boolean = true
): Promise<ChatApiResponse> => {
  const baseUrl = getApiBaseUrl();
  const historyPayload = history.map((m) => ({
    role: m.role,
    content: m.content,
  }));

  const response = await fetch(`${baseUrl}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message,
      history: historyPayload,
      provider: config.provider,
      api_key: config.apiKey || undefined,
      model_name: config.modelName || undefined,
      base_url: config.baseUrl || undefined,
      system_prompt: config.systemPrompt || undefined,
      synthesize_voice: synthesizeVoice,
    }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || 'Failed to send chat message.');
  }

  return response.json();
};

export const synthesizeDirectTTS = async (text: string): Promise<{ audio_url: string; metrics: any }> => {
  const baseUrl = getApiBaseUrl();
  const response = await fetch(`${baseUrl}/api/tts`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || 'Failed to synthesize speech.');
  }

  return response.json();
};

export const fetchVoices = async (): Promise<VoiceResponse> => {
  const baseUrl = getApiBaseUrl();
  const response = await fetch(`${baseUrl}/api/voices`);
  if (!response.ok) {
    throw new Error('Failed to fetch voice information.');
  }
  return response.json();
};

export const fetchMetrics = async (): Promise<SystemMetrics> => {
  const baseUrl = getApiBaseUrl();
  const response = await fetch(`${baseUrl}/metrics`);
  if (!response.ok) {
    throw new Error('Failed to fetch system metrics.');
  }
  return response.json();
};

export const checkHealth = async (): Promise<{ status: string; model_loaded: boolean; device: string }> => {
  const baseUrl = getApiBaseUrl();
  const response = await fetch(`${baseUrl}/health`);
  if (!response.ok) {
    throw new Error('Backend is unreachable.');
  }
  return response.json();
};
