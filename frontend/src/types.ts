export interface ChatMetrics {
  llm_time_seconds?: number;
  tts_time_seconds?: number;
  audio_duration_seconds?: number;
  real_time_factor?: number;
  device?: string;
  total_inference_time_seconds?: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  audioUrl?: string;
  metrics?: ChatMetrics;
  isSynthesizing?: boolean;
}

export type LLMProvider = 'offline' | 'openai' | 'groq' | 'ollama' | 'custom';

export interface LLMConfig {
  provider: LLMProvider;
  apiKey: string;
  modelName: string;
  baseUrl: string;
  systemPrompt: string;
}

export interface VoiceOption {
  id: string;
  name: string;
  index: number;
  gender: string;
}

export interface VoiceResponse {
  current_voice: {
    name: string;
    dataset: string;
    index: number;
    speaker_id: string;
  };
  voices: VoiceOption[];
  model: string;
  vocoder: string;
  sample_rate: number;
  device: string;
  status: string;
}

export interface SystemMetrics {
  request_count: number;
  chat_count: number;
  error_count: number;
  total_audio_seconds: number;
  total_inference_seconds: number;
  latency?: {
    mean_s: number;
    p50_s: number;
    p95_s: number;
    min_s: number;
    max_s: number;
  };
  device: string;
}
