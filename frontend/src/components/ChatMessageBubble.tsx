import React, { useState } from 'react';
import { Bot, User, Copy, Check, Clock, Gauge, Volume2 } from 'lucide-react';
import type { ChatMessage } from '../types';
import { AudioMessagePlayer } from './AudioMessagePlayer';

interface ChatMessageBubbleProps {
  message: ChatMessage;
  onPlayStateChange?: (isPlaying: boolean) => void;
  autoPlay?: boolean;
}

export const ChatMessageBubble: React.FC<ChatMessageBubbleProps> = ({
  message,
  onPlayStateChange,
  autoPlay = false,
}) => {
  const [copied, setCopied] = useState(false);
  const isAssistant = message.role === 'assistant';

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`message-row ${isAssistant ? 'assistant' : 'user'} animate-fade-in`}>
      <div className={`message-avatar ${isAssistant ? 'bot-avatar' : 'user-avatar'}`}>
        {isAssistant ? <Bot size={18} /> : <User size={18} />}
      </div>

      <div className="message-content-container">
        <div className="message-header-meta">
          <span className="sender-name">{isAssistant ? 'Voice AI (SpeechT5)' : 'You'}</span>
          <span className="message-time">{message.timestamp}</span>
        </div>

        <div className="message-bubble">
          <p className="message-text">{message.content}</p>

          {/* Assistant Audio Player */}
          {isAssistant && message.audioUrl && (
            <div className="message-audio-section">
              <AudioMessagePlayer
                audioUrl={message.audioUrl}
                onPlayStateChange={onPlayStateChange}
                autoPlay={autoPlay}
              />
            </div>
          )}

          {/* Synthesizing state loader */}
          {message.isSynthesizing && (
            <div className="synthesizing-loader">
              <div className="typing-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
              <span className="synthesizing-text">Synthesizing neural voice...</span>
            </div>
          )}
        </div>

        {/* Footer Metrics & Actions */}
        <div className="message-footer">
          {isAssistant && message.metrics && (
            <div className="metrics-badge-group">
              {message.metrics.llm_time_seconds !== undefined && (
                <span className="metric-tag" title="LLM Response Latency">
                  <Clock size={11} />
                  LLM: {message.metrics.llm_time_seconds}s
                </span>
              )}
              {message.metrics.tts_time_seconds !== undefined && (
                <span className="metric-tag" title="TTS Generation Latency">
                  <Volume2 size={11} />
                  TTS: {message.metrics.tts_time_seconds}s
                </span>
              )}
              {message.metrics.real_time_factor !== undefined && (
                <span
                  className={`metric-tag ${
                    message.metrics.real_time_factor < 1 ? 'rtf-fast' : 'rtf-slow'
                  }`}
                  title="Real Time Factor (< 1.0 is faster than realtime)"
                >
                  <Gauge size={11} />
                  RTF: {message.metrics.real_time_factor}
                </span>
              )}
            </div>
          )}

          <button
            onClick={handleCopy}
            className="copy-btn"
            title="Copy to clipboard"
          >
            {copied ? <Check size={13} color="#10b981" /> : <Copy size={13} />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
