import React from 'react';
import { Bot, Activity, Volume2, VolumeX, CheckCircle2, AlertCircle } from 'lucide-react';
import { AudioVisualizer } from './AudioVisualizer';
import type { VoiceResponse } from '../types';

interface HeaderProps {
  isPlayingVoice: boolean;
  isSynthesizing: boolean;
  autoPlayVoice: boolean;
  onToggleAutoPlay: (enabled: boolean) => void;
  onOpenMetrics: () => void;
  voiceInfo: VoiceResponse | null;
  isConnected: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  isPlayingVoice,
  isSynthesizing,
  autoPlayVoice,
  onToggleAutoPlay,
  onOpenMetrics,
  voiceInfo,
  isConnected,
}) => {
  return (
    <header className="app-header glass-panel">
      <div className="header-left">
        <div className="app-logo">
          <div className="logo-icon-wrapper">
            <Bot size={22} className="logo-icon" />
            <span className="logo-pulse"></span>
          </div>
          <div className="logo-text-group">
            <div className="logo-title-row">
              <h1 className="logo-title">Chloe</h1>
              <span className="badge-pill">Voice AI</span>
            </div>
            <p className="logo-subtitle">Neural SpeechT5 & Conversational LLM</p>
          </div>
        </div>
      </div>

      {/* Center Audio Visualizer */}
      <div className="header-center">
        <div className="visualizer-container">
          <AudioVisualizer isPlaying={isPlayingVoice} isSynthesizing={isSynthesizing} />
          <div className="visualizer-status-text">
            {isPlayingVoice ? (
              <span className="status-speaking">Chloe is speaking...</span>
            ) : isSynthesizing ? (
              <span className="status-synthesizing">Chloe is thinking & voicing...</span>
            ) : (
              <span className="status-ready">Chloe is listening</span>
            )}
          </div>
        </div>
      </div>

      {/* Header Right Controls */}
      <div className="header-right">
        {/* Device & Status Badge */}
        <div className="model-status-pill" title="Hardware backend & neural engine">
          {isConnected ? (
            <CheckCircle2 size={13} color="#10b981" />
          ) : (
            <AlertCircle size={13} color="#f59e0b" />
          )}
          <span className="status-label">
            {voiceInfo?.device ? voiceInfo.device.toUpperCase() : 'CONNECTING...'}
          </span>
        </div>

        {/* Auto-Play Toggle */}
        <button
          onClick={() => onToggleAutoPlay(!autoPlayVoice)}
          className={`header-icon-btn ${autoPlayVoice ? 'auto-play-active' : ''}`}
          title={autoPlayVoice ? 'Voice Auto-play: ON' : 'Voice Auto-play: OFF'}
        >
          {autoPlayVoice ? <Volume2 size={18} /> : <VolumeX size={18} />}
        </button>

        {/* Metrics Button */}
        <button
          onClick={onOpenMetrics}
          className="header-icon-btn"
          title="System Statistics & Latency"
        >
          <Activity size={18} />
        </button>
      </div>
    </header>
  );
};
