import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import { Header } from './components/Header';
import { ChatMessageBubble } from './components/ChatMessageBubble';
import { ChatInput } from './components/ChatInput';
import { PromptSuggestions } from './components/PromptSuggestions';
import { SettingsModal } from './components/SettingsModal';
import { MetricsDrawer } from './components/MetricsDrawer';
import type { ChatMessage, LLMConfig, VoiceResponse } from './types';
import { sendChatMessage, fetchVoices, checkHealth } from './api';
import { Bot } from 'lucide-react';

const DEFAULT_CONFIG: LLMConfig = {
  provider: 'offline',
  apiKey: '',
  modelName: '',
  baseUrl: '',
  systemPrompt:
    'You are a courteous, helpful voice customer support AI assistant. Keep responses clear, concise, and easy to understand when spoken.',
};

export const App: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    const saved = localStorage.getItem('voiceai_chat_history');
    return saved ? JSON.parse(saved) : [];
  });

  const [config, setConfig] = useState<LLMConfig>(() => {
    const saved = localStorage.getItem('voiceai_llm_config');
    return saved ? JSON.parse(saved) : DEFAULT_CONFIG;
  });

  const [voiceInfo, setVoiceInfo] = useState<VoiceResponse | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [isPlayingVoice, setIsPlayingVoice] = useState(false);
  const [synthesizeVoice, setSynthesizeVoice] = useState(true);
  const [autoPlayVoice, setAutoPlayVoice] = useState(true);

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isMetricsOpen, setIsMetricsOpen] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Load server status & voices on mount
  useEffect(() => {
    const initBackend = async () => {
      try {
        await checkHealth();
        setIsConnected(true);
        const voices = await fetchVoices();
        setVoiceInfo(voices);
      } catch (err) {
        console.warn('Backend not yet ready or unreachable:', err);
        setIsConnected(false);
      }
    };
    initBackend();
    const interval = setInterval(initBackend, 10000);
    return () => clearInterval(interval);
  }, []);

  // Save messages to local storage
  useEffect(() => {
    localStorage.setItem('voiceai_chat_history', JSON.stringify(messages));
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Save config
  const handleSaveConfig = (newConfig: LLMConfig) => {
    setConfig(newConfig);
    localStorage.setItem('voiceai_llm_config', JSON.stringify(newConfig));
  };

  const handleSendMessage = async (text: string) => {
    if (!text.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: `usr_${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    const tempBotMessageId = `bot_${Date.now()}`;
    const placeholderBotMessage: ChatMessage = {
      id: tempBotMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      isSynthesizing: true,
    };

    setMessages((prev) => [...prev, userMessage, placeholderBotMessage]);
    setIsLoading(true);
    setIsSynthesizing(synthesizeVoice);

    try {
      const response = await sendChatMessage(
        text,
        messages,
        config,
        synthesizeVoice
      );

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === tempBotMessageId
            ? {
                ...msg,
                content: response.reply,
                audioUrl: response.audio_url,
                metrics: response.metrics,
                isSynthesizing: false,
              }
            : msg
        )
      );
    } catch (err: any) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === tempBotMessageId
            ? {
                ...msg,
                content: `Sorry, an error occurred: ${err.message || 'Could not reach server'}. Please ensure FastAPI backend is running on http://localhost:8000.`,
                isSynthesizing: false,
              }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
      setIsSynthesizing(false);
    }
  };

  const handleClearChat = () => {
    if (window.confirm('Clear current conversation history?')) {
      setMessages([]);
      localStorage.removeItem('voiceai_chat_history');
    }
  };

  return (
    <div className="app-container">
      {/* Top Header Navigation */}
      <Header
        isPlayingVoice={isPlayingVoice}
        isSynthesizing={isSynthesizing}
        autoPlayVoice={autoPlayVoice}
        onToggleAutoPlay={setAutoPlayVoice}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenMetrics={() => setIsMetricsOpen(true)}
        voiceInfo={voiceInfo}
        isConnected={isConnected}
      />

      {/* Central Chat Card */}
      <main className="chat-main-card glass-panel">
        <div className="messages-scroll-area">
          {messages.length === 0 ? (
            <div className="welcome-hero-container">
              <div className="welcome-avatar-glow">
                <Bot size={36} />
              </div>
              <h2 className="welcome-title">How can I assist you today?</h2>
              <p className="welcome-subtitle">
                Ask any question below. Responses are generated with intelligent AI and synthesized in real-time using our neural SpeechT5 voice model.
              </p>

              <PromptSuggestions onSelectPrompt={handleSendMessage} />
            </div>
          ) : (
            messages.map((msg, idx) => (
              <ChatMessageBubble
                key={msg.id || idx}
                message={msg}
                onPlayStateChange={setIsPlayingVoice}
                autoPlay={autoPlayVoice && idx === messages.length - 1 && msg.role === 'assistant'}
              />
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <ChatInput
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          synthesizeVoice={synthesizeVoice}
          onToggleVoice={setSynthesizeVoice}
          onClearChat={handleClearChat}
          hasMessages={messages.length > 0}
        />
      </main>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        config={config}
        onSaveConfig={handleSaveConfig}
      />

      {/* System Metrics Drawer */}
      <MetricsDrawer
        isOpen={isMetricsOpen}
        onClose={() => setIsMetricsOpen(false)}
      />
    </div>
  );
};

export default App;
