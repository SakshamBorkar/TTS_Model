import React, { useState } from 'react';
import { X, Key, Cpu, Server, MessageSquare, CheckCircle, AlertCircle } from 'lucide-react';
import type { LLMConfig, LLMProvider } from '../types';
import { getApiBaseUrl, setApiBaseUrl, checkHealth } from '../api';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: LLMConfig;
  onSaveConfig: (config: LLMConfig) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  config,
  onSaveConfig,
}) => {
  const [formData, setFormData] = useState<LLMConfig>({ ...config });
  const [apiUrl, setApiUrlState] = useState<string>(getApiBaseUrl());
  const [showKey, setShowKey] = useState(false);
  const [testStatus, setTestStatus] = useState<'idle' | 'testing' | 'success' | 'error'>('idle');
  const [testMessage, setTestMessage] = useState('');

  if (!isOpen) return null;

  const handleProviderChange = (provider: LLMProvider) => {
    let defaultModel = '';
    let defaultUrl = '';

    if (provider === 'openai') {
      defaultModel = 'gpt-4o-mini';
    } else if (provider === 'groq') {
      defaultModel = 'qwen/qwen3.8-27b';
      defaultUrl = 'https://api.groq.com/openai/v1';
    } else if (provider === 'ollama') {
      defaultModel = 'llama3.2';
      defaultUrl = 'http://localhost:11434/v1';
    }

    setFormData((prev) => ({
      ...prev,
      provider,
      modelName: defaultModel || prev.modelName,
      baseUrl: defaultUrl || prev.baseUrl,
    }));
  };

  const handleTestConnection = async () => {
    setTestStatus('testing');
    setTestMessage('Testing backend server connection...');
    try {
      setApiBaseUrl(apiUrl.trim());
      const res = await checkHealth();
      setTestStatus('success');
      setTestMessage(`Connected! TTS Model loaded on ${res.device.toUpperCase()}.`);
    } catch (err: any) {
      setTestStatus('error');
      setTestMessage(err.message || 'Could not connect to FastAPI server.');
    }
  };

  const handleSave = () => {
    setApiBaseUrl(apiUrl.trim());
    onSaveConfig(formData);
    onClose();
  };

  return (
    <div className="modal-backdrop">
      <div className="modal-card glass-panel animate-fade-in">
        <div className="modal-header">
          <div className="modal-title-group">
            <Cpu size={20} className="modal-icon" />
            <h2>Assistant & LLM Settings</h2>
          </div>
          <button onClick={onClose} className="modal-close-btn">
            <X size={20} />
          </button>
        </div>

        <div className="modal-body">
          {/* Backend URL */}
          <div className="form-group">
            <label className="form-label">
              <Server size={14} />
              <span>FastAPI Backend URL</span>
            </label>
            <div className="input-with-action">
              <input
                type="text"
                value={apiUrl}
                onChange={(e) => setApiUrlState(e.target.value)}
                placeholder="http://localhost:8000"
                className="form-input"
              />
              <button
                type="button"
                onClick={handleTestConnection}
                disabled={testStatus === 'testing'}
                className="test-btn"
              >
                {testStatus === 'testing' ? 'Testing...' : 'Test Connection'}
              </button>
            </div>
            {testStatus === 'success' && (
              <p className="status-msg success">
                <CheckCircle size={13} /> {testMessage}
              </p>
            )}
            {testStatus === 'error' && (
              <p className="status-msg error">
                <AlertCircle size={13} /> {testMessage}
              </p>
            )}
          </div>

          {/* LLM Provider Selection */}
          <div className="form-group">
            <label className="form-label">
              <Cpu size={14} />
              <span>LLM Provider</span>
            </label>
            <div className="provider-grid">
              {[
                { id: 'offline', name: 'Built-in Assistant', desc: 'Offline rule-based support engine (Zero setup/key)' },
                { id: 'openai', name: 'OpenAI', desc: 'GPT-4o Mini / GPT-4o' },
                { id: 'groq', name: 'Groq', desc: 'Ultra-fast Llama 3.3 70B' },
                { id: 'ollama', name: 'Ollama', desc: 'Local open-weights models' },
                { id: 'custom', name: 'Custom OpenAI-API', desc: 'Any OpenAI-compatible server' },
              ].map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => handleProviderChange(p.id as LLMProvider)}
                  className={`provider-card ${formData.provider === p.id ? 'active-provider' : ''}`}
                >
                  <span className="p-name">{p.name}</span>
                  <span className="p-desc">{p.desc}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Provider Specific Inputs */}
          {formData.provider !== 'offline' && (
            <>
              {/* API Key */}
              {formData.provider !== 'ollama' && (
                <div className="form-group">
                  <label className="form-label">
                    <Key size={14} />
                    <span>{formData.provider.toUpperCase()} API Key</span>
                  </label>
                  <div className="input-with-action">
                    <input
                      type={showKey ? 'text' : 'password'}
                      value={formData.apiKey}
                      onChange={(e) =>
                        setFormData({ ...formData, apiKey: e.target.value })
                      }
                      placeholder={`Enter your ${formData.provider} API key`}
                      className="form-input"
                    />
                    <button
                      type="button"
                      onClick={() => setShowKey(!showKey)}
                      className="test-btn"
                    >
                      {showKey ? 'Hide' : 'Show'}
                    </button>
                  </div>
                  <span className="form-hint">Stored locally in your browser session.</span>
                </div>
              )}

              {/* Model Name */}
              <div className="form-group">
                <label className="form-label">
                  <span>Model Identifier</span>
                </label>
                <input
                  type="text"
                  value={formData.modelName}
                  onChange={(e) =>
                    setFormData({ ...formData, modelName: e.target.value })
                  }
                  placeholder="e.g. gpt-4o-mini, llama-3.3-70b-versatile, llama3.2"
                  className="form-input"
                />
              </div>

              {/* Base URL */}
              {(formData.provider === 'ollama' || formData.provider === 'custom' || formData.provider === 'groq') && (
                <div className="form-group">
                  <label className="form-label">
                    <span>Base API Endpoint</span>
                  </label>
                  <input
                    type="text"
                    value={formData.baseUrl}
                    onChange={(e) =>
                      setFormData({ ...formData, baseUrl: e.target.value })
                    }
                    placeholder="https://api.groq.com/openai/v1 or http://localhost:11434/v1"
                    className="form-input"
                  />
                </div>
              )}
            </>
          )}

          {/* System Prompt */}
          <div className="form-group">
            <label className="form-label">
              <MessageSquare size={14} />
              <span>Voice Assistant Personality / System Instructions</span>
            </label>
            <textarea
              rows={3}
              value={formData.systemPrompt}
              onChange={(e) =>
                setFormData({ ...formData, systemPrompt: e.target.value })
              }
              placeholder="Instructions for the AI assistant..."
              className="form-textarea"
            />
          </div>
        </div>

        <div className="modal-footer">
          <button onClick={onClose} className="cancel-btn">
            Cancel
          </button>
          <button onClick={handleSave} className="save-btn">
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
};
