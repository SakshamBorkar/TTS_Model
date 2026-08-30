import React, { useState, useRef, useEffect } from 'react';
import { Send, Mic, MicOff, Volume2, VolumeX, Trash2 } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
  synthesizeVoice: boolean;
  onToggleVoice: (enabled: boolean) => void;
  onClearChat: () => void;
  hasMessages: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  isLoading,
  synthesizeVoice,
  onToggleVoice,
  onClearChat,
  hasMessages,
}) => {
  const [text, setText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    // Check Web Speech API support
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onresult = (event: any) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          transcript += event.results[i][0].transcript;
        }
        setText(transcript);
      };

      recognition.onerror = (event: any) => {
        console.warn('Speech recognition error:', event.error);
        setIsRecording(false);
      };

      recognition.onend = () => {
        setIsRecording(false);
      };

      recognitionRef.current = recognition;
    }
  }, []);

  const handleToggleRecord = () => {
    if (!recognitionRef.current) {
      alert('Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari.');
      return;
    }

    if (isRecording) {
      recognitionRef.current.stop();
      setIsRecording(false);
    } else {
      setText('');
      recognitionRef.current.start();
      setIsRecording(true);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = () => {
    if (!text.trim() || isLoading) return;
    if (isRecording && recognitionRef.current) {
      recognitionRef.current.stop();
      setIsRecording(false);
    }
    onSendMessage(text.trim());
    setText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    // Auto-adjust height up to max
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 140)}px`;
  };

  return (
    <div className="chat-input-wrapper">
      <div className="input-toolbar">
        <div className="toolbar-left">
          <button
            onClick={() => onToggleVoice(!synthesizeVoice)}
            className={`tool-pill-btn ${synthesizeVoice ? 'active-voice' : ''}`}
            title={synthesizeVoice ? 'Voice synthesis enabled' : 'Voice synthesis muted'}
          >
            {synthesizeVoice ? <Volume2 size={14} /> : <VolumeX size={14} />}
            <span>{synthesizeVoice ? 'Voice On' : 'Voice Off'}</span>
          </button>
        </div>

        <div className="toolbar-right">
          {hasMessages && (
            <button
              onClick={onClearChat}
              className="clear-chat-btn"
              title="Clear conversation"
            >
              <Trash2 size={13} />
              <span>Clear</span>
            </button>
          )}
        </div>
      </div>

      <div className={`input-box-container ${isRecording ? 'recording-active' : ''}`}>
        <textarea
          ref={textareaRef}
          value={text}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder={
            isRecording
              ? 'Listening... Speak your question clearly into your microphone'
              : 'Ask a question or type a message (Enter to send)...'
          }
          rows={1}
          disabled={isLoading}
          className="chat-textarea"
        />

        <div className="input-actions-group">
          {/* Speech-to-Text Mic Button */}
          <button
            type="button"
            onClick={handleToggleRecord}
            className={`mic-button ${isRecording ? 'is-recording' : ''}`}
            title={isRecording ? 'Stop listening' : 'Voice dictation'}
            disabled={isLoading}
          >
            {isRecording ? <MicOff size={18} /> : <Mic size={18} />}
          </button>

          {/* Send Button */}
          <button
            type="button"
            onClick={handleSubmit}
            disabled={!text.trim() || isLoading}
            className={`send-button ${text.trim() && !isLoading ? 'can-send' : ''}`}
            title="Send Message"
          >
            <Send size={16} />
          </button>
        </div>
      </div>
    </div>
  );
};
