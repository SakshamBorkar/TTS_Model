import React from 'react';
import { Sparkles, Package, CreditCard, Key, Cpu } from 'lucide-react';

interface PromptSuggestionsProps {
  onSelectPrompt: (prompt: string) => void;
}

const PROMPTS = [
  {
    icon: <Package size={15} color="#38bdf8" />,
    label: 'Order Status',
    text: 'What is the current delivery status of my latest package?',
  },
  {
    icon: <CreditCard size={15} color="#10b981" />,
    label: 'Account Balance',
    text: 'Can you check my current account balance and recent refund?',
  },
  {
    icon: <Key size={15} color="#f59e0b" />,
    label: 'Password Reset',
    text: 'How do I reset the password for my registered account?',
  },
  {
    icon: <Cpu size={15} color="#c084fc" />,
    label: 'TTS Architecture',
    text: 'Explain how the SpeechT5 acoustic model and HiFi-GAN vocoder work together.',
  },
];

export const PromptSuggestions: React.FC<PromptSuggestionsProps> = ({ onSelectPrompt }) => {
  return (
    <div className="prompt-suggestions-wrapper">
      <div className="suggestions-header">
        <Sparkles size={14} className="sparkle-icon" />
        <span>Suggested conversation starters</span>
      </div>
      <div className="prompt-chips-grid">
        {PROMPTS.map((p, idx) => (
          <button
            key={idx}
            onClick={() => onSelectPrompt(p.text)}
            className="prompt-chip"
          >
            <div className="chip-icon-box">{p.icon}</div>
            <div className="chip-text-content">
              <span className="chip-title">{p.label}</span>
              <span className="chip-desc">{p.text}</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};
