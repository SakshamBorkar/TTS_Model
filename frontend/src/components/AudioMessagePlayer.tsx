import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, RotateCcw, Download } from 'lucide-react';

interface AudioMessagePlayerProps {
  audioUrl: string;
  onPlayStateChange?: (isPlaying: boolean) => void;
  autoPlay?: boolean;
}

export const AudioMessagePlayer: React.FC<AudioMessagePlayerProps> = ({
  audioUrl,
  onPlayStateChange,
  autoPlay = false,
}) => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleLoadedMetadata = () => {
      setDuration(audio.duration || 0);
      if (autoPlay) {
        audio.play().catch(() => {
          // Browser auto-play policy might block if unprompted
          setIsPlaying(false);
          onPlayStateChange?.(false);
        });
      }
    };

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime || 0);
    };

    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
      onPlayStateChange?.(false);
    };

    const handlePlay = () => {
      setIsPlaying(true);
      onPlayStateChange?.(true);
    };

    const handlePause = () => {
      setIsPlaying(false);
      onPlayStateChange?.(false);
    };

    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);

    return () => {
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
    };
  }, [audioUrl, autoPlay, onPlayStateChange]);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play().catch(console.error);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio) return;
    const newTime = parseFloat(e.target.value);
    audio.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const handleReplay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = 0;
    audio.play().catch(console.error);
  };

  const formatTime = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${mins}:${s < 10 ? '0' : ''}${s}`;
  };

  return (
    <div className="audio-player-widget">
      <audio ref={audioRef} src={audioUrl} preload="metadata" />

      <button
        onClick={togglePlay}
        className={`play-pause-btn ${isPlaying ? 'playing' : ''}`}
        title={isPlaying ? 'Pause Voice' : 'Play Voice'}
      >
        {isPlaying ? <Pause size={15} /> : <Play size={15} />}
      </button>

      <div className="audio-slider-container">
        <input
          type="range"
          min={0}
          max={duration || 100}
          step={0.01}
          value={currentTime}
          onChange={handleSeek}
          className="audio-seek-slider"
        />
        <div className="audio-time-label">
          <span>{formatTime(currentTime)}</span>
          <span>/</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      <div className="audio-actions">
        <button onClick={handleReplay} className="audio-action-btn" title="Replay">
          <RotateCcw size={14} />
        </button>
        <a
          href={audioUrl}
          download="synthesized_response.wav"
          className="audio-action-btn"
          title="Download WAV"
        >
          <Download size={14} />
        </a>
      </div>
    </div>
  );
};
