import React, { useEffect, useRef } from 'react';

interface AudioVisualizerProps {
  isPlaying: boolean;
  isSynthesizing?: boolean;
}

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({ isPlaying, isSynthesizing }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let phase = 0;

    const render = () => {
      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);

      const numBars = 36;
      const barWidth = 3;
      const barGap = 4;
      const totalWidth = numBars * (barWidth + barGap);
      const startX = (width - totalWidth) / 2;
      const centerY = height / 2;

      phase += 0.08;

      for (let i = 0; i < numBars; i++) {
        let barHeight = 4;

        if (isPlaying) {
          const sinVal = Math.sin(phase + i * 0.35);
          const cosVal = Math.cos(phase * 1.3 - i * 0.2);
          const noise = (sinVal + cosVal + 2) / 4;
          barHeight = Math.max(4, noise * (height * 0.75));
        } else if (isSynthesizing) {
          const wave = Math.sin(phase * 2 + i * 0.4);
          barHeight = Math.max(4, Math.abs(wave) * (height * 0.45));
        } else {
          // Ambient idle breathing wave
          const subtle = Math.sin(phase * 0.5 + i * 0.2);
          barHeight = 4 + Math.max(0, subtle * 6);
        }

        const x = startX + i * (barWidth + barGap);
        const y = centerY - barHeight / 2;

        // Gradient for visualizer
        const gradient = ctx.createLinearGradient(0, centerY - barHeight / 2, 0, centerY + barHeight / 2);
        if (isPlaying) {
          gradient.addColorStop(0, '#38bdf8');
          gradient.addColorStop(0.5, '#818cf8');
          gradient.addColorStop(1, '#c084fc');
        } else if (isSynthesizing) {
          gradient.addColorStop(0, '#f59e0b');
          gradient.addColorStop(1, '#ef4444');
        } else {
          gradient.addColorStop(0, 'rgba(129, 140, 248, 0.3)');
          gradient.addColorStop(1, 'rgba(192, 132, 252, 0.2)');
        }

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, barHeight, 2);
        ctx.fill();
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [isPlaying, isSynthesizing]);

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <canvas
        ref={canvasRef}
        width={260}
        height={38}
        style={{
          borderRadius: '12px',
          background: 'rgba(15, 23, 42, 0.4)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
        }}
      />
    </div>
  );
};
