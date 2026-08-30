import React, { useEffect, useState } from 'react';
import { X, Activity } from 'lucide-react';
import type { SystemMetrics } from '../types';
import { fetchMetrics } from '../api';

interface MetricsDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

export const MetricsDrawer: React.FC<MetricsDrawerProps> = ({ isOpen, onClose }) => {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    fetchMetrics()
      .then((data) => {
        setMetrics(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load metrics:', err);
        setLoading(false);
      });
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop">
      <div className="modal-card metrics-card glass-panel animate-fade-in">
        <div className="modal-header">
          <div className="modal-title-group">
            <Activity size={20} className="modal-icon" />
            <h2>System & TTS Performance Metrics</h2>
          </div>
          <button onClick={onClose} className="modal-close-btn">
            <X size={20} />
          </button>
        </div>

        <div className="modal-body">
          {loading ? (
            <div className="metrics-loading">Fetching real-time statistics...</div>
          ) : metrics ? (
            <div className="metrics-grid">
              <div className="metric-stat-card">
                <span className="stat-label">Total Requests</span>
                <span className="stat-value">{metrics.request_count}</span>
                <span className="stat-sub">({metrics.chat_count} chat queries)</span>
              </div>

              <div className="metric-stat-card">
                <span className="stat-label">Audio Generated</span>
                <span className="stat-value">{metrics.total_audio_seconds}s</span>
                <span className="stat-sub">Cumulative speech duration</span>
              </div>

              <div className="metric-stat-card">
                <span className="stat-label">Inference Time</span>
                <span className="stat-value">{metrics.total_inference_seconds}s</span>
                <span className="stat-sub">Cumulative compute time</span>
              </div>

              <div className="metric-stat-card">
                <span className="stat-label">Device Backend</span>
                <span className="stat-value accent-cyan">{metrics.device.toUpperCase()}</span>
                <span className="stat-sub">Active compute engine</span>
              </div>

              {metrics.latency && (
                <div className="latency-breakdown-card">
                  <h3>Inference Latency Percentiles</h3>
                  <div className="latency-row-list">
                    <div className="latency-item">
                      <span>Mean:</span>
                      <strong>{metrics.latency.mean_s?.toFixed(3)}s</strong>
                    </div>
                    <div className="latency-item">
                      <span>P50 (Median):</span>
                      <strong>{metrics.latency.p50_s?.toFixed(3)}s</strong>
                    </div>
                    <div className="latency-item">
                      <span>P95:</span>
                      <strong>{metrics.latency.p95_s?.toFixed(3)}s</strong>
                    </div>
                    <div className="latency-item">
                      <span>Min / Max:</span>
                      <strong>
                        {metrics.latency.min_s?.toFixed(3)}s / {metrics.latency.max_s?.toFixed(3)}s
                      </strong>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p className="metrics-error">Could not retrieve metrics from backend.</p>
          )}
        </div>

        <div className="modal-footer">
          <button onClick={onClose} className="save-btn">
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
