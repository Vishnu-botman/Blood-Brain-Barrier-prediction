import React from 'react';

export default function Header({ onOpenBenchmark, onOpenInfo, backendOnline }) {
  return (
    <header className="header">
      <div className="brand-section">
        <div className="brand-icon">🧬</div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <h1 className="brand-title">BBBP Molecular Screening</h1>
            <span className="track-badge">TRACK 3</span>
          </div>
          <p className="brand-subtitle">
            <span>Blood-Brain Barrier Permeability Consensus Engine</span>
            <span style={{ color: 'var(--text-dim)' }}>•</span>
            <span style={{ 
              display: 'inline-flex', 
              alignItems: 'center', 
              gap: '4px',
              color: backendOnline ? 'var(--status-positive)' : 'var(--status-negative)',
              fontSize: '0.75rem',
              fontWeight: '600'
            }}>
              <span style={{ 
                width: '7px', 
                height: '7px', 
                borderRadius: '50%', 
                backgroundColor: backendOnline ? '#10b981' : '#ef4444',
                boxShadow: backendOnline ? '0 0 8px #10b981' : 'none'
              }}></span>
              {backendOnline ? 'ML Backend Online' : 'Backend Connecting...'}
            </span>
          </p>
        </div>
      </div>

      <div className="header-actions">
        <button className="nav-btn" onClick={onOpenBenchmark}>
          <span>🏆</span>
          <span>Leaderboard & Ablation</span>
        </button>
        <button className="nav-btn" onClick={onOpenInfo}>
          <span>ℹ️</span>
          <span>Methodology & Tanimoto</span>
        </button>
      </div>
    </header>
  );
}
