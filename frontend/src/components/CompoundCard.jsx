import React, { useState } from 'react';
import SearchableSelect from './SearchableSelect';

export default function CompoundCard({
  slot,
  index,
  totalSlots,
  catalog,
  onUpdateSlot,
  onRemoveSlot,
  onScreenCompound,
}) {
  const [showAttention, setShowAttention] = useState(false);
  const { data, loading, error, mode, customSmiles, selectedIndex } = slot;

  const handleModeChange = (newMode) => {
    onUpdateSlot(slot.id, { mode: newMode });
  };

  const handleCatalogSelect = (idx, item) => {
    if (item) {
      onUpdateSlot(slot.id, {
        selectedIndex: idx,
        name: item.name,
        smiles: item.smiles,
        split_tag: item.split_tag,
        ground_truth: item.ground_truth,
      });
      onScreenCompound(slot.id, item.smiles, item.name, item.split_tag, item.ground_truth);
    }
  };

  const handleCustomSmilesChange = (e) => {
    const val = e.target.value;
    onUpdateSlot(slot.id, { customSmiles: val });
  };

  const handleCustomScreen = (e) => {
    e.preventDefault();
    if (customSmiles.trim()) {
      onScreenCompound(slot.id, customSmiles.trim(), `Custom Compound #${index + 1}`, 'Novel / Custom', null);
    }
  };

  // Determine split badge style
  const splitClass = data?.split_tag?.includes('Train')
    ? 'badge-train'
    : data?.split_tag?.includes('Test')
    ? 'badge-test'
    : 'badge-valid';

  // Decision details
  const decision = data?.decision;
  const isPositive = decision === 'BBB+';
  const isNegative = decision === 'BBB-';
  const isUncertain = decision === 'UNCERTAIN';

  const bannerClass = isPositive
    ? 'positive'
    : isNegative
    ? 'negative'
    : 'uncertain';

  const bannerIcon = isPositive ? '🟢' : isNegative ? '🔴' : '🟡';
  const bannerLabel = isPositive
    ? 'BBB+ (Permeable)'
    : isNegative
    ? 'BBB− (Impermeable)'
    : 'UNCERTAIN (Abstained)';

  // Ground truth match assessment
  let gtMatchTag = null;
  if (data?.ground_truth) {
    const gtIsPos = data.ground_truth === 'BBB+';
    if (decision === data.ground_truth) {
      gtMatchTag = (
        <span className="badge badge-gt match">
          ✓ Matches Known Label ({data.ground_truth})
        </span>
      );
    } else if (decision === 'UNCERTAIN') {
      gtMatchTag = (
        <span className="badge badge-gt">
          ℹ Abstained (Ground Truth: {data.ground_truth})
        </span>
      );
    } else {
      gtMatchTag = (
        <span className="badge badge-gt diff">
          ⚠ Differs from Ground Truth ({data.ground_truth})
        </span>
      );
    }
  }

  return (
    <div className="compound-card">
      {/* Header */}
      <div className="card-header">
        <div className="card-title-group">
          <span className="card-number">#{index + 1}</span>
          <h3 className="card-title" title={data?.name || `Compound #${index + 1}`}>
            {data?.name || `Compound #${index + 1}`}
          </h3>
        </div>
        {totalSlots > 1 && (
          <button
            className="card-close-btn"
            onClick={() => onRemoveSlot(slot.id)}
            title="Remove compound slot"
          >
            ✕
          </button>
        )}
      </div>

      {/* Input Mode Selector */}
      <div className="mode-switch">
        <button
          type="button"
          className={`mode-tab ${mode === 'catalog' ? 'active' : ''}`}
          onClick={() => handleModeChange('catalog')}
        >
          Catalog (2,050 Compounds)
        </button>
        <button
          type="button"
          className={`mode-tab ${mode === 'custom' ? 'active' : ''}`}
          onClick={() => handleModeChange('custom')}
        >
          Custom SMILES
        </button>
      </div>

      {/* Inputs */}
      {mode === 'catalog' ? (
        <div className="form-group">
          <label className="form-label">Search & Select Molecular Candidate</label>
          <SearchableSelect
            options={catalog}
            selectedIndex={selectedIndex}
            onSelect={handleCatalogSelect}
            placeholder="Type drug name, SMILES, or #num..."
          />
        </div>
      ) : (
        <form onSubmit={handleCustomScreen} className="form-group">
          <label className="form-label">Enter SMILES Notation</label>
          <div style={{ display: 'flex', gap: '8px' }}>
            <input
              type="text"
              className="text-input mono"
              value={customSmiles}
              onChange={handleCustomSmilesChange}
              placeholder="e.g. CCO or c1ccccc1"
            />
            <button type="submit" className="btn btn-primary" style={{ padding: '8px 14px' }}>
              Screen
            </button>
          </div>
        </form>
      )}

      {/* Tags: Split & Ground Truth */}
      <div className="tag-row">
        {data?.split_tag && (
          <span className={`badge ${splitClass}`}>
            📁 {data.split_tag}
          </span>
        )}
        {gtMatchTag}
      </div>

      {/* SMILES Display */}
      {data?.smiles && (
        <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)', wordBreak: 'break-all' }}>
          <strong style={{ color: 'var(--text-muted)' }}>SMILES: </strong>
          <code style={{ fontFamily: 'var(--font-mono)' }}>{data.smiles}</code>
        </div>
      )}

      {/* 2D Molecular Structure SVG */}
      <div className="molecule-viewer">
        {loading ? (
          <div className="molecule-loading">
            <span className="pulse">⏳</span> Generating molecular graph & 2D render...
          </div>
        ) : error ? (
          <div style={{ color: '#ef4444', textAlign: 'center', padding: '16px' }}>
            ⚠️ {error}
          </div>
        ) : data?.svg ? (
          <div
            dangerouslySetInnerHTML={{ __html: data.svg }}
            style={{ width: '100%', display: 'flex', justifyContent: 'center' }}
          />
        ) : (
          <span style={{ color: 'var(--text-dim)' }}>Select a molecule to view structure</span>
        )}
      </div>

      {/* Prediction Output & Metrics */}
      {data && !loading && !error && (
        <>
          {/* Decision Banner */}
          <div className={`decision-banner ${bannerClass}`}>
            <div className="decision-main">
              <span className="decision-icon">{bannerIcon}</span>
              <div className="decision-text">
                <span className="decision-label">{bannerLabel}</span>
                <span className="decision-subtext">
                  {isUncertain ? 'Disagreement or low scaffold similarity' : '4-Model Consensus Estimate'}
                </span>
              </div>
            </div>
            <div className="decision-prob">
              <div className="prob-number">{(data.ensemble_probability * 100).toFixed(1)}%</div>
              <div className="prob-caption">Ensemble BBB+</div>
            </div>
          </div>

          {/* Key Metrics */}
          <div className="metrics-row">
            <div className="metric-box">
              <span className="metric-label">Model Agreement</span>
              <div className="metric-value">{data.agreement}</div>
              <span className="metric-note">Models concurring (≥0.50)</span>
            </div>

            <div className="metric-box">
              <span className="metric-label">Training Tanimoto</span>
              <div className="metric-value">
                {data.nearest_training_tanimoto?.toFixed(2)}
              </div>
              <span className="metric-note" title={data.tanimoto_note}>
                {data.nearest_training_tanimoto >= 0.999 ? '100% Training Identity' : `${(data.nearest_training_tanimoto * 100).toFixed(0)}% Scaffold Proximity`}
              </span>
            </div>
          </div>

          {/* Model Breakdown Bars */}
          <div className="breakdown-section">
            <div className="breakdown-header">
              <span className="breakdown-title">Architectural Consensus</span>
            </div>

            <div className="model-bar-item">
              <div className="model-bar-labels">
                <span className="model-name">GCN Baseline (2-Layer)</span>
                <span className="model-val">{(data.model_probabilities?.gcn * 100).toFixed(1)}%</span>
              </div>
              <div className="model-progress-track">
                <div
                  className="model-progress-fill"
                  style={{ width: `${Math.min(Math.max(data.model_probabilities?.gcn * 100, 2), 100)}%` }}
                />
              </div>
            </div>

            <div className="model-bar-item">
              <div className="model-bar-labels">
                <span className="model-name">GAT (4 Heads)</span>
                <span className="model-val">{(data.model_probabilities?.gat * 100).toFixed(1)}%</span>
              </div>
              <div className="model-progress-track">
                <div
                  className="model-progress-fill"
                  style={{ width: `${Math.min(Math.max(data.model_probabilities?.gat * 100, 2), 100)}%` }}
                />
              </div>
            </div>

            <div className="model-bar-item">
              <div className="model-bar-labels">
                <span className="model-name">GraphSAGE</span>
                <span className="model-val">{(data.model_probabilities?.sage * 100).toFixed(1)}%</span>
              </div>
              <div className="model-progress-track">
                <div
                  className="model-progress-fill"
                  style={{ width: `${Math.min(Math.max(data.model_probabilities?.sage * 100, 2), 100)}%` }}
                />
              </div>
            </div>

            <div className="model-bar-item">
              <div className="model-bar-labels">
                <span className="model-name">Random Forest (Morgan FP)</span>
                <span className="model-val">{(data.model_probabilities?.rf * 100).toFixed(1)}%</span>
              </div>
              <div className="model-progress-track">
                <div
                  className="model-progress-fill"
                  style={{ width: `${Math.min(Math.max(data.model_probabilities?.rf * 100, 2), 100)}%` }}
                />
              </div>
            </div>
          </div>

          {/* GAT Attention Inspector */}
          {data.attention && data.attention.length > 0 && (
            <div className="attention-details">
              <div
                className="attention-summary"
                onClick={() => setShowAttention(!showAttention)}
              >
                <span>🔍 GAT Layer-1 Atom Attention ({data.attention.length} Atoms)</span>
                <span>{showAttention ? '▲ Collapse' : '▼ Inspect'}</span>
              </div>
              {showAttention && (
                <div className="attention-content">
                  <p style={{ fontSize: '0.74rem', color: 'var(--text-dim)', marginBottom: '10px' }}>
                    Sum of incoming multi-head attention weights per atom index (qualitative inspection):
                  </p>
                  <div className="attention-grid">
                    {data.attention.map((at) => {
                      const maxAttn = Math.max(...data.attention.map((a) => a.attention), 0.1);
                      const heightPct = Math.max((at.attention / maxAttn) * 100, 6);
                      return (
                        <div key={at.atom_index} className="attention-bar-col">
                          <div
                            className="attention-bar"
                            style={{ height: `${heightPct}%` }}
                            title={`${at.label}: ${at.attention}`}
                          />
                          <span className="attention-label">{at.label}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
