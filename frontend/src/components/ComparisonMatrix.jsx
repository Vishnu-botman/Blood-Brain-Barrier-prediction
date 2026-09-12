import React from 'react';

export default function ComparisonMatrix({ slots }) {
  const validSlots = slots.filter((s) => s.data && !s.loading && !s.error);

  if (validSlots.length < 2) return null;

  const models = [
    { key: 'ensemble', label: 'Ensemble Proposed' },
    { key: 'gcn', label: 'GCN Baseline' },
    { key: 'gat', label: 'GAT (4 Heads)' },
    { key: 'sage', label: 'GraphSAGE' },
    { key: 'rf', label: 'Random Forest' },
  ];

  const getProb = (data, modelKey) => {
    if (modelKey === 'ensemble') return data.ensemble_probability;
    return data.model_probabilities?.[modelKey] || 0;
  };

  return (
    <section className="matrix-section">
      <div className="matrix-header">
        <h2 className="matrix-title">
          <span>📊</span> Side-by-Side Comparative Matrix
        </h2>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
          Comparing {validSlots.length} Active Molecular Candidates
        </span>
      </div>

      {/* Comparison Table */}
      <div className="table-responsive">
        <table className="matrix-table">
          <thead>
            <tr>
              <th style={{ minWidth: '160px' }}>Attribute</th>
              {validSlots.map((slot, i) => (
                <th key={slot.id} style={{ minWidth: '220px' }}>
                  Molecule #{i + 1}: {slot.data.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Dataset Split</strong></td>
              {validSlots.map((s) => (
                <td key={s.id}>
                  <span className={`badge ${
                    s.data.split_tag.includes('Train')
                      ? 'badge-train'
                      : s.data.split_tag.includes('Test')
                      ? 'badge-test'
                      : 'badge-valid'
                  }`}>
                    {s.data.split_tag}
                  </span>
                </td>
              ))}
            </tr>

            <tr>
              <td><strong>Screening Decision</strong></td>
              {validSlots.map((s) => {
                const dec = s.data.decision;
                const badgeColor = dec === 'BBB+' ? '#10b981' : dec === 'BBB-' ? '#ef4444' : '#f59e0b';
                return (
                  <td key={s.id}>
                    <span style={{ 
                      fontWeight: '700', 
                      color: badgeColor,
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '5px'
                    }}>
                      {dec === 'BBB+' ? '🟢' : dec === 'BBB-' ? '🔴' : '🟡'} {dec}
                    </span>
                  </td>
                );
              })}
            </tr>

            <tr>
              <td><strong>Known Ground Truth</strong></td>
              {validSlots.map((s) => (
                <td key={s.id}>
                  {s.data.ground_truth ? (
                    <code>{s.data.ground_truth}</code>
                  ) : (
                    <span style={{ color: 'var(--text-dim)' }}>Unlabeled</span>
                  )}
                </td>
              ))}
            </tr>

            <tr>
              <td><strong>Ensemble Probability</strong></td>
              {validSlots.map((s) => (
                <td key={s.id} style={{ fontFamily: 'var(--font-mono)', fontWeight: '700', fontSize: '1rem', color: '#38bdf8' }}>
                  {(s.data.ensemble_probability * 100).toFixed(1)}%
                </td>
              ))}
            </tr>

            <tr>
              <td><strong>GCN Baseline</strong></td>
              {validSlots.map((s) => (
                <td key={s.id} style={{ fontFamily: 'var(--font-mono)' }}>
                  {(s.data.model_probabilities?.gcn * 100).toFixed(1)}%
                </td>
              ))}
            </tr>

            <tr>
              <td><strong>GAT Model</strong></td>
              {validSlots.map((s) => (
                <td key={s.id} style={{ fontFamily: 'var(--font-mono)' }}>
                  {(s.data.model_probabilities?.gat * 100).toFixed(1)}%
                </td>
              ))}
            </tr>

            <tr>
              <td><strong>GraphSAGE</strong></td>
              {validSlots.map((s) => (
                <td key={s.id} style={{ fontFamily: 'var(--font-mono)' }}>
                  {(s.data.model_probabilities?.sage * 100).toFixed(1)}%
                </td>
              ))}
            </tr>

            <tr>
              <td><strong>Random Forest</strong></td>
              {validSlots.map((s) => (
                <td key={s.id} style={{ fontFamily: 'var(--font-mono)' }}>
                  {(s.data.model_probabilities?.rf * 100).toFixed(1)}%
                </td>
              ))}
            </tr>

            <tr>
              <td><strong>Model Agreement</strong></td>
              {validSlots.map((s) => (
                <td key={s.id} style={{ fontWeight: '600' }}>
                  {s.data.agreement}
                </td>
              ))}
            </tr>

            <tr>
              <td><strong>Training Tanimoto Proximity</strong></td>
              {validSlots.map((s) => (
                <td key={s.id}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: '600' }}>
                    {s.data.nearest_training_tanimoto?.toFixed(3)}
                  </span>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>
                    {s.data.nearest_training_tanimoto >= 0.999 ? '(Exact training match)' : '(Held-out / novel)'}
                  </div>
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>

      {/* Visual Model Comparison Bars */}
      <div style={{ marginTop: '28px', background: 'var(--bg-card-inner)', borderRadius: 'var(--radius-lg)', padding: '20px' }}>
        <h4 style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          Cross-Compound Probability Comparison
        </h4>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {models.map((m) => (
            <div key={m.key}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '6px' }}>
                <span style={{ fontWeight: '600', color: 'var(--text-main)' }}>{m.label}</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {validSlots.map((slot, idx) => {
                  const prob = getProb(slot.data, m.key);
                  const colors = ['#0284c7', '#8b5cf6', '#10b981'];
                  const barColor = colors[idx % colors.length];
                  return (
                    <div key={slot.id} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ width: '130px', fontSize: '0.75rem', color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        #{idx + 1} {slot.data.name}
                      </span>
                      <div style={{ flex: 1, height: '8px', background: 'rgba(255,255,255,0.06)', borderRadius: '4px', overflow: 'hidden' }}>
                        <div
                          style={{
                            height: '100%',
                            width: `${Math.min(Math.max(prob * 100, 2), 100)}%`,
                            backgroundColor: barColor,
                            borderRadius: '4px',
                            transition: 'width 0.4s ease',
                          }}
                        />
                      </div>
                      <span style={{ width: '50px', textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: barColor, fontWeight: '600' }}>
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
