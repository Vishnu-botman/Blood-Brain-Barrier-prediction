import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import CompoundCard from './components/CompoundCard';
import ComparisonMatrix from './components/ComparisonMatrix';
import BenchmarkModal from './components/BenchmarkModal';
import InfoModal from './components/InfoModal';

export default function App() {
  const [catalog, setCatalog] = useState([]);
  const [presets, setPresets] = useState([]);
  const [backendOnline, setBackendOnline] = useState(false);
  const [benchmarkModalOpen, setBenchmarkModalOpen] = useState(false);
  const [infoModalOpen, setInfoModalOpen] = useState(false);

  // Slots state (1 to 3 slots)
  const [slots, setSlots] = useState([
    {
      id: 1,
      mode: 'catalog',
      selectedIndex: 0,
      customSmiles: '',
      data: null,
      loading: false,
      error: null,
    },
    {
      id: 2,
      mode: 'catalog',
      selectedIndex: 1,
      customSmiles: '',
      data: null,
      loading: false,
      error: null,
    },
  ]);

  // Initial backend health check & catalog loading
  useEffect(() => {
    // 1. Health check
    fetch('/api/health')
      .then((res) => res.json())
      .then((data) => {
        if (data.status === 'online') {
          setBackendOnline(true);
        }
      })
      .catch((err) => {
        console.warn('Backend offline or still starting up:', err);
        setBackendOnline(false);
      });

    // 2. Load compounds catalog (all 2,050 compounds)
    fetch('/api/compounds?limit=2500')
      .then((res) => res.json())
      .then((data) => {
        if (data.compounds && data.compounds.length > 0) {
          setCatalog(data.compounds);

          // Find Propanolol & Ondansetron indices
          const propIdx = data.compounds.findIndex((c) =>
            c.name.toLowerCase().includes('propanolol')
          );
          const ondanIdx = data.compounds.findIndex((c) =>
            c.name.toLowerCase().includes('ondansetron')
          );

          const idx1 = propIdx >= 0 ? propIdx : 0;
          const idx2 = ondanIdx >= 0 ? ondanIdx : Math.min(1, data.compounds.length - 1);

          setSlots((prev) => [
            { ...prev[0], selectedIndex: idx1 },
            { ...prev[1], selectedIndex: idx2 },
          ]);

          // Trigger initial screening for the default 2 molecules
          screenCompound(1, data.compounds[idx1].smiles, data.compounds[idx1].name, data.compounds[idx1].split_tag, data.compounds[idx1].ground_truth);
          screenCompound(2, data.compounds[idx2].smiles, data.compounds[idx2].name, data.compounds[idx2].split_tag, data.compounds[idx2].ground_truth);
        }
      })
      .catch((err) => console.error('Failed to load catalog:', err));

    // 3. Load presets
    fetch('/api/presets')
      .then((res) => res.json())
      .then((data) => {
        if (data.presets) {
          setPresets(data.presets);
        }
      })
      .catch((err) => console.error('Failed to load presets:', err));
  }, []);

  // Screen a specific compound slot
  const screenCompound = async (slotId, smiles, name, split_tag, ground_truth) => {
    setSlots((prev) =>
      prev.map((s) =>
        s.id === slotId ? { ...s, loading: true, error: null } : s
      )
    );

    try {
      const res = await fetch('/api/screen', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles, name, split_tag, ground_truth }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned error ${res.status}`);
      }

      const result = await res.json();
      setSlots((prev) =>
        prev.map((s) =>
          s.id === slotId
            ? { ...s, data: result, loading: false, error: null }
            : s
        )
      );
    } catch (err) {
      setSlots((prev) =>
        prev.map((s) =>
          s.id === slotId
            ? { ...s, loading: false, error: err.message || 'Screening failed' }
            : s
        )
      );
    }
  };

  const updateSlot = (slotId, updates) => {
    setSlots((prev) =>
      prev.map((s) => (s.id === slotId ? { ...s, ...updates } : s))
    );
  };

  const removeSlot = (slotId) => {
    if (slots.length > 1) {
      setSlots((prev) => prev.filter((s) => s.id !== slotId));
    }
  };

  const addSlot = () => {
    if (slots.length < 3 && catalog.length > 0) {
      const newId = Date.now();
      // Pick a compound not currently active
      const activeIndices = new Set(slots.map((s) => s.selectedIndex));
      let pickIdx = catalog.findIndex((_, idx) => !activeIndices.has(idx));
      if (pickIdx < 0) pickIdx = 0;

      const pick = catalog[pickIdx];
      const newSlot = {
        id: newId,
        mode: 'catalog',
        selectedIndex: pickIdx,
        customSmiles: '',
        data: null,
        loading: false,
        error: null,
      };

      setSlots((prev) => [...prev, newSlot]);
      screenCompound(newId, pick.smiles, pick.name, pick.split_tag, pick.ground_truth);
    }
  };

  const screenAll = () => {
    slots.forEach((s) => {
      if (s.mode === 'catalog' && catalog[s.selectedIndex]) {
        const item = catalog[s.selectedIndex];
        screenCompound(s.id, item.smiles, item.name, item.split_tag, item.ground_truth);
      } else if (s.mode === 'custom' && s.customSmiles.trim()) {
        screenCompound(s.id, s.customSmiles.trim(), `Custom Molecule #${s.id}`, 'Novel / Custom', null);
      }
    });
  };

  // Preset loaders
  const loadPresetPair = (name1, name2) => {
    const idx1 = catalog.findIndex((c) => c.name.toLowerCase().includes(name1.toLowerCase()));
    const idx2 = catalog.findIndex((c) => c.name.toLowerCase().includes(name2.toLowerCase()));

    const c1 = idx1 >= 0 ? catalog[idx1] : catalog[0];
    const c2 = idx2 >= 0 ? catalog[idx2] : catalog[Math.min(1, catalog.length - 1)];

    const newSlots = [
      { id: 1, mode: 'catalog', selectedIndex: idx1 >= 0 ? idx1 : 0, customSmiles: '', data: null, loading: false, error: null },
      { id: 2, mode: 'catalog', selectedIndex: idx2 >= 0 ? idx2 : 1, customSmiles: '', data: null, loading: false, error: null },
    ];

    setSlots(newSlots);
    if (c1) screenCompound(1, c1.smiles, c1.name, c1.split_tag, c1.ground_truth);
    if (c2) screenCompound(2, c2.smiles, c2.name, c2.split_tag, c2.ground_truth);
  };

  const gridClass = `compounds-grid compounds-grid-${slots.length}`;

  return (
    <div className="app-container">
      {/* Top Header */}
      <Header
        onOpenBenchmark={() => setBenchmarkModalOpen(true)}
        onOpenInfo={() => setInfoModalOpen(true)}
        backendOnline={backendOnline}
      />

      {/* Control Bar */}
      <div className="control-bar">
        <div className="control-bar-top">
          <div className="section-label">
            <span>🔬</span>
            <span>Screening Engine & Multi-Compound Comparator</span>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)', fontWeight: 'normal' }}>
              ({slots.length}/3 Active Slots)
            </span>
          </div>

          <div className="btn-group">
            <button className="btn btn-primary" onClick={screenAll}>
              <span>⚡</span> Screen All Compounds
            </button>
            <button
              className="btn btn-outline"
              onClick={addSlot}
              disabled={slots.length >= 3}
            >
              <span>➕</span> Add Compound Slot
            </button>
            <button
              className="btn btn-outline"
              onClick={() => loadPresetPair('propanolol', 'ondansetron')}
            >
              <span>🔄</span> Reset to Default
            </button>
          </div>
        </div>

        {/* Quick Presets Strip */}
        <div className="presets-strip">
          <span className="presets-label">Compare Presets:</span>
          <button
            className="preset-chip"
            style={{ borderColor: 'rgba(239, 68, 68, 0.35)', color: '#f87171' }}
            onClick={() => loadPresetPair('amoxicillin', 'ampicillin')}
          >
            🔴 Impermeable (BBB−): Amoxicillin & Ampicillin
          </button>
          <button
            className="preset-chip"
            style={{ borderColor: 'rgba(245, 158, 11, 0.35)', color: '#fbbf24' }}
            onClick={() => loadPresetPair('etoposide', 'nafcillin')}
          >
            🟡 Borderline (UNCERTAIN): Etoposide & Nafcillin
          </button>
          <button
            className="preset-chip"
            style={{ borderColor: 'rgba(16, 185, 129, 0.35)', color: '#34d399' }}
            onClick={() => loadPresetPair('propanolol', 'diazepam')}
          >
            🟢 Permeable (BBB+): Propranolol & Diazepam
          </button>
          <button
            className="preset-chip"
            onClick={() => loadPresetPair('propanolol', 'ondansetron')}
          >
            🧪 Train vs Test Split
          </button>
        </div>
      </div>

      {/* Side-by-side Compound Cards */}
      <main className={gridClass}>
        {slots.map((slot, index) => (
          <CompoundCard
            key={slot.id}
            slot={slot}
            index={index}
            totalSlots={slots.length}
            catalog={catalog}
            onUpdateSlot={updateSlot}
            onRemoveSlot={removeSlot}
            onScreenCompound={screenCompound}
          />
        ))}
      </main>

      {/* Cross-Compound Comparison Matrix (if >= 2 compounds) */}
      <ComparisonMatrix slots={slots} />

      {/* Footer */}
      <footer style={{ marginTop: '48px', textAlign: 'center', color: 'var(--text-dim)', fontSize: '0.8rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '20px' }}>
        <p>
          MoleculeNet BBBP Track 3 Screening System • 4-Model Consensus Ensemble (GCN, GAT, GraphSAGE, Random Forest).
        </p>
        <p style={{ marginTop: '4px' }}>
          Probabilities are uncalibrated in-silico screening estimates. Wet-lab experimental confirmation is required prior to clinical use.
        </p>
      </footer>

      {/* Modals */}
      <BenchmarkModal
        isOpen={benchmarkModalOpen}
        onClose={() => setBenchmarkModalOpen(false)}
      />
      <InfoModal
        isOpen={infoModalOpen}
        onClose={() => setInfoModalOpen(false)}
      />
    </div>
  );
}
