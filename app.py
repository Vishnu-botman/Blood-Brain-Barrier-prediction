"""Streamlit Web UI for BBBP Molecular Screening & Multi-Compound Comparator.

Features:
- Searchable dropdown containing 2,050+ compounds with common names & split tags.
- Side-by-side comparative screening for up to 3 compounds.
- Dynamic 'Add Compound' and individual 'Close / Remove' controls.
- Explicit 'Run Screening Analysis' module trigger with instant evaluation.
- Detailed Tanimoto Similarity explanation & split breakdown ([Train Set], [Test Set], [Valid Set]).
- 2D molecular diagram rendering via RDKit.
- Ensemble consensus breakdown across 2-layer GCN, GAT, GraphSAGE, and Random Forest.
- Cross-compound comparison matrix table and grouped visualization.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import Draw

from chem_data import canonical, graph
from predict import ScreeningSystem

st.set_page_config(
    page_title="BBBP Screening | Multi-Compound Comparator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Premium CSS ---
st.markdown(
    """
    <style>
    /* Card containers */
    .compound-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 18px 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.12);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .compound-card:hover {
        border-color: rgba(59, 130, 246, 0.4);
        box-shadow: 0 6px 20px -2px rgba(59, 130, 246, 0.15);
    }
    
    /* Decision Badges */
    .decision-badge {
        display: inline-block;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 6px 14px;
        border-radius: 9999px;
        letter-spacing: 0.025em;
        text-align: center;
        margin: 8px 0;
    }
    .decision-plus {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: #ffffff;
        box-shadow: 0 2px 10px rgba(16, 185, 129, 0.35);
    }
    .decision-minus {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: #ffffff;
        box-shadow: 0 2px 10px rgba(239, 68, 68, 0.35);
    }
    .decision-uncertain {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: #ffffff;
        box-shadow: 0 2px 10px rgba(245, 158, 11, 0.35);
    }

    /* Small Tags */
    .gt-tag {
        display: inline-block;
        font-size: 0.78rem;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        margin: 2px 2px 2px 0;
    }
    .gt-match {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .gt-diff {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .split-train {
        background-color: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .split-test {
        background-color: rgba(168, 85, 247, 0.15);
        color: #a855f7;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .split-valid {
        background-color: rgba(234, 179, 8, 0.15);
        color: #eab308;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }

    /* Concept Explainer Box */
    .explainer-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 16px 0 24px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_screening_system():
    return ScreeningSystem()


@st.cache_data
def load_compound_catalog():
    candidates = [
        Path("data_sources/BBBP_with_names.csv"),
        Path("data_sources/bbbp.csv"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("Compound dataset not found in data_sources/")

    df = pd.read_csv(path, encoding="utf-8")
    if "name" not in df.columns:
        df["name"] = [f"Compound #{i+1}" for i in range(len(df))]
    if "num" not in df.columns:
        df["num"] = list(range(1, len(df) + 1))
    if "p_np" not in df.columns and "label" in df.columns:
        df["p_np"] = df["label"]

    # Load scaffold splits to accurately tag each compound
    splits_dir = Path("artifacts/splits")
    train_set, valid_set, test_set = set(), set(), set()
    if (splits_dir / "train.csv").exists():
        train_set = set(pd.read_csv(splits_dir / "train.csv")["SMILES"])
    if (splits_dir / "valid.csv").exists():
        valid_set = set(pd.read_csv(splits_dir / "valid.csv")["SMILES"])
    if (splits_dir / "test.csv").exists():
        test_set = set(pd.read_csv(splits_dir / "test.csv")["SMILES"])

    def determine_split(smi):
        c = canonical(smi)
        if c in train_set:
            return "Train Set"
        elif c in test_set:
            return "Test Set"
        elif c in valid_set:
            return "Valid Set"
        return "Novel/Other"

    df["split_tag"] = df["smiles"].apply(determine_split)
    # Formatted display label for dropdown
    df["display_name"] = df.apply(
        lambda r: f"{r['name']} (#{r['num']}) [{r['split_tag']}]", axis=1
    )
    return df


try:
    system = load_screening_system()
    catalog = load_compound_catalog()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Initialization error: {exc}")
    st.code("python run_hackathon.py\nstreamlit run app.py", language="bash")
    st.stop()

# Lookup dicts
NAME_TO_ROW = {row["display_name"]: row for _, row in catalog.iterrows()}
ALL_DISPLAY_NAMES = catalog["display_name"].tolist()

# Pre-identify common examples
PROP_NAME = next((n for n in ALL_DISPLAY_NAMES if "Propanolol" in n), ALL_DISPLAY_NAMES[0])
ATEN_NAME = next((n for n in ALL_DISPLAY_NAMES if "Atenolol" in n), ALL_DISPLAY_NAMES[1])
ONDAN_NAME = next((n for n in ALL_DISPLAY_NAMES if "ondansetron" in n), ALL_DISPLAY_NAMES[2])
ETOP_NAME = next((n for n in ALL_DISPLAY_NAMES if "Etoposide" in n), ALL_DISPLAY_NAMES[3])

# --- Sidebar ---
with st.sidebar:
    st.header("🧬 BBBP Screening System")
    st.markdown(
        """
        **Track 3: Blood-Brain Barrier Screening**
        - **Task**: Binary graph classification (MoleculeNet BBBP).
        - **Ensemble**:
          1. Mandatory **2-layer GCN** baseline
          2. 2-layer **GAT** (4 heads)
          3. 2-layer **GraphSAGE**
          4. **Random Forest** (Morgan fingerprints)
        - **Proposed Method**: Mean ensemble of all 4 models
        """
    )

    st.markdown("---")
    st.subheader("Leaderboard Benchmark")
    metrics_data = {
        "Model": ["GCN (Baseline)", "GAT", "GraphSAGE", "Random Forest", "GNN Consensus", "Proposed Ensemble"],
        "Test ROC-AUC": ["0.8547", "0.8222", "0.8761", "0.9303", "0.8636", "0.9162"],
    }
    st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)
    st.caption("Ensemble outperforms mandatory GCN by **+0.0615** ROC-AUC on seed-42 test split.")

    st.markdown("---")
    st.subheader("💡 What is Tanimoto Similarity?")
    st.markdown(
        """
        - **Tanimoto Similarity** compares molecular Morgan fingerprints (1024-bit structural descriptors).
        - Value range: `0.0` (completely different) to `1.0` (identical structure).
        - **Nearest Training Tanimoto**: Compares the input molecule against all 1,572 molecules in the training set.
        - **Why Train Set compounds show 1.00**: Because that exact molecule was present in training data (self-match).
        - **Why Test Set compounds show < 1.00**: Because they are held-out unseen structures!
        """
    )


# --- Main Title & Header ---
st.title("🧬 Blood-Brain Barrier (BBBP) Screening & Comparison")
st.markdown(
    "Screen molecular candidates for blood-brain barrier permeability using our **4-model consensus ensemble**. "
    "Select compounds by common name from our **2,050 compound catalog** or input custom novel SMILES."
)

# --- Concept Banner: Explaining "Similarity: 1" and Tanimoto ---
with st.expander("ℹ️ Why do some compounds have 'Nearest Training Tanimoto = 1.00'? (Click to expand)", expanded=False):
    st.markdown(
        """
        <div class="explainer-box">
            <h4>🔬 Understanding "Nearest Training Tanimoto Similarity"</h4>
            <p>
            During training, <b>1,572 molecules (80% of the dataset)</b> formed the <b>Train Split</b>, while 
            the remaining compounds were held out in the <b>Test Split</b> (10%) and <b>Validation Split</b> (10%).
            </p>
            <ul>
                <li><b>If you choose a <code>[Train Set]</code> molecule</b> (e.g. <i>Propanolol</i> or <i>Atenolol</i>): 
                The model trained on this exact compound. Its nearest training match is <b>itself</b>, so the Tanimoto similarity is identically <b>1.00 (100% identity)</b>.</li>
                <li><b>If you choose a <code>[Test Set]</code> molecule</b> (e.g. <i>ondansetron</i> or <i>Etoposide</i>) or enter a custom novel SMILES: 
                The molecule was never seen during training, so its similarity will be a decimal value <b>less than 1.00</b> (e.g. <code>0.26</code>, <code>0.65</code>, <code>0.76</code>).</li>
                <li><b>Abstention Policy Threshold (0.35)</b>: If a molecule's similarity is below <code>0.35</code>, the model flags it as an out-of-domain scaffold and abstains with <code>UNCERTAIN</code>.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Session State Management ---
if "slots" not in st.session_state:
    st.session_state.slots = [
        {"id": 1, "selection": PROP_NAME, "use_custom": False, "custom_smiles": ""},
        {"id": 2, "selection": ONDAN_NAME, "use_custom": False, "custom_smiles": ""},
    ]
    st.session_state.next_id = 3
if "screen_triggered" not in st.session_state:
    st.session_state.screen_triggered = True

# --- SCREEN MODULE CONTROLS ---
st.markdown("### 🔬 Screening Engine & Comparison Controls")

t_col1, t_col2, t_col3, t_col4 = st.columns([1.5, 1.2, 1.2, 1.2])

with t_col1:
    # Explicit Screen Module Button
    if st.button("🔬 Screen Selected Molecules", type="primary", use_container_width=True):
        st.session_state.screen_triggered = True
        st.toast("Screening complete!", icon="✅")

with t_col2:
    can_add = len(st.session_state.slots) < 3
    if st.button("➕ Add Compound Slot", disabled=not can_add, use_container_width=True):
        candidates = [ETOP_NAME, ATEN_NAME, "Diazepam (#581) [Train Set]"]
        curr_names = {s["selection"] for s in st.session_state.slots}
        new_pick = next((c for c in candidates if c in NAME_TO_ROW and c not in curr_names), ALL_DISPLAY_NAMES[0])
        st.session_state.slots.append({
            "id": st.session_state.next_id,
            "selection": new_pick,
            "use_custom": False,
            "custom_smiles": "",
        })
        st.session_state.next_id += 1
        st.rerun()

with t_col3:
    if st.button("🧪 Preset: Test Molecules (<1.0 Sim)", use_container_width=True, help="Load unseen test split molecules with varying similarity"):
        st.session_state.slots = [
            {"id": 1, "selection": ONDAN_NAME, "use_custom": False, "custom_smiles": ""},
            {"id": 2, "selection": ETOP_NAME, "use_custom": False, "custom_smiles": ""},
        ]
        st.session_state.next_id = 3
        st.rerun()

with t_col4:
    if st.button("🔄 Reset to Default (Train vs Test)", use_container_width=True):
        st.session_state.slots = [
            {"id": 1, "selection": PROP_NAME, "use_custom": False, "custom_smiles": ""},
            {"id": 2, "selection": ONDAN_NAME, "use_custom": False, "custom_smiles": ""},
        ]
        st.session_state.next_id = 3
        st.rerun()

st.caption(f"Currently comparing **{len(st.session_state.slots)} / 3** compounds side-by-side.")
st.markdown("---")

# --- Side-by-Side Compound Columns ---
num_slots = len(st.session_state.slots)
cols = st.columns(num_slots)
comparison_records = []

for idx, (col, slot) in enumerate(zip(cols, st.session_state.slots)):
    slot_id = slot["id"]
    with col:
        # Card Header with Close button
        header_left, header_right = st.columns([3, 1])
        with header_left:
            st.subheader(f"Molecule #{idx + 1}")
        with header_right:
            can_close = num_slots > 1
            if st.button("✕ Close", key=f"btn_close_{slot_id}", disabled=not can_close, help="Remove this compound"):
                st.session_state.slots = [s for s in st.session_state.slots if s["id"] != slot_id]
                st.rerun()

        # Input Mode Selector
        mode = st.radio(
            "Source",
            options=["Catalog (2,050 compounds)", "Custom SMILES"],
            horizontal=True,
            key=f"mode_{slot_id}",
            index=1 if slot.get("use_custom", False) else 0,
        )
        use_custom = (mode == "Custom SMILES")
        slot["use_custom"] = use_custom

        active_smiles = ""
        compound_name = ""
        known_label_str = "Unknown"
        split_tag = "Novel"

        if not use_custom:
            current_selection = slot.get("selection", ALL_DISPLAY_NAMES[0])
            if current_selection not in ALL_DISPLAY_NAMES:
                current_selection = ALL_DISPLAY_NAMES[0]
            current_index = ALL_DISPLAY_NAMES.index(current_selection)

            selected_name = st.selectbox(
                "Choose Compound:",
                options=ALL_DISPLAY_NAMES,
                index=current_index,
                key=f"select_name_{slot_id}",
                help="Type to search through 2,050 common names",
            )
            slot["selection"] = selected_name
            row_data = NAME_TO_ROW[selected_name]
            active_smiles = str(row_data["smiles"]).strip()
            compound_name = row_data["name"]
            split_tag = row_data["split_tag"]

            raw_p_np = row_data.get("p_np", None)
            if pd.notna(raw_p_np):
                known_label_str = "BBB+ (Penetrates)" if int(raw_p_np) == 1 else "BBB− (Non-penetrating)"
        else:
            compound_name = f"Custom Compound #{idx + 1}"
            active_smiles = st.text_input(
                "Enter SMILES string:",
                value=slot.get("custom_smiles", "CCO"),
                placeholder="e.g. CCO, c1ccccc1",
                key=f"custom_smi_{slot_id}",
            ).strip()
            slot["custom_smiles"] = active_smiles
            split_tag = "Novel / Custom"

        # Split Tag & Info Badges
        split_badge_class = (
            "split-train" if "Train" in split_tag else ("split-test" if "Test" in split_tag else "split-valid")
        )
        st.markdown(
            f'<span class="gt-tag {split_badge_class}">📁 {split_tag}</span>',
            unsafe_allow_html=True,
        )
        if known_label_str != "Unknown":
            st.caption(f"**Known Ground Truth:** `{known_label_str}`")

        # Display SMILES
        st.caption(f"**SMILES:** `{active_smiles}`")

        # 2D Chemical Structure
        mol = Chem.MolFromSmiles(active_smiles) if active_smiles else None
        if mol is not None:
            mol_img = Draw.MolToImage(mol, size=(380, 220))
            st.image(mol_img, use_container_width=True, caption=f"2D Structure: {compound_name}")
        else:
            st.error("Invalid SMILES. Please enter a valid molecular structure.")
            continue

        # Model Prediction
        try:
            prediction = system.predict(active_smiles)
        except Exception as err:
            st.error(f"Prediction failed: {err}")
            continue

        decision = prediction["decision"]
        prob = prediction["positive_class_probability"]
        agreement = prediction["agreement"]
        sim = prediction["nearest_training_tanimoto"]
        in_domain = prediction["in_domain"]
        model_probs = prediction["model_positive_probabilities"]

        # Decision Badge
        badge_class = (
            "decision-plus" if decision == "BBB+" else ("decision-minus" if decision == "BBB-" else "decision-uncertain")
        )
        badge_icon = "🟢" if decision == "BBB+" else ("🔴" if decision == "BBB-" else "🟡")
        badge_text = "BBB+ (Penetrates)" if decision == "BBB+" else ("BBB− (Impermeable)" if decision == "BBB-" else "UNCERTAIN (Abstained)")

        st.markdown(
            f"""
            <div style="text-align: center; margin: 10px 0;">
                <span class="decision-badge {badge_class}">{badge_icon} {badge_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Explain why similarity is 1.00 or <1.00 directly under the metric
        sim_note = ""
        if sim >= 0.999:
            sim_note = "Exact identity match in Training Set (1.00)"
        elif sim < 0.35:
            sim_note = "⚠️ Below 0.35 domain cutoff (Unfamiliar scaffold)"
        else:
            sim_note = f"Unseen test molecule ({sim:.1%} similarity to closest training scaffold)"

        # Primary Metrics
        st.metric(
            label="Ensemble BBB+ Probability",
            value=f"{prob:.1%}",
            delta=f"{(prob - 0.5)*100:+.1f}% vs Neutral (0.50)",
        )

        m1, m2 = st.columns(2)
        m1.metric("Model Agreement", agreement)
        m2.metric("Training Tanimoto", f"{sim:.2f}", help=sim_note)

        st.caption(f"**Similarity Note:** {sim_note}")

        # Ground truth match comparison
        if known_label_str != "Unknown":
            is_gt_positive = "BBB+" in known_label_str
            if decision == "BBB+" and is_gt_positive:
                st.markdown('<span class="gt-tag gt-match">✓ Matches Known BBBP Label (BBB+)</span>', unsafe_allow_html=True)
            elif decision == "BBB-" and not is_gt_positive:
                st.markdown('<span class="gt-tag gt-match">✓ Matches Known BBBP Label (BBB−)</span>', unsafe_allow_html=True)
            elif decision == "UNCERTAIN":
                st.markdown(f'<span class="gt-tag gt-diff">ℹ Abstained (Ground truth is {known_label_str})</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="gt-tag gt-diff">⚠ Differs from Ground Truth ({known_label_str})</span>', unsafe_allow_html=True)

        # Model Breakdown Chart
        st.markdown("##### Model Probability Breakdown")
        df_models = pd.DataFrame({
            "Model": ["GCN (Baseline)", "GAT", "GraphSAGE", "Random Forest"],
            "Probability": [model_probs["gcn"], model_probs["gat"], model_probs["sage"], model_probs["rf"]],
        })
        st.bar_chart(df_models.set_index("Model"), height=180, use_container_width=True)

        # Qualitative Attention Weights Expander
        with st.expander("🔍 GAT Attention Weights", expanded=False):
            try:
                item = graph(active_smiles)
                batch = torch.zeros(item.num_nodes, dtype=torch.long)
                with torch.no_grad():
                    _, attns = system.models["gat"](item.x, item.edge_index, batch, return_attention_weights=True)
                edges, weights = attns[0]
                values = np.zeros(item.num_nodes)
                for dst, weight in zip(edges[1].tolist(), weights.mean(dim=1).tolist()):
                    values[dst] += weight

                fig, ax = plt.subplots(figsize=(5, 2.2))
                ax.bar(range(len(values)), values, color="#3b82f6", edgecolor="none")
                ax.set_xlabel("Atom Index (RDKit order)", fontsize=8)
                ax.set_ylabel("Incoming Attention", fontsize=8)
                ax.tick_params(axis="both", labelsize=7)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Layer-1 incoming attention weights across 4 heads. Qualitative inspection only.")
            except Exception as e:
                st.info(f"Attention inspection unavailable: {e}")

        # Store record for comparison table
        comparison_records.append({
            "Compound": f"{compound_name} [{split_tag}]",
            "Split": split_tag,
            "SMILES": active_smiles,
            "Ground Truth": known_label_str,
            "Decision": decision,
            "Ensemble Prob": f"{prob:.1%}",
            "GCN (Baseline)": f"{model_probs['gcn']:.1%}",
            "GAT": f"{model_probs['gat']:.1%}",
            "GraphSAGE": f"{model_probs['sage']:.1%}",
            "Random Forest": f"{model_probs['rf']:.1%}",
            "Agreement": agreement,
            "Training Tanimoto": f"{sim:.4f}",
        })

# --- Comparison Summary Matrix (if >= 2 compounds) ---
if len(comparison_records) > 1:
    st.markdown("---")
    st.header("📊 Side-by-Side Comparison Matrix")

    df_comp = pd.DataFrame(comparison_records).set_index("Compound").T
    st.dataframe(df_comp, use_container_width=True)

    # Comparative Grouped Bar Chart
    chart_rows = []
    for rec in comparison_records:
        c_name = rec["Compound"]
        chart_rows.extend([
            {"Compound": c_name, "Model": "Ensemble", "BBB+ Probability": float(rec["Ensemble Prob"].replace("%", "")) / 100},
            {"Compound": c_name, "Model": "GCN (Base)", "BBB+ Probability": float(rec["GCN (Baseline)"].replace("%", "")) / 100},
            {"Compound": c_name, "Model": "GAT", "BBB+ Probability": float(rec["GAT"].replace("%", "")) / 100},
            {"Compound": c_name, "Model": "GraphSAGE", "BBB+ Probability": float(rec["GraphSAGE"].replace("%", "")) / 100},
            {"Compound": c_name, "Model": "Random Forest", "BBB+ Probability": float(rec["Random Forest"].replace("%", "")) / 100},
        ])
    df_chart = pd.DataFrame(chart_rows)
    pivot_chart = df_chart.pivot(index="Model", columns="Compound", values="BBB+ Probability")
    st.markdown("##### Cross-Compound Model Probability Comparison")
    st.bar_chart(pivot_chart, height=270, use_container_width=True)

st.markdown("---")
st.caption(
    "MoleculeNet BBBP Track 3 Screening Demo. Leaderboard ROC-AUC evaluates all test molecules. "
    "Probabilities are uncalibrated structural-screening estimates; experimental confirmation is required before clinical use."
)
