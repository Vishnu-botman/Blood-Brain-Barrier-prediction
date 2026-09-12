"""Automated 3-Minute Video Recording Script for BBBP Track 3 Hackathon Demo.

Features:
- Launches Microsoft Edge in headless mode via Playwright.
- High-definition recording (1920x1080, 60fps equivalent).
- Injects an on-screen Presenter Teleprompter & Subtitle Banner at the bottom.
- Injects a glowing animated visual cursor so viewers can track clicks and hovers.
- Performs realistic, timed interactions strictly following the 3-minute script:
  * 0:00 - 0:25: Intro & Task Declaration
  * 0:25 - 0:55: Dataset Honesty & Bemis-Murcko Scaffold Split Modal
  * 0:55 - 1:35: Leaderboard Benchmark & Ablation Study Modal (+0.0615 ROC-AUC)
  * 1:35 - 2:05: Live Screening: Propranolol (1.00 Sim) vs Ondansetron (0.65 Sim)
  * 2:05 - 2:30: Impermeability Testing: Amoxicillin & Ampicillin (BBB− Verdict)
  * 2:30 - 2:48: Heuristic Abstention: Etoposide & Nafcillin (UNCERTAIN Band)
  * 2:48 - 3:00: GAT Attention Inspector, Comparison Matrix & Clinical Wrap-up
"""
import os
import shutil
import sys
import time
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from playwright.sync_api import sync_playwright


def setup_overlay(page):
    """Inject glowing visual cursor and bottom presenter teleprompter banner."""
    page.evaluate(
        """
        () => {
            // 1. Teleprompter Banner
            const banner = document.createElement('div');
            banner.id = 'demo-teleprompter';
            banner.style.position = 'fixed';
            banner.style.bottom = '18px';
            banner.style.left = '50%';
            banner.style.transform = 'translateX(-50%)';
            banner.style.width = '88%';
            banner.style.maxWidth = '1150px';
            banner.style.background = 'rgba(10, 15, 29, 0.92)';
            banner.style.backdropFilter = 'blur(16px)';
            banner.style.border = '1px solid rgba(56, 189, 248, 0.4)';
            banner.style.borderRadius = '14px';
            banner.style.padding = '12px 20px';
            banner.style.zIndex = '999999';
            banner.style.boxShadow = '0 12px 40px rgba(0, 0, 0, 0.8), 0 0 24px rgba(56, 189, 248, 0.2)';
            banner.style.display = 'flex';
            banner.style.alignItems = 'center';
            banner.style.justifyContent = 'space-between';
            banner.style.gap = '16px';
            banner.style.fontFamily = "'Plus Jakarta Sans', sans-serif";
            banner.style.transition = 'all 0.3s ease';

            banner.innerHTML = `
                <div style="display: flex; align-items: center; gap: 12px; min-width: 0;">
                    <div id="tp-badge" style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.4); color: #38bdf8; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700; padding: 4px 10px; border-radius: 999px; white-space: nowrap;">
                        0:00 / 2:50
                    </div>
                    <div style="min-width: 0;">
                        <div id="tp-title" style="font-family: 'Outfit', sans-serif; font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-bottom: 2px;">
                            Track 3: BBBP Molecular Screening Demonstration
                        </div>
                        <div id="tp-script" style="font-size: 0.82rem; color: #94a3b8; font-style: italic; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                            "Hello everyone! This is our Hackathon submission for Track 3: Blood-Brain Barrier Molecular Screening..."
                        </div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 8px; flex-shrink: 0;">
                    <span style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Voiceover Cue</span>
                    <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981;"></span>
                </div>
            `;
            document.body.appendChild(banner);

            // 2. Custom Visual Cursor
            const cursor = document.createElement('div');
            cursor.id = 'demo-cursor';
            cursor.style.position = 'fixed';
            cursor.style.width = '24px';
            cursor.style.height = '24px';
            cursor.style.borderRadius = '50%';
            cursor.style.border = '2px solid #38bdf8';
            cursor.style.background = 'rgba(56, 189, 248, 0.25)';
            cursor.style.boxShadow = '0 0 14px #38bdf8';
            cursor.style.pointerEvents = 'none';
            cursor.style.zIndex = '1000000';
            cursor.style.transform = 'translate(-50%, -50%)';
            cursor.style.transition = 'width 0.15s ease, height 0.15s ease, background 0.15s ease';
            document.body.appendChild(cursor);

            window.setCursorPos = (x, y) => {
                cursor.style.left = `${x}px`;
                cursor.style.top = `${y}px`;
            };

            window.clickCursorEffect = () => {
                cursor.style.width = '34px';
                cursor.style.height = '34px';
                cursor.style.background = 'rgba(56, 189, 248, 0.6)';
                setTimeout(() => {
                    cursor.style.width = '24px';
                    cursor.style.height = '24px';
                    cursor.style.background = 'rgba(56, 189, 248, 0.25)';
                }, 200);
            };

            window.updateTeleprompter = (timeStr, title, scriptText) => {
                const b = document.getElementById('tp-badge');
                const t = document.getElementById('tp-title');
                const s = document.getElementById('tp-script');
                if (b) b.textContent = timeStr;
                if (t) t.textContent = title;
                if (s) s.textContent = scriptText;
            };
        }
        """
    )


def update_cue(page, time_str, title, script):
    """Update on-screen teleprompter cue."""
    page.evaluate(
        f"""
        ([timeStr, title, script]) => {{
            if (window.updateTeleprompter) {{
                window.updateTeleprompter(timeStr, title, script);
            }}
        }}
        """,
        [time_str, title, f'"{script}"'],
    )


def move_and_click(page, selector, wait_after=1.0):
    """Move cursor smoothly to element, show click pulse, and click."""
    try:
        box = page.locator(selector).first.bounding_box()
        if box:
            x = box['x'] + box['width'] / 2
            y = box['y'] + box['height'] / 2
            page.evaluate(f"([x, y]) => window.setCursorPos(x, y)", [x, y])
            time.sleep(0.4)
            page.evaluate("() => window.clickCursorEffect()")
            page.locator(selector).first.click()
            time.sleep(wait_after)
    except Exception as e:
        print(f"Interaction notice for {selector}: {e}")
        try:
            page.locator(selector).first.click(timeout=3000)
            time.sleep(wait_after)
        except Exception:
            pass


def record_demo():
    output_dir = Path("artifacts/videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Initializing Playwright browser recording session...")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(
            record_video_dir=str(output_dir),
            record_video_size={"width": 1920, "height": 1080},
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()

        print("🌐 Navigating to http://localhost:3000...")
        page.goto("http://localhost:3000", wait_until="networkidle")
        time.sleep(2)

        setup_overlay(page)
        time.sleep(1)

        # =====================================================================
        # 1. INTRO & TASK DECLARATION (0:00 - 0:25, ~25s)
        # =====================================================================
        print("🎬 Section 1: Intro & Task Declaration...")
        update_cue(
            page,
            "0:00 - 0:25",
            "Track 3: Blood-Brain Barrier Penetration Screening (MoleculeNet BBBP)",
            "Hello everyone! This is our Hackathon submission for Track 3: Blood-Brain Barrier Molecular Screening...",
        )
        page.evaluate("([x, y]) => window.setCursorPos(x, y)", [320, 50])
        time.sleep(4)

        # Hover over ML Backend indicator
        page.evaluate("([x, y]) => window.setCursorPos(x, y)", [560, 68])
        time.sleep(3)

        # Smooth scroll down slightly to show comparator slots
        page.mouse.wheel(0, 150)
        time.sleep(4)
        page.mouse.wheel(0, -150)
        time.sleep(6)

        # =====================================================================
        # 2. DATASET INTEGRITY & SCAFFOLD SPLITTING (0:25 - 0:55, ~30s)
        # =====================================================================
        print("🎬 Section 2: Dataset Integrity & Bemis-Murcko Split Modal...")
        update_cue(
            page,
            "0:25 - 0:55",
            "Dataset Integrity: Fixed 80/10/10 Bemis–Murcko Scaffold Split (Seed 42)",
            "Addressing data integrity: the source archive contained raw data but no official organizer split...",
        )

        # Click Methodology & Tanimoto button
        move_and_click(page, "button:has-text('Methodology & Tanimoto')", wait_after=1.5)
        time.sleep(5)

        # Hover over callout box
        page.evaluate("([x, y]) => window.setCursorPos(x, y)", [960, 420])
        time.sleep(5)

        # Scroll within modal to show abstention rules
        page.evaluate("document.querySelector('.modal-body').scrollBy({top: 180, behavior: 'smooth'})")
        time.sleep(6)

        # Close modal
        move_and_click(page, "button:has-text('Understood')", wait_after=1.5)
        time.sleep(3)

        # =====================================================================
        # 3. LEADERBOARD & ABLATION STUDY (0:55 - 1:35, ~40s)
        # =====================================================================
        print("🎬 Section 3: Leaderboard Benchmark & Ablation Study...")
        update_cue(
            page,
            "0:55 - 1:35",
            "Leaderboard Benchmark: Proposed Ensemble (+0.0615 ROC-AUC over Baseline)",
            "For our modeling pipeline, we evaluated four distinct architectures: 2-layer GCN, GAT, GraphSAGE, and Random Forest...",
        )

        # Click Leaderboard & Ablation button
        move_and_click(page, "button:has-text('Leaderboard & Ablation')", wait_after=1.5)
        time.sleep(5)

        # Hover over the win banner
        page.evaluate("([x, y]) => window.setCursorPos(x, y)", [960, 290])
        time.sleep(6)

        # Hover down over the table rows: GCN vs Proposed Ensemble
        page.evaluate("([x, y]) => window.setCursorPos(x, y)", [960, 420])
        time.sleep(6)
        page.evaluate("([x, y]) => window.setCursorPos(x, y)", [960, 560])
        time.sleep(6)

        # Close modal
        move_and_click(page, ".modal-card .card-close-btn", wait_after=1.5)
        time.sleep(3)

        # =====================================================================
        # 4. LIVE SCREENING: PROPRANOLOL VS ONDANSETRON (1:35 - 2:05, ~30s)
        # =====================================================================
        print("🎬 Section 4: Live Screening (Train vs Test)...")
        update_cue(
            page,
            "1:35 - 2:05",
            "Live Screening: Propranolol (1.00 Identity) vs Ondansetron (0.65 Proximity)",
            "Now let's demonstrate the screening interface in action: comparing Propranolol and Ondansetron side-by-side...",
        )

        # Click Train vs Test Split preset chip
        move_and_click(page, "button:has-text('Train vs Test Split')", wait_after=2.0)
        time.sleep(4)

        # Hover over Propranolol 2D molecule render
        page.evaluate("([x, y]) => window.setCursorPos(x, y)", [600, 480])
        time.sleep(4)

        # Hover over 100% Training Identity metric
        page.evaluate("([x, y]) => window.setCursorPos(x, y)", [720, 680])
        time.sleep(4)

        # Hover over Ondansetron 2D render & 65% Proximity
        page.evaluate("([x, y]) => window.setCursorPos(x, y)", [1300, 480])
        time.sleep(4)
        page.evaluate("([x, y]) => window.setCursorPos(x, y)", [1420, 680])
        time.sleep(5)

        # =====================================================================
        # 5. IMPERMEABILITY TESTING: AMOXICILLIN & AMPICILLIN (2:05 - 2:30, ~25s)
        # =====================================================================
        print("🎬 Section 5: Impermeability Testing (BBB− Verdicts)...")
        update_cue(
            page,
            "2:05 - 2:30",
            "Impermeability Testing: Amoxicillin & Ampicillin (BBB− Verdict, 21.5% Prob)",
            "Next, testing impermeability: on Amoxicillin and Ampicillin, the ensemble predicts low probabilities (21.5% and 24.6%)...",
        )

        # Click Impermeable (BBB−) preset chip
        move_and_click(page, "button:has-text('Impermeable (BBB−)')", wait_after=2.0)
        time.sleep(4)

        # Hover over Red Decision Badges
        page.evaluate("([x, y]) => window.setCursorPos(x, y)", [600, 580])
        time.sleep(4)
        page.evaluate("([x, y]) => window.setCursorPos(x, y)", [1300, 580])
        time.sleep(4)

        # Hover over the breakdown progress bars
        page.evaluate("([x, y]) => window.setCursorPos(x, y)", [600, 810])
        time.sleep(5)

        # =====================================================================
        # 6. HEURISTIC ABSTENTION: ETOPOSIDE & NAFCILLIN (2:30 - 2:48, ~18s)
        # =====================================================================
        print("🎬 Section 6: Heuristic Abstention (UNCERTAIN Band)...")
        update_cue(
            page,
            "2:30 - 2:48",
            "Heuristic Abstention: Etoposide & Nafcillin (UNCERTAIN Band 0.35 - 0.65)",
            "Finally, observe Etoposide: its mean probability is 45.2%. Because it falls in our 0.35 to 0.65 ambiguity band...",
        )

        # Click Borderline (UNCERTAIN) preset chip
        move_and_click(page, "button:has-text('Borderline (UNCERTAIN)')", wait_after=2.0)
        time.sleep(4)

        # Hover over Yellow UNCERTAIN badge
        page.evaluate("([x, y]) => window.setCursorPos(x, y)", [600, 580])
        time.sleep(6)

        # =====================================================================
        # 7. GAT ATTENTION INSPECTION & WRAP-UP (2:48 - 3:00, ~12s)
        # =====================================================================
        print("🎬 Section 7: GAT Attention Inspector, Matrix & Wrap-up...")
        update_cue(
            page,
            "2:48 - 3:00",
            "GAT Layer-1 Atom Attention Inspector & Clinical Screening Notice",
            "GAT attention weights provide qualitative inspection, not causal chemical proof. Wet-lab assays are essential. Thank you!",
        )

        # Expand GAT attention inspector
        move_and_click(page, ".attention-summary", wait_after=1.5)
        time.sleep(4)

        # Scroll down to comparison matrix
        page.evaluate("window.scrollBy({top: 480, behavior: 'smooth'})")
        time.sleep(5)

        print("✅ Demo sequence complete! Finalizing video stream...")
        time.sleep(2)

        # Close context to write out video
        context.close()
        browser.close()

    # Find the newly recorded video file
    video_files = list(output_dir.glob("*.webm"))
    if video_files:
        # Sort by creation time to get the latest
        latest_video = max(video_files, key=lambda f: f.stat().st_mtime)
        final_path = output_dir / "bbbp_hackathon_demo_3min.webm"
        shutil.copyfile(latest_video, final_path)
        print(f"🎉 Recording saved successfully at: {final_path.resolve()}")
        print(f"📊 Video file size: {final_path.stat().st_size / (1024*1024):.2f} MB")
        return final_path

    return None


if __name__ == "__main__":
    record_demo()
