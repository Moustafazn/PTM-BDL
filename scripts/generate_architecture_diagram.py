#!/usr/bin/env python3
"""Generate publication-quality PTM-BDL architecture diagram using matplotlib."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results" / "figures" / "architecture_diagram.png"

fig, ax = plt.subplots(1, 1, figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis('off')

C = {
    'esm': '#4361ee', 'gear': '#3a86ff', 'chem': '#8338ec',
    'attn': '#1b4965', 'pool': '#5a7d9a',
    'pho': '#e63946', 'gly': '#2a9d8f', 'delta': '#d4a373',
    'gate': '#bc4749',
    'fus': '#f77f00', 'head': '#264653',
    'bg1': '#dce4f0', 'bg2': '#fcecd4', 'bgf': '#fff3e0',
    'brown': '#795548',
}

def rbox(x, y, w, h, text, color, fs=9, tc='white'):
    ax.add_patch(FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.1",
                 facecolor=color, edgecolor='none', zorder=2))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center',
            fontsize=fs, fontweight='bold', color=tc, zorder=3)

def bg(x, y, w, h, color, txt='', fs=9):
    ax.add_patch(FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.2",
                 facecolor=color, edgecolor='#aaaaaa', lw=0.8, alpha=0.5, zorder=1))
    if txt:
        ax.text(x+w/2, y+h+0.25, txt, fontsize=fs, fontweight='bold',
                color='#222', ha='center', zorder=3)

def arr(x1, y1, x2, y2, c='#888', lw=1.5):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw), zorder=2)

def lbl(x, y, t, s=8, c='#555'):
    ax.text(x, y, t, ha='center', va='center', fontsize=s, color=c, zorder=3)

# ── TITLE ──
ax.text(8, 10.6, 'PTM-BDL: Two-Stage Fusion Architecture', ha='center',
        fontsize=16, fontweight='bold', color='#0d1b2a')

# ═══════════════════════════════════════════════════════
# STAGE 1 — left (title ABOVE the box)
# ═══════════════════════════════════════════════════════
bg(0.3, 4.8, 6.0, 5.0, C['bg1'], 'STAGE 1 — Static Branch (Sequence + Structure + Drug)')

# Encoder boxes (top row)
rbox(0.7, 8.8, 1.6, 0.65, 'ESM-2', C['esm'], 10)
rbox(2.5, 8.8, 1.6, 0.65, 'GearNet', C['gear'], 10)
rbox(4.3, 8.8, 1.6, 0.65, 'ChemBERTa', C['chem'], 10)

# Dimension labels (BELOW encoder boxes)
lbl(1.5, 8.55, '1280-d / residue', 7)
lbl(3.3, 8.55, '512-d / residue', 7)
lbl(5.1, 8.55, '384-d / token', 7)

# Projections (row 3)
rbox(0.8, 7.4, 5.2, 0.6, 'Modality Projections → 512-d Shared Space', '#6c757d', 9.5)

# Joint attention (row 4)
rbox(0.8, 6.3, 5.2, 0.65, '4-Layer Joint Self-Attention', C['attn'], 10)
lbl(3.4, 5.95, '8 heads × 512-d  |  Cross-modal: seq ↔ struct ↔ drug', 7.5)

# Pooling (row 5)
rbox(1.5, 5.3, 3.2, 0.4, 'Attention Pooling', C['pool'], 9)

# S_rep
lbl(3.1, 5.0, 'S_rep (512-d)', 10, C['attn'])

# Arrows
arr(1.5, 8.8, 1.5, 8.05, C['esm'])
arr(3.3, 8.8, 3.3, 8.05, C['gear'])
arr(5.1, 8.8, 5.1, 8.05, C['chem'])
arr(3.4, 7.4, 3.4, 7.0, '#888')
arr(3.4, 6.3, 3.4, 5.75, '#888')
arr(3.1, 5.3, 3.1, 5.15, '#888')

# ═══════════════════════════════════════════════════════
# STAGE 2 — right (title ABOVE the box, wider frame)
# ═══════════════════════════════════════════════════════
bg(6.8, 4.0, 8.8, 5.8, C['bg2'], 'STAGE 2 — PTM Biological Dynamics Layer (PTM-BDL)')

# PTM input boxes (top row)
rbox(7.2, 8.8, 2.0, 0.65, '12 Phospho', C['pho'], 10)
rbox(9.5, 8.8, 2.0, 0.65, '12 Glyco', C['gly'], 10)
rbox(12.0, 8.8, 2.6, 0.65, 'Δ Drug-Induced', C['delta'], 9.5)

# Dimension labels (BELOW PTM boxes)
lbl(8.2, 8.55, 'Y(9) / S(2) / T(1)', 7)
lbl(10.5, 8.55, 'N-linked (×12)', 7)
lbl(13.3, 8.55, '[level, Δ, ratio]', 7)

# Token construction (row 3)
rbox(7.2, 7.4, 7.8, 0.6, '24 Typed Tokens + Type / Protein / Slot Embeddings', C['brown'], 9.5, 'white')

# Type gate (row 4)
rbox(7.5, 6.3, 7.2, 0.6, 'Type-Gated Projection (sigmoid gate × subtype emb)', C['gate'], 9)

# Typed self-attention (row 5)
rbox(7.5, 5.2, 7.2, 0.65, 'Typed Self-Attention (2 layers × 4 heads)', C['pho'], 10)
lbl(11.1, 4.88, 'Cross-type: phospho ↔ glyco attention', 8, '#888')

# Residual + pool (row 6)
rbox(8.5, 4.3, 4.8, 0.4, 'Residual Gate → Mask-Aware Pool', C['brown'], 8.5, 'white')

# P_rep
lbl(10.9, 3.95, 'P_rep (64-d)', 10, C['pho'])

# Arrows
arr(8.2, 8.8, 8.2, 8.05, C['pho'])
arr(10.5, 8.8, 10.5, 8.05, C['gly'])
arr(13.3, 8.8, 13.3, 8.05, C['delta'])
arr(11.1, 7.4, 11.1, 6.95, '#888')
arr(11.1, 6.3, 11.1, 5.9, '#888')
arr(11.1, 5.2, 11.1, 4.75, '#888')
arr(10.9, 4.3, 10.9, 4.1, '#888')

# ═══════════════════════════════════════════════════════
# FUSION (bottom, title ABOVE)
# ═══════════════════════════════════════════════════════
bg(3.5, 0.6, 9.5, 3.0, C['bgf'], 'BILINEAR LATE FUSION')

# S⊙P
rbox(5.5, 2.6, 3.5, 0.7, 'S_rep ⊙ P_rep', C['fus'], 12)
lbl(7.25, 2.25, 'Element-wise bilinear product', 8)

# Heads
rbox(4.2, 1.1, 3.4, 0.65, 'Regression Head → IC50', C['head'], 9.5)
rbox(8.5, 1.1, 4.0, 0.65, 'Classification → Resistance', C['head'], 9.5)

# Stage → Fusion arrows
arr(3.1, 4.85, 6.0, 3.35, C['attn'], 2.5)
arr(10.9, 3.85, 8.5, 3.35, C['pho'], 2.5)

# Fusion → heads
arr(6.2, 2.6, 5.9, 1.8, C['fus'])
arr(8.0, 2.6, 9.0, 1.8, C['fus'])

# Note
lbl(8.25, 0.25, 'Design: Drug enters Stage 1 only → prevents PTM shortcut learning. '
    'PTM-BDL is drug-conditioned through Δ inputs, not drug identity.', 8, '#777')

plt.tight_layout()
plt.savefig(str(OUT), dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved: {OUT}")
