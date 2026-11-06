








from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MATRIX_FILE = ROOT / 'observation_likelihood_local.npy'
OUTPUT = ROOT / 'plotters' / 'confusion_matrix_local'


def main():
    probability = np.load(MATRIX_FILE, allow_pickle=False)
    if (probability.shape != (37, 37) or not np.isfinite(probability).all()
            or np.any(probability < 0)
            or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-12, rtol=0)):
        raise ValueError('37×371')

    with plt.rc_context({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 8.5,
        'mathtext.fontset': 'stix',
        'axes.linewidth': 0.6,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
    }):
        
        fig = plt.figure(figsize=(183 / 25.4, 183 / 25.4 * 5.2 / 5.8),
                         dpi=300, facecolor='white')
        ax = fig.add_axes([0.16, 0.20, 0.65, 0.725])
        color_ax = fig.add_axes([0.845, 0.20, 0.025, 0.725])
        heatmap = ax.pcolormesh(
            np.arange(38), np.arange(38), probability,
            cmap='Blues', vmin=0.0, vmax=1.0,
            edgecolors='white', linewidth=0.45, shading='flat',
            antialiased=True, rasterized=False,
        )
        ax.set_xlim(0, 37)
        ax.set_ylim(37, 0)
        ax.set_aspect('equal')
        positions = [0.5] + [4 + 6 * region for region in range(6)]
        labels = ['H'] + [rf'$\Omega_{{{region}}}$' for region in range(1, 7)]
        ax.set_xticks(positions, labels, fontsize=11)
        ax.set_yticks(positions, labels, fontsize=11)
        ax.tick_params(axis='both', length=0, pad=5)
        for boundary in [1, 7, 13, 19, 25, 31]:
            ax.axvline(boundary, color='#B8C5D1', linewidth=0.6, alpha=0.8)
            ax.axhline(boundary, color='#B8C5D1', linewidth=0.6, alpha=0.8)
        ax.set_xlabel('PIDM-estimated state', fontsize=10, labelpad=13)
        ax.set_ylabel('True state', fontsize=10, labelpad=13)
        ax.set_title('PIDM local-diffusion observation matrix', fontsize=11, pad=12)
        for spine in ax.spines.values():
            spine.set_visible(False)

        colorbar = fig.colorbar(heatmap, cax=color_ax,
                               ticks=[0.0, 0.25, 0.50, 0.75, 1.0])
        colorbar.ax.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.00'])
        colorbar.ax.tick_params(labelsize=8.5, length=0, pad=7)
        colorbar.set_label('Conditional probability', fontsize=10, labelpad=12)
        colorbar.outline.set_linewidth(0.6)
        colorbar.solids.set_rasterized(False)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for text in fig.findobj(Text):
            if text.get_visible() and text.get_text():
                box = text.get_window_extent(renderer)
                if (box.x0 < -1 or box.y0 < -1
                        or box.x1 > fig.bbox.x1 + 1 or box.y1 > fig.bbox.y1 + 1):
                    raise RuntimeError(f'{text.get_text()}')
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT.with_suffix('.svg'))
        fig.savefig(OUTPUT.with_suffix('.pdf'))
        fig.savefig(OUTPUT.with_suffix('.png'), dpi=300)
        plt.close(fig)
    print(f'Input: {MATRIX_FILE}')
    print(f'Saved: {OUTPUT}.svg / .pdf / .png')


if __name__ == '__main__':
    main()
