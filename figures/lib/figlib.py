from common import *
import matplotlib.patches as mp
def scalebar(ax, length=10, loc=(0.06, 0.04), label=None):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim(); x = x0 + loc[0] * (x1 - x0); y = y0 + loc[1] * (y1 - y0)
    ax.plot([x, x + length], [y, y], color='k', lw=1.5, solid_capstyle='butt', zorder=20)
    ax.text(x + length / 2, y + 0.012 * (y1 - y0), label or f'{length} m', ha='center', va='bottom', fontsize=7, zorder=20)
def north(ax, loc=(0.9, 0.86), size=0.09):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim(); x = x0 + loc[0] * (x1 - x0); y = y0 + loc[1] * (y1 - y0); L = size * (y1 - y0)
    ax.annotate('', xy=(x, y + L), xytext=(x, y), arrowprops=dict(arrowstyle='-|>', color='k', lw=0.8, mutation_scale=7), zorder=20)
    ax.text(x, y + L * 1.08, 'N', ha='center', va='bottom', fontsize=7, zorder=20)
def panel(ax, s, dx=-0.02, dy=1.0):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='right')
def coverage_bg(ax, run='frozen'):
    z = np.load(TAB / f'coverage_{run}.npz'); cnt = z['cnt']; x0, y0, c = float(z['x0']), float(z['y0']), float(z['cell'])
    ext = [x0, x0 + cnt.shape[1] * c, y0, y0 + cnt.shape[0] * c]
    ax.imshow(np.where(cnt > 0, 1.0, np.nan), extent=ext, origin='lower', cmap='Greys', vmin=0, vmax=4, alpha=0.5, zorder=0, rasterized=True, interpolation='nearest')
