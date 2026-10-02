"""Draft result figures for the SEGAN revision (Elsevier single/double column, vector PDF + PNG preview)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REPO = Path(__file__).resolve().parents[1]
REL = REPO / 'src'
DATA = REPO / 'data'
OUT = REPO / 'results' / 'figure_data'
FIG = REPO / 'figures' / 'output'
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REL / 'code'))
from forecast_closed_loop import thermal_step  # noqa: E402

# Reference categorical order (slots 1-4), fixed per controller everywhere.
C = {'Scenario': '#2a78d6', 'Deterministic': '#eb6834', 'Common-future': '#1baf7a', 'Residual-box': '#eda100'}
MK = {'Scenario': 'o', 'Deterministic': 's', 'Common-future': '^', 'Residual-box': 'D'}
INK, INK2, GRID, BAND = '#0b0b0b', '#52514e', '#e4e3df', '#ebeae6'
SINGLE, DOUBLE = 2.7, 5.40  # inches; DOUBLE = elsarticle preprint 	extwidth (390 pt), figures are placed at 1:1
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .5, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.2,
    'legend.frameon': False, 'savefig.dpi': 300, 'pdf.fonttype': 42})

CFG = json.loads((REL / 'configuration_without_identifiers.json').read_text())
TH, DT = CFG['thermal'], CFG['dt_hours']
REF = json.loads((REPO / 'results' / 'reference_results.json').read_text())
with np.load(DATA / 'study_inputs.npz') as d:
    INP = {k: d[k] for k in d.files}
AMB = INP['sydney_utc10_end_ambient']
NAME = {'deterministic': 'Deterministic', 'scenario': 'Scenario', 'mean_open_loop': 'Common-future', 'component_residual': 'Residual-box'}


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('saved', name)


def free_float(n, start, steps, warmup, scale=None):
    """Zero-cooling trajectory. Cooling only lowers temperature, so its cold degree-hours are a floor for any controller."""
    th = dict(TH)
    if scale:
        th[scale[0]] = (np.array(th[scale[0]]) * scale[1]).tolist()
    a = lambda k: np.array(th[k][:n])
    T, cold, traj = np.full(n, th['initial_C']), 0., []
    for i, t in enumerate(range(start, start + steps)):
        T = thermal_step(T, float(AMB[t]), np.zeros(n), a('R_C_per_kw'), a('C_kwh_per_C'), a('COP'), a('internal_gain_kw_thermal'), DT)
        traj.append(T)
        if i >= warmup:
            cold += DT * np.maximum(th['comfort_low_C'] - T, 0).sum()
    return cold, np.array(traj)


# ---------------------------------------------------------------- Fig. A: closed-loop trajectories
def fig_trajectories():
    sc, de = OUT / 'sc50.npz', OUT / 'det50.npz'
    if not (sc.exists() and de.exists()):
        print('skip trajectories (missing runs)'); return
    s, d = np.load(sc), np.load(de)
    t = np.array(s['time'], dtype='datetime64[m]').astype('O')
    _, ff = free_float(50, 3648, 672, 48)
    # Representative window: the first three evaluation days (17-19 Jan, daily maxima 31.6/34.0/27.3 °C).
    w = slice(48, 48 + 3 * 48)
    fig, ax = plt.subplots(3, 1, figsize=(DOUBLE, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[1.25, 1, .8], hspace=.38))
    a0 = ax[0]
    a0.axhspan(TH['comfort_low_C'], TH['comfort_high_C'], color=BAND, lw=0, zorder=0)
    a0.plot(t[w], s['ambient'][w], color=INK2, lw=.9, ls='--', label='Ambient')
    a0.plot(t[w], ff[w].mean(1), color='#9a9993', lw=1.0, ls=':', label='Free-floating (no cooling)')
    a0.plot(t[w], d['T'][w].mean(1), color=C['Deterministic'], label='Deterministic MPC')
    a0.plot(t[w], s['T'][w].mean(1), color=C['Scenario'], label='Scenario MPC')
    a0.fill_between(t[w], s['T'][w].min(1), s['T'][w].max(1), color=C['Scenario'], alpha=.15, lw=0, label='Scenario MPC, min–max over homes')
    a0.set_ylabel('Temperature (°C)')
    from matplotlib.patches import Patch
    h, l = a0.get_legend_handles_labels()
    a0.legend(h + [Patch(color=BAND)], l + ['Soft comfort band 22–26 °C'], ncol=3, loc='lower left', bbox_to_anchor=(0, 1.0))
    a0.set_ylim(14, 36)
    a1 = ax[1]
    agg = lambda x: x[w].sum(1)
    a1.plot(t[w], agg(s['base']), color='#9a9993', lw=.9, label='Base load')
    a1.plot(t[w], agg(s['pv']), color=C['Residual-box'], lw=1.0, label='PV')
    a1.plot(t[w], agg(s['cool']), color=C['Scenario'], label='HVAC (scenario)')
    a1.plot(t[w], agg(d['cool']), color=C['Deterministic'], lw=1.0, label='HVAC (deterministic)')
    a1.plot(t[w], agg(s['net']), color=INK, lw=.8, ls='--', label='Net grid (scenario)')
    a1.axhline(0, color=INK2, lw=.5)
    a1.set_ylabel('Community power (kW)')
    a1.legend(ncol=3, loc='lower left', bbox_to_anchor=(0, 1.0))
    a2 = ax[2]
    vol = np.maximum(s['trade'][w], 0).sum(1)
    a2.bar(t[w], vol, width=1 / 48 * .8, color=C['Scenario'], lw=0)
    a2.set_ylabel('P2P volume (kW)')
    for k in ax:  # peak tariff shading
        for day in np.unique(np.array(s['time'][w], dtype='datetime64[D]')):
            day = day.astype('O')
            k.axvspan(np.datetime64(day) + np.timedelta64(16, 'h'), np.datetime64(day) + np.timedelta64(21, 'h'), color='#f6e7d9', lw=0, zorder=-1)
    a2.text(1, 1.02, 'shaded: peak tariff 16:00–21:00', transform=a2.transAxes, ha='right', va='bottom', fontsize=8, color=INK2)
    a2.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    a2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    save(fig, 'figA_closed_loop_trajectories')


# ---------------------------------------------------------------- Fig. B: comfort decomposition
def comfort_rows(homes, seed=202):
    floor, _ = free_float(homes, 3648, 672, 48)
    rows = []
    for b in REF['central_baselines']:
        if b['homes'] == homes and b['seed'] == seed:
            s = b['summary']
            rows.append(dict(name=NAME[b['baseline']], bill=s['monetary_bill'], floor=floor,
                             avoid=s['cold_degree_hours'] - floor, hot=s['hot_degree_hours']))
    order = ['Deterministic', 'Scenario', 'Common-future', 'Residual-box']
    return sorted(rows, key=lambda r: order.index(r['name'])), floor


def fig_comfort():
    fig, axs = plt.subplots(2, 1, figsize=(DOUBLE, 4.3), gridspec_kw=dict(hspace=.55))
    for ax, homes in zip(axs, (50, 100)):
        rows, floor = comfort_rows(homes)
        y = np.arange(len(rows))[::-1]
        av = np.array([r['avoid'] for r in rows]); hot = np.array([r['hot'] for r in rows])
        ax.barh(y, av, height=.55, color=[C[r['name']] for r in rows], lw=0)
        ax.barh(y, hot, left=av + max(av) * .004, height=.55, color='none', edgecolor=[C[r['name']] for r in rows], hatch='////', lw=.8)
        for yi, r in zip(y, rows):
            ax.text(r['avoid'] + r['hot'] + max(av) * .02, yi, f"{r['avoid']:.0f} + {r['hot']:.0f}  |  {r['bill']:.0f} AUD", va='center', fontsize=8, color=INK)
        ax.set_yticks(y, [r['name'] for r in rows])
        ax.set_xlim(0, max(av + hot) * 1.6)
        ax.set_xlabel('Avoidable discomfort (°C·h)')
        ax.set_title(f'{homes} homes (free-floating cold floor of {floor:,.0f} °C·h not shown)', fontsize=8.5, loc='left')
        ax.grid(axis='y', visible=False)
    from matplotlib.patches import Patch
    axs[0].legend(handles=[Patch(facecolor='#c9c8c3', label='solid: cold violation above floor'),
                           Patch(facecolor='none', edgecolor=INK2, hatch='////', label='hatched: hot violation')],
                  loc='lower right', bbox_to_anchor=(1, 1.1), ncol=2)
    save(fig, 'figB_comfort_decomposition')


# ---------------------------------------------------------------- Fig. C: cost–comfort tradeoff incl. seeds
def fig_tradeoff():
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.7), gridspec_kw=dict(wspace=.25))
    for ax, homes in zip(axs, (50, 100)):
        floor, _ = free_float(homes, 3648, 672, 48)
        det = [b['summary']['monetary_bill'] for b in REF['central_baselines'] if b['homes'] == homes and b['baseline'] == 'deterministic'][0]
        top = det * 1.08
        lo = min(b['summary']['monetary_bill'] for b in REF['central_baselines'] if b['homes'] == homes)
        ax.set_ylim(lo * .97, top)
        for b in REF['central_baselines']:
            if b['homes'] != homes:
                continue
            s, nm = b['summary'], NAME[b['baseline']]
            x = s['cold_degree_hours'] - floor + s['hot_degree_hours']
            yv = s['monetary_bill']
            if nm == 'Residual-box':
                ax.annotate(f'Residual-box: {yv:,.0f} AUD (off scale)', xy=(x, top), xytext=(x, top*.985), ha='right', va='top', fontsize=8,
                            color=INK, arrowprops=dict(arrowstyle='-|>', color=C[nm], lw=1))
                ax.scatter(x, top, s=30, marker=MK[nm], color=C[nm], edgecolor='white', lw=1.2, zorder=3, clip_on=False, label=nm)
                continue
            ax.scatter(x, yv, s=30, marker=MK[nm], color=C[nm], edgecolor='white', lw=1.2, zorder=3,
                       label=nm if (nm != 'Scenario' or b['seed'] == 202) else None)
            if nm == 'Scenario':
                off = {101: (6, -7), 202: (6, 2), 303: (6, 3)}[b['seed']]
                ax.annotate(f"seed {b['seed']}", (x, yv), xytext=off, textcoords='offset points', fontsize=7.5, color=INK2, ha='left' if off[0] > 0 else 'right')
        ax.set_xlabel('Avoidable discomfort (°C·h)')
        ax.set_ylabel('Evaluation-period bill (AUD)')
        ax.set_title(f'{homes} homes', loc='left')
    axs[0].legend(loc='upper left', bbox_to_anchor=(0, 1.28), ncol=4)
    save(fig, 'figC_cost_comfort_tradeoff')


# ---------------------------------------------------------------- Fig. D: PJ-ADMM convergence + timing
def fig_convergence():
    p = OUT / 'convergence.json'
    if not p.exists():
        print('skip convergence'); return
    cv = json.loads(p.read_text())
    fig, grid = plt.subplots(2, 2, figsize=(DOUBLE, 4.6), gridspec_kw=dict(wspace=.32, hspace=.55))
    axs = [grid[0, 0], grid[0, 1], grid[1, 0]]
    leg_ax = grid[1, 1]; leg_ax.axis('off')
    col = {'pj_10': '#2a78d6', 'pj_50': '#1baf7a', 'pj_100': '#4a3aa7', 'gs_10': '#52514e'}
    lab = {'pj_10': 'PJ, 10 homes', 'pj_50': 'PJ, 50 homes', 'pj_100': 'PJ, 100 homes', 'gs_10': 'GS, 10 homes'}
    for k in ('pj_10', 'pj_50', 'pj_100', 'gs_10'):
        if k not in cv or not cv[k]['trace']:
            continue
        tr = cv[k]['trace']
        it = np.array([r['iteration'] for r in tr])
        if k.startswith('gs'):
            prim = np.array([max(r['primal'] / r['tolerance'], r['max_clearing'] / 1e-3) for r in tr])
            dual = np.array([r['stationarity'] / r['tolerance'] for r in tr])
        else:
            prim = np.array([max(r['primal'] / r['ep'], r['raw_max_kw'] / 1e-3) for r in tr])
            dual = np.array([max(r['dual'], r['proximal']) / r['ed'] for r in tr])
        ls = '--' if k.startswith('gs') else '-'
        axs[0].semilogy(it, prim, color=col[k], ls=ls, lw=1, label=lab[k])
        axs[1].semilogy(it, dual, color=col[k], ls=ls, lw=1, label=lab[k])
    for a, ttl in zip(axs[:2], ('Primal / clearing', 'Dual / stationarity')):
        a.axhline(1, color=INK, lw=.7)
        a.text(1.1, 1.15, 'tolerance', ha='left', va='bottom', fontsize=8, color=INK2)
        a.set_xlabel('Iteration'); a.set_ylabel('Residual / tolerance'); a.set_title(ttl, loc='left')
    h, l = axs[0].get_legend_handles_labels()
    for a in axs[:2]:
        a.set_xscale('log')
    # timing panel: aggregates from the reference results (no per-trial values are archived)
    ax = axs[2]
    pj = [g for g in REF['timing'] if g['method'] == 'pj']
    n = np.array([g['n'] for g in pj]); m = np.array([g['algorithm_seconds']['mean'] for g in pj])
    mx = np.array([g['cold_control_latency_seconds']['maximum'] for g in pj])
    ax.plot(n, m, '-o', color=C['Scenario'], ms=4, label='Mean algorithm time')
    ax.plot(n, mx, ':s', color=INK2, ms=3.5, label='Max cold-start latency')
    ax.axhline(1800, color=INK, lw=.7); ax.text(10, 1800 * 1.12, '1800-s control interval', fontsize=8, color=INK2)
    ax.set_yscale('log'); ax.set_ylim(.5, 5000); ax.set_xticks(n)
    ax.set_xlabel('Homes'); ax.set_ylabel('Seconds'); ax.set_title('PJ-ADMM time per step', loc='left')
    h2, l2 = ax.get_legend_handles_labels()
    leg_ax.legend(h + h2, l + l2, loc='center left', fontsize=8)
    save(fig, 'figD_convergence_timing')


# ---------------------------------------------------------------- Fig. E: configuration sensitivity
def fig_sensitivity():
    names = {'R_C_per_kw': 'Thermal resistance', 'C_kwh_per_C': 'Thermal capacitance', 'COP': 'COP',
             'cooling_max_kw_electric': 'Cooling capacity', 'pv': 'PV capacity'}
    by = {}
    for r in REF['sensitivity']:
        floor, _ = free_float(10, 3552, 144, 48, (r['parameter'], r['factor']) if r['parameter'] in ('R_C_per_kw', 'C_kwh_per_C') else None)
        s = r['summary']
        by.setdefault(r['parameter'], {})[r['factor']] = (s['monetary_bill'], s['cold_degree_hours'] - floor + s['hot_degree_hours'])
    order = ['pv', 'COP', 'R_C_per_kw', 'C_kwh_per_C', 'cooling_max_kw_electric']
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.2), sharey=True, gridspec_kw=dict(wspace=.08))
    y = np.arange(len(order))[::-1]
    for j, (ax, ttl) in enumerate(zip(axs, ('Bill (AUD)', 'Avoidable discomfort (°C·h)'))):
        for yi, k in zip(y, order):
            lo, hi = sorted(by[k])
            ax.plot([by[k][lo][j], by[k][hi][j]], [yi, yi], color=GRID, lw=3, solid_capstyle='round', zorder=1)
            ax.scatter(by[k][lo][j], yi, s=26, color='#9fc3ee', edgecolor='#2a78d6', lw=.8, zorder=3, label='low factor' if yi == y[0] else None)
            ax.scatter(by[k][hi][j], yi, s=26, color='#1c4f8f', edgecolor='white', lw=1, zorder=3, label='high factor' if yi == y[0] else None)
        ax.set_xlabel(ttl); ax.grid(axis='y', visible=False)
    axs[0].set_yticks(y, [f'{names[k]} (×{sorted(by[k])[0]:g} / ×{sorted(by[k])[1]:g})' for k in order])
    axs[1].legend(loc='upper right')
    save(fig, 'figE_sensitivity')


# ---------------------------------------------------------------- Fig. F: network extension
def fig_network():
    runs = [(OUT / 'net05.npz', 'background ×0.5', '#7fb0ea'), (OUT / 'net10.npz', 'background ×1.0', '#1c4f8f')]
    if not all(p.exists() for p, *_ in runs):
        print('skip network'); return
    fig, axs = plt.subplots(2, 1, figsize=(DOUBLE, 3.6), sharex=True, gridspec_kw=dict(hspace=.12))
    for p, lab, c in runs:
        d = np.load(p); w = slice(int(d['warmup']), None)
        t = np.array(d['time'], dtype='datetime64[m]').astype('O')[w]
        net = d['network'][w]
        axs[0].plot(t, net[:, 0], color=c, label=lab)
        axs[1].plot(t, 100 * net[:, 2], color=c, label=lab)
        over = net[:, 2] > 1 + 1e-8
        if over.any():
            axs[1].scatter(np.array(t)[over], 100 * net[over, 2], s=22, color='#e34948', edgecolor='white', lw=.8, zorder=4, label=f'line overload ({over.sum()} steps)')
    axs[0].axhline(.90, color=INK, lw=.7); axs[0].text(t[0], .902, 'lower voltage limit 0.90 p.u.', fontsize=8, color=INK2, va='bottom')
    axs[0].set_ylabel('Minimum voltage (p.u.)')
    axs[1].axhline(100, color=INK, lw=.7)
    axs[1].set_ylabel('Max line loading (%)')
    axs[0].legend(loc='upper left', bbox_to_anchor=(0, 1.2), ncol=2); axs[1].legend(loc='center left', bbox_to_anchor=(.01, .52), ncol=3)
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    save(fig, 'figF_network_extension')


if __name__ == '__main__':
    for f in (fig_trajectories, fig_comfort, fig_tradeoff, fig_convergence, fig_sensitivity, fig_network):
        f()
