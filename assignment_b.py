# %%

from geometry import create_geometry, BOUNDARY_CONDITION_CONVECTION, BOUNDARY_CONDITION_QOUT
import calfem.core as cfc
import calfem.vis_mpl as cfv
import numpy as np
import os
import matplotlib
import matplotlib.tri as mtri
from datetime import datetime
matplotlib.use("qtagg")

# == Parameters for assignment b) ==
rho = 1500.0         # kg/m^3 density
cp = 800.0           # J/(kg K) specific heat
k = 2.0              # W/(m K) thermal conductivity
Q_max = 4e5          # W/m^3 max body heat
q_out_max = 2000.0   # W/m^2 max outgoing heat flux
T_0 = 293.0          # K initial temperature
T_inf = 293.0        # K ambient temperature
ac = 50.0            # W/(m^2 K) convection coefficient
t = 1.0              # m thickness
ttot = 10.0 * 60.0   # s total simulated time (10 min)
n_steps = 300
SHOW_PLOTS = False

geo, coords, edof, dofs, bdofs, elementmarkers = create_geometry()

nDofs = np.size(dofs)
K = np.zeros((nDofs, nDofs))
C = np.zeros((nDofs, nDofs))

D = np.array([[k, 0.0], [0.0, k]])

# Build conductivity K and capacity C.
for el_nodes in edof:
    idx = el_nodes - 1
    ex_el = coords[idx, 0]
    ey_el = coords[idx, 1]

    flw_res = cfc.flw2te(ex_el, ey_el, [t], D, 0.0)
    Ke = flw_res[0] if isinstance(flw_res, tuple) else flw_res

    x1, x2, x3 = ex_el
    y1, y2, y3 = ey_el
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    Ce = (rho * cp * t * area / 12.0) * np.array(
        [[2.0, 1.0, 1.0],
         [1.0, 2.0, 1.0],
         [1.0, 1.0, 2.0]]
    )

    cfc.assem(el_nodes, K, Ke)
    cfc.assem(el_nodes, C, Ce)

# Boundary sets.
conv_nodes = set(bdofs[BOUNDARY_CONDITION_CONVECTION])
qout_nodes = set(bdofs[BOUNDARY_CONDITION_QOUT])

conv_edges = []
qout_edges = []
for el_nodes in edof:
    nodes = list(el_nodes)
    for n1, n2 in [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])]:
        edge = tuple(sorted((int(n1), int(n2))))
        if n1 in conv_nodes and n2 in conv_nodes:
            conv_edges.append(edge)
        if n1 in qout_nodes and n2 in qout_nodes:
            qout_edges.append(edge)

conv_edges = sorted(set(conv_edges))
qout_edges = sorted(set(qout_edges))

# Convection stiffness part is constant.
for n1, n2 in conv_edges:
    c1 = coords[n1 - 1]
    c2 = coords[n2 - 1]
    L = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
    Ke_conv = (ac * L * t / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
    cfc.assem(np.array([n1, n2]), K, Ke_conv)

# Precompute unit load vectors.
f_Q_unit = np.zeros((nDofs, 1))
for el_nodes in edof:
    idx = el_nodes - 1
    ex_el = coords[idx, 0]
    ey_el = coords[idx, 1]
    flw_res = cfc.flw2te(ex_el, ey_el, [t], D, 1.0)
    fe_Q_el = flw_res[1] if isinstance(
        flw_res, tuple) else np.zeros((len(el_nodes), 1))
    idx_vec = (el_nodes - 1).astype(int)
    f_Q_unit[idx_vec, 0] += np.asarray(fe_Q_el).reshape(-1)

f_conv_const = np.zeros((nDofs, 1))
for n1, n2 in conv_edges:
    c1 = coords[n1 - 1]
    c2 = coords[n2 - 1]
    L = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
    fe_conv = (ac * T_inf * L * t / 2.0) * np.array([1.0, 1.0])
    edge_idx = np.array([n1 - 1, n2 - 1], dtype=int)
    f_conv_const[edge_idx, 0] += fe_conv

f_qout_unit = np.zeros((nDofs, 1))
for n1, n2 in qout_edges:
    c1 = coords[n1 - 1]
    c2 = coords[n2 - 1]
    L = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
    fe_qout_unit = (-L * t / 2.0) * np.array([1.0, 1.0])
    edge_idx = np.array([n1 - 1, n2 - 1], dtype=int)
    f_qout_unit[edge_idx, 0] += fe_qout_unit

# Implicit Euler integration.
dt = ttot / n_steps
A_sys = C / dt + K
T = np.full((nDofs, 1), T_0)

time = np.linspace(0.0, ttot, n_steps + 1)
Tmax = np.zeros(n_steps + 1)
Tmax[0] = float(np.max(T))

snapshot_idx = np.linspace(0, n_steps, 6, dtype=int)
snapshots = {0: T.copy()}

output_dir = "assignment_b_plots"
os.makedirs(output_dir, exist_ok=True)
run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

for i in range(1, n_steps + 1):
    ti = time[i]
    scale = np.sin(np.pi * ti / ttot)

    # Q(t) = Qmax*sin(pi*t/ttot), q_out(t) = qmax*sin(pi*t/ttot)
    f_t = scale * Q_max * f_Q_unit + f_conv_const + scale * q_out_max * f_qout_unit

    rhs = f_t + (C / dt) @ T
    T = np.linalg.solve(A_sys, rhs)
    Tmax[i] = float(np.max(T))

    if i in snapshot_idx:
        snapshots[i] = T.copy()

# One-window dashboard:
# top row = Tmax(t), bottom 2 rows = 6 temperature snapshots.
tri = mtri.Triangulation(
    coords[:, 0], coords[:, 1], triangles=(edof - 1).astype(int))
all_snapshot_values = np.hstack(
    [snapshots[idx].reshape(-1, 1) for idx in snapshot_idx])
vmin = float(np.min(all_snapshot_values))
vmax = float(np.max(all_snapshot_values))
shared_levels = np.linspace(vmin, vmax, 37)

fig = cfv.plt.figure(figsize=(14, 11), constrained_layout=True)
gs = fig.add_gridspec(3, 3)

# Top overview curve
ax_top = fig.add_subplot(gs[0, :])
ax_top.plot(time / 60.0, Tmax, linewidth=2)
ax_top.set_xlabel("Time [min]")
ax_top.set_ylabel("Maximum temperature [K]")
ax_top.set_title("Maximum temperature over time")
ax_top.grid(True)

# Six snapshot panels
axs = [
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[1, 2]),
    fig.add_subplot(gs[2, 0]),
    fig.add_subplot(gs[2, 1]),
    fig.add_subplot(gs[2, 2]),
]

last_plot = None
for ax, idx in zip(axs, snapshot_idx):
    values = snapshots[idx].reshape(-1)
    last_plot = ax.tricontourf(
        tri, values, levels=shared_levels)
    ax.set_title(f"t = {time[idx] / 60.0:.2f} min")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

tick_step = max(1, len(shared_levels) // 12)
fig.colorbar(
    last_plot,
    ax=axs,
    boundaries=shared_levels,
    ticks=shared_levels[::tick_step],
    drawedges=True,
    spacing="proportional",
    label="Temperature [K]"
)
fig.suptitle("Assignment b): Transient temperature overview", fontsize=15)

# Keep separate image exports too.
fig_curve = cfv.plt.figure(figsize=(8, 5))
ax_curve = fig_curve.add_subplot(111)
ax_curve.plot(time / 60.0, Tmax, linewidth=2)
ax_curve.set_xlabel("Time [min]")
ax_curve.set_ylabel("Maximum temperature [K]")
ax_curve.set_title("Maximum temperature over time")
ax_curve.grid(True)
fig_curve.savefig(os.path.join(
    output_dir, "max_temperature_vs_time.png"), dpi=160, bbox_inches="tight")
cfv.plt.close(fig_curve)

fig_overview = cfv.plt.figure(figsize=(14, 8), constrained_layout=True)
axs_overview = fig_overview.subplots(2, 3)
last_plot_overview = None
for ax, idx in zip(axs_overview.ravel(), snapshot_idx):
    values = snapshots[idx].reshape(-1)
    last_plot_overview = ax.tricontourf(
        tri, values, levels=shared_levels)
    ax.set_title(f"t = {time[idx] / 60.0:.2f} min")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
fig_overview.colorbar(
    last_plot_overview,
    ax=axs_overview.ravel().tolist(),
    boundaries=shared_levels,
    ticks=shared_levels[::tick_step],
    drawedges=True,
    spacing="proportional",
    label="Temperature [K]"
)
fig.savefig(
    os.path.join(output_dir, "dashboard_one_window.png"),
    dpi=160,
    bbox_inches="tight"
)
fig.savefig(
    os.path.join(output_dir, f"dashboard_one_window_37levels_{run_tag}.png"),
    dpi=160,
    bbox_inches="tight"
)
fig_overview.savefig(
    os.path.join(output_dir, "temperature_distributions_overview.png"),
    dpi=160,
    bbox_inches="tight"
)
fig_overview.savefig(
    os.path.join(
        output_dir, f"temperature_distributions_overview_37levels_{run_tag}.png"),
    dpi=160,
    bbox_inches="tight"
)
cfv.plt.close(fig_overview)

print(f"Saved plots to: {os.path.abspath(output_dir)}")
print(f"Level count: {len(shared_levels)}")
if SHOW_PLOTS:
    cfv.plt.show(block=True)
else:
    cfv.plt.close("all")
# %%
