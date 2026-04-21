# %%

import numpy as np
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.core as cfc
from geometry import create_geometry, NO_BOUNDARY_CONDITION, BOUNDARY_CONDITION_CONVECTION, BOUNDARY_CONDITION_QOUT

# == Parameters ==
E = 30  # GPa
poisson = 0.3
alpha = 8e-6  # 1/K
rho = 1500  # kg/m^3
cp = 800  # J/kgK
k = 2  # W/mK
q_out = 2000  # W/m^2
Q = 4e5  # W/m^3
T_0 = 293  # K
T_inf = 293  # K
ac = 50  # W/m^2K
t = 1  # m
# ====

geo, coords, edof, dofs, bdofs, elementmarkers = create_geometry()

nDofs = np.size(dofs)
K = np.zeros((nDofs, nDofs))
f = np.zeros((nDofs, 1))

# Materialmatris för värmeledning
D = np.array([[k, 0],
              [0, k]])

for el_nodes, el_marker in zip(edof, elementmarkers):
    # el_nodes är dofs för det aktuella elementet
    idx = el_nodes - 1  # Indexera för Python (0-baserat)

    # ex och ey för elementet
    ex_el = coords[idx, 0]
    ey_el = coords[idx, 1]

    # Elementstyvhetsmatris (fl_mat) och lastvektor (fl_rhs)
    # ep = [tjocklek], Q = källterm (W/m^3)
    Ke, fe = cfc.flw2te(ex_el, ey_el, [t], D, Q)

    # Assemblera in i globala K och f
    cfc.assem(el_nodes, K, Ke, f, fe)
print("Global stiffness matrix K:\n", K)

# --- Fortsättning efter din första loop ---

# Hämta noder på ränder
conv_nodes = set(bdofs[BOUNDARY_CONDITION_CONVECTION])
qout_nodes = set(bdofs[BOUNDARY_CONDITION_QOUT])

# 2. Loop för Konvektion (Kc och fc)
for el_nodes in edof:
    # Hitta noder i elementet som ligger på konvektionsranden
    nodes_on_boundary = [n for n in el_nodes if n in conv_nodes]

    if len(nodes_on_boundary) == 2:
        n1, n2 = nodes_on_boundary
        c1 = coords[n1-1]
        c2 = coords[n2-1]
        L = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

        # Kc_el = (ac*L*t/6) * [[2, 1], [1, 2]]
        Ke_conv = (ac * L * t / 6) * np.array([[2, 1], [1, 2]])
        # fc_el = (ac*T_inf*L*t/2) * [1, 1]
        fe_conv = (ac * T_inf * L * t / 2) * np.array([[1], [1]])

        cfc.assem(np.array([n1, n2]), K, Ke_conv, f, fe_conv)

# 3. Loop för utgående flöde q_out (fb)
for el_nodes in edof:
    nodes_on_qout = [n for n in el_nodes if n in qout_nodes]

    if len(nodes_on_qout) == 2:
        n1, n2 = nodes_on_qout
        c1 = coords[n1-1]
        c2 = coords[n2-1]
        L = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

        # q_out är positivt utåt i fysiken, men i f-vektorn
        # representerar det tillförd effekt. Sätt minus för bortförd värme.
        fe_qout = (-q_out * L * t / 2) * np.array([[1], [1]])

        cfc.assem(np.array([n1, n2]), K, np.zeros((2, 2)), f, fe_qout)

# 4. Lös systemet
a = np.linalg.solve(K, f)

# 5. Visualisera
cfv.draw_nodal_values(a, coords, edof, title="Temperaturfördelning")
cfv.showAndWait()
# %%
