#%% 
%matplotlib inline
import numpy as np
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.core as cfc
import matplotlib.pyplot as plt
from plantml import plantml

from geometry_test import create_geometry, NO_BC, BC_CONV, BC_TEMP

alpha = 1000
k= 10
t = 0.01
Q = 4e5 # W/m^3
T_inf = 293 # K
rho = 1500 # kg/m^3
cp = 800 # J/kgK
E = 30 # GPa
poisson = 0.3 
q_out = 2000 # W/m^2
T_0 = 293 # K




geo, coords, edof, dofs, bdofs, elementmarkers = create_geometry()

nDofs = np.size(dofs)
K = np.zeros((nDofs, nDofs))
f = np.zeros((nDofs, 1))

# Materialmatris för värmeledning
D = np.array([[k, 0], 
              [0, k]])

for el_nodes, el_marker in zip(edof, elementmarkers):
    # el_nodes är dofs för det aktuella elementet
    idx = el_nodes - 1 # Indexera för Python (0-baserat)
    
    # ex och ey för elementet
    ex_el = coords[idx, 0]
    ey_el = coords[idx, 1]
    
    # Elementstyvhetsmatris (fl_mat) och lastvektor (fl_rhs)
    # ep = [tjocklek], Q = källterm (W/m^3)
    # 1. Anropa funktionen (Q kan vara 0 eller ett värde)
    res = cfc.flw2te(ex_el, ey_el, [t], D, Q)

# Kontrollera om vi fick både Ke och fe, eller bara Ke
    if isinstance(res, tuple):
        Ke, fe = res
    else:
        Ke = res
    # Om Q=0 returnerar CALFEM bara Ke. Då skapar vi en nollvektor för fe manuellt.
    # En triangel (flw2te) har 3 noder, alltså 3 rader i fe.
        fe = np.zeros((3, 1)) 

# Assemblera in i globala K och f
    cfc.assem(el_nodes, K, Ke, f, fe)
    

# Hämta noder på ränder som sets för snabb sökning
conv_nodes = set(bdofs[BC_CONV])
qout_nodes = set(bdofs[BC_TEMP])

for el_nodes in edof:
    
    nodes_on_conv = [n for n in el_nodes if n in conv_nodes]
    
    if len(nodes_on_conv) == 2:
        n1, n2 = nodes_on_conv
        c1, c2 = coords[n1-1], coords[n2-1]
        L = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
        
        Ke_conv = (alpha * L * t / 6) * np.array([[2, 1], [1, 2]])
        fe_conv = (alpha * T_inf * L * t / 2) * np.array([[1], [1]])
        
        # Här skickar vi med både K och f som vanligt
        cfc.assem(np.array([n1, n2]), K, Ke_conv, f, fe_conv)

    """ # Ingen konstant flöde  
    nodes_on_qout = [n for n in el_nodes if n in qout_nodes]
    
    if len(nodes_on_qout) == 2:
        n1, n2 = nodes_on_qout
        c1, c2 = coords[n1-1], coords[n2-1]
        L = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
        
        # Skapa en noll-matris för Ke eftersom flödet inte påverkar styvheten
        # Detta är säkrare än None för CALFEM
        Ke_zero = np.zeros((2, 2))
        fe_qout = (-q_out * L * t / 2) * np.array([[1], [1]])
        
        cfc.assem(np.array([n1, n2]), K, Ke_zero, f, fe_qout)  """
bc_nodes = np.array(list(bdofs[BC_TEMP]))
bc_values = np.ones(len(bc_nodes)) * 373
# 4. Lös systemet
print("bc_nodes:", bc_nodes)
print(np.shape(K))
print(np.shape(f))
print(np.shape(bc_nodes))
print(np.shape(bc_values))
a, r = cfc.solveq(K, f, bc_nodes, bc_values)
# Koordinater vi letar efter
targets = {"Nod 4 (Mitten)": [0.005, 0.005], "Nod 7 (Hörn)": [0.005, 0.01]}

for name, pos in targets.items():
    
    dists = np.sqrt((coords[:,0]-pos[0])**2 + (coords[:,1]-pos[1])**2)
    node_idx = np.argmin(dists)
    print(f"{name} (Index {node_idx}): {a[node_idx][0]:.2f} K")
print("Nodal temperatures a:\n", a)

plt.figure(figsize=(10, 8))
plt.rc('image', cmap='inferno')
cfv.draw_nodal_values(a, coords, edof, title="Temperaturfördelning")
cfv.colorbar()


# %%
