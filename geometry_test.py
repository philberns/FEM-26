#%%

import numpy as np
import calfem.vis_mpl as cfv
import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm

# Markörer
NO_BC = 0
BC_CONV = 1
BC_TEMP = 2

def create_geometry():
    geo = cfg.Geometry()
    coords = [[0,0], [0.005,0], [0.01,0], [0,0.005], [0.005,0.005], 
              [0.01,0.005], [0,0.01], [0.005,0.01], [0.01,0.01]]
    for c in coords:
        geo.point(c)

    
    # Ytterkanter (Linje-index 0-7)
    geo.line([0, 1], marker=NO_BC)   # ID 0
    geo.line([1, 2], marker=NO_BC)   # ID 1
    geo.line([2, 5], marker=BC_TEMP) # ID 2
    geo.line([5, 8], marker=BC_TEMP) # ID 3
    geo.line([8, 7], marker=BC_CONV) # ID 4
    geo.line([7, 6], marker=NO_BC)   # ID 5
    geo.line([6, 3], marker=NO_BC) # ID 6
    geo.line([3, 0], marker=NO_BC) # ID 7

    # Inre linjer för att koppla mot mitten (p4) (Linje-index 8-23)
    # Vi skapar dem dubbelt för att slippa använda minus-tecken i ytorna
    geo.line([0, 4]); geo.line([4, 0]) # ID 8, 9
    geo.line([1, 4]); geo.line([4, 1]) # ID 10, 11
    geo.line([2, 4]); geo.line([4, 2]) # ID 12, 13
    geo.line([5, 4]); geo.line([4, 5]) # ID 14, 15
    geo.line([8, 4]); geo.line([4, 8]) # ID 16, 17
    geo.line([7, 4]); geo.line([4, 7]) # ID 18, 19
    geo.line([6, 4]); geo.line([4, 6]) # ID 20, 21
    geo.line([3, 4]); geo.line([4, 3]) # ID 22, 23

    # Skapa 8 ytor (Använd de interna linje-indexen)
    # T1: p0->p1, p1->p4, p4->p0
    geo.surface([0, 10, 9])  
    # T2: p1->p2, p2->p4, p4->p1
    geo.surface([1, 12, 11]) 
    # T3: p2->p5, p5->p4, p4->p2
    geo.surface([2, 14, 13]) 
    # T4: p5->p8, p8->p4, p4->p5
    geo.surface([3, 16, 15]) 
    # T5: p8->p7, p7->p4, p4->p8
    geo.surface([4, 18, 17]) 
    # T6: p7->p6, p6->p4, p4->p7
    geo.surface([5, 20, 19]) 
    # T7: p6->p3, p3->p4, p4->p6
    geo.surface([6, 22, 21]) 
    # T8: p3->p0, p0->p4, p4->p3
    geo.surface([7, 8, 23])

    return geo, *cfm.create_mesh(geo, el_type=2, el_size_factor=1.0, dofs_per_node=1)

geo, coords, edof, dofs, bdofs, elementmarkers = create_geometry()
cfv.figure(fig_size=(10,10))
cfv.draw_geometry(geo)
cfv.figure()
cfv.drawMesh(
    coords=coords,
    edof=edof,
    dofs_per_node=1,
    el_type=2,
    filled=True
)
nDofs = np.size(dofs) # number of degrees of freedom, which happens to be equal to number of nodes since dofs per node = 1
ex, ey = cfc.coordxtr(edof, coords, dofs) # ex is the x-coordinates for each node in each element, ey same same
n_elements = (edof.shape)[0]
# %%
