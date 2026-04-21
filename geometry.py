# %%

import numpy as np
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.core as cfc


# Boundary markers
NO_BOUNDARY_CONDITION = 0
BOUNDARY_CONDITION_CONVECTION = 1
BOUNDARY_CONDITION_QOUT = 2


def create_geometry():
    geo = cfg.Geometry()
    r = 0.002
    # Coordinates for the centers of the circles (holes) in the rectangle
    centers = [[0.004, 0.008], [0.006, 0.018], [0.004, 0.028], [0.006, 0.038]]

    # ID's to keep track of points and lines
    p_id = 1
    l_id = 1

    # Points for the rectangle
    geo.point([0, 0], ID=p_id)
    p1 = p_id
    p_id += 1
    geo.point([0, 0.046], ID=p_id)
    p2 = p_id
    p_id += 1
    geo.point([0.01, 0.046], ID=p_id)
    p3 = p_id
    p_id += 1
    geo.point([0.01, 0], ID=p_id)
    p4 = p_id
    p_id += 1

    # Lines for the rectangle
    l1 = l_id
    geo.line([p1, p2], ID=l1, marker=NO_BOUNDARY_CONDITION)
    l_id += 1
    l2 = l_id
    geo.line([p2, p3], ID=l2, marker=BOUNDARY_CONDITION_CONVECTION)
    l_id += 1
    l3 = l_id
    geo.line([p3, p4], ID=l3, marker=NO_BOUNDARY_CONDITION)
    l_id += 1
    l4 = l_id
    geo.line([p4, p1], ID=l4, marker=BOUNDARY_CONDITION_QOUT)
    l_id += 1

    outer_loop = [l1, l2, l3, l4]

    # Create circles
    all_hole_loops = []  # Lista som ska innehålla listor av linje-ID:n

    for c_coord in centers:
        x, y = c_coord
        # Definiera en mindre storlek för cirkelpunkterna
        # Prova dig fram, t.ex. r/5 eller r/10
        c_el_size = 0.0005

        # Create points for the circle
        cp = p_id
        geo.point([x, y], ID=cp, el_size=c_el_size)
        p_id += 1
        pr = p_id
        geo.point([x+r, y], ID=pr, el_size=c_el_size)
        p_id += 1
        pt = p_id
        geo.point([x, y+r], ID=pt, el_size=c_el_size)
        p_id += 1
        pl = p_id
        geo.point([x-r, y], ID=pl, el_size=c_el_size)
        p_id += 1
        pb = p_id
        geo.point([x, y-r], ID=pb, el_size=c_el_size)
        p_id += 1

        # Create lines for the circle and assign boundary conditions
        c1 = l_id
        geo.circle([pr, cp, pt], ID=c1, marker=BOUNDARY_CONDITION_CONVECTION)
        l_id += 1
        c2 = l_id
        geo.circle([pt, cp, pl], ID=c2, marker=BOUNDARY_CONDITION_CONVECTION)
        l_id += 1
        c3 = l_id
        geo.circle([pl, cp, pb], ID=c3, marker=BOUNDARY_CONDITION_CONVECTION)
        l_id += 1
        c4 = l_id
        geo.circle([pb, cp, pr], ID=c4, marker=BOUNDARY_CONDITION_CONVECTION)
        l_id += 1

        all_hole_loops.append([c1, c2, c3, c4])

    # Create the surface with holes
    geo.surface(outer_loop, holes=all_hole_loops)
    # Create a mesh and retrieve variables relevant for FEM
    coords, edof, dofs, bdofs, elementmarkers = cfm.create_mesh(
        geo, el_type=2, el_size_factor=0.08, dofs_per_node=1)

    return geo, coords, edof, dofs, bdofs, elementmarkers


if __name__ == "__main__":
    geo, coords, edof, dofs, bdofs, elementmarkers = create_geometry()
    cfv.figure(fig_size=(10, 10))
    cfv.draw_geometry(geo)
    cfv.figure()
    cfv.drawMesh(
        coords=coords,
        edof=edof,
        dofs_per_node=1,
        el_type=2,
        filled=True
        # coords, edof, dofs, bdofs, elementmarkers = cfm.create_mesh(g3, el_type=2, el_size_factor=0.08, dofs_per_node=1)
    )
    # number of degrees of freedom, which happens to be equal to number of nodes since dofs per node = 1
    nDofs = np.size(dofs)
    # ex is the x-coordinates for each node in each element, ey same same
    ex, ey = cfc.coordxtr(edof, coords, dofs)
    n_elements = (edof.shape)[0]
    cfv.show()
# %%
