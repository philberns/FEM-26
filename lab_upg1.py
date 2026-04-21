import numpy as np
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.core as cfc


A = 10
k = 5
L = 6
Q = 100

koord = [2, 4, 6, 8]

Edof = [[1, 1, 2],
        [2, 2, 3],
        [3, 3, 4]]

nen = 2  # noder per element
ndof = 4  # antal frihetsgrader (1D)
nelm = len(Edof)  # antal element
Dof = np.arange(1, ndof + 1)  # alla frihetsgrader

ex = cfc.coordxtr(Edof, np.array(koord).reshape(-1, 1),
                  Dof.reshape(-1, 1), nen)
print("ex:\n", ex)
