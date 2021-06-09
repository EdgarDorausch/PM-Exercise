from __future__ import annotations
from typing import *
if TYPE_CHECKING:
    pass

import matplotlib.pyplot as plt
from datastructures import Particle, VerletList
import numpy as np

class MyParticle(Particle):
    def __init__(self):
        super().__init__(np.random.uniform(size=[2]), np.random.randint(0,2,size=[1]))
    
    def evolve(self):
        pass

    def interact(self, other: Particle):
        pass

if __name__ == "__main__":
    np.random.seed(0)
    # pl = [MyParticle() for _ in range(100)]
    pos_list = np.random.uniform(size=[2,100])
    prop_list = np.random.uniform(size=[2,100])

    vl = VerletList(
        pos_list,
        prop_list,
        pos_min=np.array([0.,0.]),
        pos_max=np.array([1.,1.]),
        r_cutoff=0.1,
        r_skin=0.02
    )

    cl = vl.cell_list

    print(vl.verlet_list[2])

    print(cl.get_adjacent_cells([6,9]))

    get_len = np.vectorize(lambda x: len(x), otypes=[int])

    print(get_len(cl.cell_list)[::-1,:])

    cell = cl.cell_list[2,1]
    verlet_neighbors = vl.verlet_list[2]

    colors = ['r' if i in verlet_neighbors else 'b'  for i in range(100)]

    colors[2] = 'g'

    circle2 = plt.Circle(cl.position_list[::-1,2], vl.r_total, color='b', fill=False)

    ax = plt.gca()
    ax.cla() # clear things for fresh plot

    ax.scatter(y=cl.position_list[0,:], x=cl.position_list[1,:], c=colors)
    ax.set_xticks(np.arange(0, 1, step=0.1))
    ax.set_yticks(np.arange(0, 1, step=0.1))
    ax.grid(True)
    ax.add_patch(circle2)
    plt.show()



