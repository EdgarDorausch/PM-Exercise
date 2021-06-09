import pandas as pd
import numpy as np
from datastructures import Particle, VerletList
import matplotlib.pyplot as plt


class MyParticle(Particle):
    def __init__(self, pos: np.ndarray):
        super().__init__(pos, np.random.randint(0,2,size=[1]))
    
    def evolve(self):
        pass

    def interact(self, other: Particle):
        pass


if __name__ == "__main__":
    df: pd.DataFrame = pd.read_csv('Exercise01/QSBacterialPos.dat', sep="\s+", header=None)
    print(df)

    # pl = [MyParticle(row) for index, row in df.iterrows()]

    print(df.shape)
    vl = VerletList(
        df.to_numpy().T,
        np.empty([0,df.shape[0]]),
        pos_min=df.min(),
        pos_max=df.max(),
        r_cutoff=0.1,
        r_skin=0.02
    )

    cl = vl.cell_list

    highlight_particle = 2

    print(vl.verlet_list[highlight_particle])

    print(cl.get_adjacent_cells([6,9]))

    get_len = np.vectorize(lambda x: len(x), otypes=[int])

    print(get_len(cl.cell_list)[::-1,:])

    cell = cl.cell_list[highlight_particle,1]
    verlet_neighbors = vl.verlet_list[2]

    colors = ['r' if i in verlet_neighbors else 'b'  for i in range(df.shape[0])]

    colors[highlight_particle] = 'g'

    circle2 = plt.Circle(cl.position_list[::-1,2], vl.r_total, color='b', fill=False)

    ax = plt.gca()
    ax.cla() # clear things for fresh plot

    ax.scatter(y=cl.position_list[0,:], x=cl.position_list[1,:], c=colors)
    ax.set_xticks(np.arange(vl.cell_list.pos_min[1], vl.cell_list.pos_max[1], step=vl.r_total))
    ax.set_yticks(np.arange(vl.cell_list.pos_min[0], vl.cell_list.pos_max[0], step=vl.r_total))
    ax.grid(True)
    ax.add_patch(circle2)
    plt.show()
