from __future__ import annotations
from typing import *
if TYPE_CHECKING:
    pass

import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class Particle(ABC):
    def __init__(self, position: np.ndarray, properties: np.ndarray):
        self.position = position
        self.properties = properties

    @abstractmethod
    def evolve(self):
        raise NotImplementedError()

    @abstractmethod
    def interact(self, other: Particle):
        raise NotImplementedError()


T = TypeVar('T', bound=Particle)

class CellList:
    def __init__(self,
        part_list: List[T],
        pos_min: np.ndarray, # Shape [POS_DIM]
        pos_max: np.ndarray, # Shape [POS_DIM]
        r_cutoff: float
    ):
        """
        """
        self.r_cutoff = r_cutoff
        self.pos_min = pos_min
        self.pos_max = pos_max

        POS_DIM = part_list[0].position.shape[0]
        PROP_DIM = part_list[0].properties.shape[0]

        grid_res = np.ceil((pos_max-pos_min)/r_cutoff).astype(int)

        # Construct numpy arrays containing all position and all property information respectively
        length = len(part_list)
        self.position_list = np.empty([POS_DIM, length])
        self.properties_list = np.empty([PROP_DIM, length])

        # create numpy array containing the cell lists
        fill_empty_list = np.vectorize(lambda x: [], otypes=[object])
        self.cell_list = fill_empty_list(np.empty(shape=grid_res, dtype=object))

        # fill arrays
        for i, particel in enumerate(part_list):
            self.position_list[:,i] = particel.position
            self.properties_list[:,i] = particel.properties

            cell_idx = tuple(x for x in np.nditer(np.floor((particel.position-pos_min)/r_cutoff).astype(int)))
            foo = self.cell_list[cell_idx]
            foo.append(i)


class MyParticle(Particle):
    def __init__(self):
        super().__init__(np.random.uniform(size=[2]), np.random.randint(0,2,size=[1]))
    
    def evolve(self):
        pass

    def interact(self, other: Particle):
        pass

if __name__ == "__main__":
    np.random.seed(0)
    pl = [MyParticle() for _ in range(100)]
    
    cl = CellList(
        pl,
        pos_min=np.array([0.,0.]),
        pos_max=np.array([1.,1.]),
        r_cutoff=0.1
    )
    print(cl)

    get_len = np.vectorize(lambda x: len(x), otypes=[int])

    print(get_len(cl.cell_list)[::-1,:])

    cell = cl.cell_list[2,1]

    colors = ['r' if i in cell else 'b'  for i in range(100)]

    plt.scatter(y=cl.position_list[0,:], x=cl.position_list[1,:], c=colors)
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.grid(True)
    plt.show()






        