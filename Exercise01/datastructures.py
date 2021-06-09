from __future__ import annotations
import time
from typing import *
if TYPE_CHECKING:
    pass

import numpy as np
from abc import ABC, abstractmethod


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
        position_list: np.ndarray, # Shape [POS_DIM, NUM_PART]
        property_list: np.ndarray, # Shape [PROP_DIM, NUM_PART]
        pos_min: np.ndarray, # Shape [POS_DIM]
        pos_max: np.ndarray, # Shape [POS_DIM]
        r_cutoff: float
    ):
        """
        """
        self.r_cutoff = r_cutoff
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.position_list = position_list
        self.properties_list = property_list

        POS_DIM, NUM_PART = position_list.shape
        PROP_DIM = property_list.shape[0]
        assert NUM_PART == property_list.shape[1], f'Length missmatch: {NUM_PART} != {property_list.shape[1]}'

        self.grid_res = np.ceil((pos_max-pos_min)/r_cutoff).astype(int)

        # create numpy array containing the cell lists
        fill_empty_list = np.vectorize(lambda x: [], otypes=[object])
        self.cell_list = fill_empty_list(np.empty(shape=self.grid_res, dtype=object))

        # fill arrays
        for p_idx in range(NUM_PART):
            cell_idx = tuple(x for x in np.nditer(np.floor((position_list[:,p_idx]-pos_min)/r_cutoff).astype(int)))
            self.cell_list[cell_idx].append(p_idx)

    def get_adjacent_cells(self, cell_index: Sequence[int]):
        # generate all neighbors along one dimension
        def dim_neighbors(neighbors: List[Tuple[int,...]], dim: int):
            for neighbor in neighbors:
                x = neighbor[0:dim]
                for delta in [1,-1]:
                    if not 0 <= neighbor[dim]+delta < self.grid_res[dim]: # If not in grid
                        continue
                    y = x + (neighbor[dim]+delta,) 
                    if dim < len(neighbor): # If last dimension
                        y += neighbor[dim+1:]
                    yield y


        neighbor_list = [tuple(cell_index)]
        for dim in range(len(cell_index)):
            neighbor_list.extend(list(dim_neighbors(neighbor_list, dim)))
        
        return neighbor_list[1:] # dont return current cell

    def get_cell_index(self, particle_index) -> Tuple[int, ...]:
        return tuple(x for x in np.nditer(np.floor((self.position_list[:, particle_index]-self.pos_min)/self.r_cutoff).astype(int)))

    def get_particle_distance(self, p_idx, q_idx):
        diff = self.position_list[:,p_idx] - self.position_list[:,q_idx]
        return np.linalg.norm(diff)



class VerletList:
    def __init__(self,
        position_list: np.ndarray, # Shape [POS_DIM, NUM_PART]
        property_list: np.ndarray, # Shape [PROP_DIM, NUM_PART]
        pos_min: np.ndarray, # Shape [POS_DIM]
        pos_max: np.ndarray, # Shape [POS_DIM]
        r_cutoff: float,
        r_skin: float
    ) -> None:
        NUM_PART = position_list.shape[1]
        print(f'Constructing Verlet List for {NUM_PART} particles')
        t0 = time.time()

        self.r_total = r_cutoff + r_skin
        self.r_skin = r_skin
        self.r_cutoff = r_cutoff
        self.cell_list = CellList(position_list, property_list, pos_min, pos_max, self.r_total)

        fill_empty_list = np.vectorize(lambda x: [], otypes=[object])
        self.verlet_list = fill_empty_list(np.empty(shape=[NUM_PART], dtype=object))

        for particle_index in range(NUM_PART):

            own_cell_index =  self.cell_list.get_cell_index(particle_index)
            own_cell_list = self.cell_list.cell_list[own_cell_index]
            self.verlet_list[particle_index].extend([
                q_idx for q_idx in own_cell_list if self.cell_list.get_particle_distance(q_idx, particle_index) <= self.r_total
                ])

            for neighbor_cell_index in self.cell_list.get_adjacent_cells(own_cell_index):
                neighbor_cell_list = self.cell_list.cell_list[neighbor_cell_index]
                self.verlet_list[particle_index].extend([
                    q_idx for q_idx in neighbor_cell_list if self.cell_list.get_particle_distance(q_idx, particle_index) <= self.r_total
                    ])

        t1 = time.time()
        print(f'Verlet List construction took ~{t1-t0:.5}s')
        
