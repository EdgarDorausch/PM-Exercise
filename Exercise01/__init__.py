from __future__ import annotations
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
    def __init__(self, part_list: List[T], pos_min: np.ndarray, pos_max: np.ndarray, pos_res: List[int]):
        particle_position_shape = part_list[0].position.shape
        particle_properties_shape = part_list[0].properties.shape

        length = len(list)
        self.position_list = np.empty([*particle_position_shape, length])
        self.properties_list = np.empty([*particle_properties_shape, length])


        fill_empty_list = np.vectorize(lambda x: [], otypes=[object])
        self.cell_list = fill_empty_list(np.empty(shape=pos_res, dtype=object))

        for i, particel in enumerate(part_list):
            self.position_list[:,i] = particel.position
            self.properties_list[:,i] = particel.properties

            cell_idx = ((particel.position-pos_min)/(pos_max-pos_min))
            #Todo:





        