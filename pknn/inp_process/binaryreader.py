import os 
import numpy as np  
import array
from typing import List as list

class raw_reader:
    """Class that reads binary MCNP meshtal files
    """
    def __init__(self, path:str, size:int):
        """Initializer

        :param path: path to raw binary MCNP meshtal file
        :type path: str
        :param size: number of voxels in the mesh
        :type size: int
        """    
        self.path = path
        self.size = size
        self._doses = []
        self._errors = []
        self.coordinates = []

    @property
    def doses(self) -> list[float]:
        return self._doses

    def binary_reader(self, filename:str):
       
        fn = os.path.join(self.path, filename)

        a = array.array('f')
        a.fromfile(open(fn, 'rb'), os.path.getsize(fn) // a.itemsize)
        arr = np.copy(a)

        self._doses = arr[:self.size]
        self._errors = arr[self.size:]

    def set_mesh(self, origin:list[float], end:list[float], counts:list[int]):
        xDiv = self.get_division(origin[0], end[0], counts[0] )
        yDiv = self.get_division(origin[1], end[1], counts[1] )
        zDiv = self.get_division(origin[2], end[2], counts[2] )

        self.coordinates = np.empty([self.size,3])
        ind = 0

        for i in xDiv:
            for j in yDiv:
                for k in zDiv:
                    self.coordinates[ind] = np.array([i, j, k])
                    ind += 1



    def get_division(self, Start, End, Int):
        step = (End-Start)/Int
        div = np.empty(Int)
        for i in range(0,Int):
            div[i] = (Start + step/2) + i*step
        return div
    
    def filter(self, max_Error=1):
        if(len(self.coordinates) != len(self._doses) | len(self.coordinates) == 0 | len(self._doses) == 0):
            raise Exception("Error")
        mask = self._errors < max_Error
        self._doses = self._doses[mask]
        self._errors = self._errors[mask]
        self.coordinates = self.coordinates[mask]

    @property
    def dose(self)->np.array:
        """Return the doses inside the raw MCNP meshtal file

        :return: dose
        :rtype: np.array
        """                
        return self._doses
    
    @property
    def errors(self)->np.array:
        """Return the errors inside the raw MCNP meshtal file

        :return: errors
        :rtype: np.array
        """                
        return self._errors