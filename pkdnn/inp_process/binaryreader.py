import os 
import numpy as np  
import array
from typing import List as list

class raw_reader:

    def __init__(self, path:str, size:int):
        self.path = path
        self.size = size
        self.doses = []
        self.errors = []
        self.coordinates = []

    def binary_reader(self, filename:str):
       
        fn = os.path.join(self.path, filename)

        a = array.array('f')
        a.fromfile(open(fn, 'rb'), os.path.getsize(fn) // a.itemsize)
        arr = np.copy(a)

        self.doses = arr[:self.size]
        self.errors = arr[self.size:]

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
        if(len(self.coordinates) != len(self.doses) | len(self.coordinates) == 0 | len(self.doses) == 0):
            raise Exception("Error")
        mask = self.errors < max_Error
        self.doses = self.doses[mask]
        self.errors = self.errors[mask]
        self.coordinates = self.coordinates[mask]