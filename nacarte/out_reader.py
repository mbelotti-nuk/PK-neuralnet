from collections import namedtuple
from typing import List as list
from typing import Tuple as tuple
import numpy as np

Tally = namedtuple("Tally", ['x', 'y', 'z', 'result', 'error'])

class MeshReader:
    def __init__(self) -> None:
        pass

    def read_NACARTE_out(self, file:str, mcnp_order=False) -> list[Tally]:
        """
        Reads NACARTE output file
        Args:
            file (str): path to NACARTE output file 

        Returns:
            list[Tally]: returns a list of the tallies result of NACARTE
        """
        fin = open(file, "r")
        tallies = []
        x_ax, y_ax, z_ax = [], [], []
        header = True
        for line in fin:
            line = line.replace(",", ".")
            lsplit = line.split()
            if len(lsplit) == 0: continue
            if header: 
                header = False
                continue
            
            lsplit = [i for i in lsplit]
            if len(lsplit) > 4:
                err = lsplit[4]
            else:
                err = None
            tallies.append( Tally(x=float(lsplit[0]), 
                                  y=float(lsplit[1]), 
                                  z=float(lsplit[2]), 
                                  result=float(lsplit[3]), 
                                  error=float(err)) )

            if mcnp_order:
                x_ax.append(float(lsplit[0]))
                y_ax.append(float(lsplit[1]))
                z_ax.append(float(lsplit[2]))
        

        if mcnp_order:
            x_ax = np.unique(x_ax)
            y_ax = np.unique(y_ax)
            z_ax = np.unique(z_ax)

            n_x = len(x_ax)
            n_y = len(y_ax)
            n_z = len(z_ax)

            temp = [None for i in range(0,len(tallies))]
            
            counter = 0
            for k in range(0,n_z):
                for j in range(0,n_y):
                    for i in range(0,n_x): 
                        index = n_y * n_z * i + n_z * j + k                      
                        temp[index] = tallies[counter]
                        counter += 1

            tallies = temp

        return tallies


    def read_meshtal(self, file:str, mcnp_order=False) -> tuple[list[Tally], dict]:
        """Read a MCNP meshtal file. 
        Since NACARTE's output is listed in a different order,
        the tallies are re-ordered following NACARTE's convention

        Args:
            file (str): path to meshtal file

        Returns:
            list[Tally]: returns a list of the meshtal result of each tally
            dict : nx, ny and nz
        """
        print("mcnp reading")
        fin = open(file, "r")
        tallies = []
        
        counter = 0

        read = False
        for line in fin:
            lsplit = line.split()
            if len(lsplit) == 0: 
                continue
            
            if 'X direction:' in line:
                n_x = len(lsplit[2:]) - 1
            if 'Y direction:' in line:
                n_y = len(lsplit[2:]) - 1
            if 'Z direction:' in line:
                n_z = len(lsplit[2:]) - 1

            if lsplit[-1].lower() == "error":
                read = True
                n = n_x * n_y * n_z
                tallies = [None for i in range(0, n)]
                continue

            if read:
                lsplit = [float(i) for i in lsplit]

                k = counter % n_z
                j = ( counter//n_z ) % n_y
                i = ( counter // (n_z*n_y) ) % n_x

                if mcnp_order:
                    index = n_y * n_z * i + n_z * j + k
                else:
                    index = n_y * n_x * k + n_x * j + i

                tallies[index] = Tally(x=lsplit[1], y=lsplit[2], z=lsplit[3], result=lsplit[4], error=lsplit[5]) 

                counter += 1
                
        infos = {"nx":n_x, "ny":n_y, "nz":n_z}
        return tallies, infos
