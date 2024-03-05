from collections import namedtuple
Tally = namedtuple("Tally", ['x', 'y', 'z', 'result', 'error'])

class MeshReader:
    def __init__(self) -> None:
        pass

    def read_NACARTE_out(self, file:str) -> list[Tally]:
        fin = open(file, "r")
        tallies = []
        header = True
        for line in fin:
            line = line.replace(",", ".")
            lsplit = line.split()
            if len(lsplit) == 0: continue
            if header: 
                header = False
                continue
            lsplit = [float(i) for i in lsplit]
            if len(lsplit) > 4:
                err = lsplit[4]
            else:
                err = None
            tallies.append( Tally(x=lsplit[0], y=lsplit[1], z=lsplit[2], result=lsplit[3], error=err) )
        return tallies


    def read_meshtal(self, file:str) -> list[Tally]:
        """
        read a MCNP meshtal file. 
        Since NACARTE's output is listed in a different order,
        the tallies are re-ordered following NACARTE's convention
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

                index = n_y * n_x * k + n_x * j + i

                tallies[index] = Tally(x=lsplit[1], y=lsplit[2], z=lsplit[3], result=lsplit[4], error=lsplit[5]) 

                # if k < n_z-1: 
                #     k+=1
                # else:
                #     k = 0
                #     if j < n_y-1:
                #         j +=1
                #     else:
                #         j = 0 
                #         i += 1

                counter += 1

        return tallies
