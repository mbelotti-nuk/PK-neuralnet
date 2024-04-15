import random
from scipy.stats import qmc
import numpy as np
import struct
from .binaryreader import raw_reader
import math
import os
from numpy import linalg as LA
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from typing import List as list, Dict as dict

class input_admin:
    """This class permits to calculate relevant values for the standard radiation configuration: 
    point radiaiton source ||   wall   || tallies.
    """     
    def __init__(self, plane_normal:np.array, ro:float ,mass_fraction:list[float]=None, atomic_number:list[int]=None,
                 path_to_dose_conversion:str="", path_to_mass_att_coeff:str="",
                 energy:float=None, plane_coords:list[float]=None):    
        """calculate relevant values for the standard radiation configuration: 
        point radiaiton source ||   wall   || tallies.

        Args:
            plane_normal (np.array): normalized vector indicating the normale to the wall surface
            ro (float): density of the wall's material
            mass_fraction (list[float], optional): list of the fraction of each element into the materials' wall. Defaults to None.
            atomic_number (list[int], optional): list of the atomic number of each element into the materials' wall. Defaults to None.
            path_to_dose_conversion (str, optional): absolute path to dose conversion factors file. .
            path_to_mass_att_coeff (str, optional): absolute path to mass attenuation coefficients file.
            energy (float, optional):energy of the point radiation source. Defaults to None.
            plane_coords (list[float], optional): coordinates of the planes of the wall. Defaults to None.


        """    


#           The standard configuration with its reference system is like:
#                                          
#                                           Wall
#                      point                 ___
#                      source               |   |                  Tallies
#    ^ Z                                    |   |                ___________
#    |                   *                  |   |               |__|__|__|__|
#    |                                      |   |               |__|__|__|__|   
#    |-------->                             |   |               |__|__|__|__|
#              Y                            |___|
#                                          p0   p1


        # Parameters characterizing wall material
        # Default to concrete values
        self.mass_fraction = mass_fraction if mass_fraction != None \
            else [0.0056, 0.497500, 0.017100, 0.002600, 0.046900, 0.314700, 0.001300, 0.019200, 0.082800, 0.012400]
        self.atomic_number = atomic_number if atomic_number != None \
            else [ 1, 8, 11, 12, 13, 14, 16, 19, 20, 26]
        self.ro = ro

        # Energy bins and relative values of dose conversion factor
        assert os.path.isfile(path_to_dose_conversion), "Dose conversion factor file was not found"
        self.cv_energies, self.cvs = self._get_data(path_to_dose_conversion, 1) # [ (pSv) /(gamma/cm^2) ]
        # Absolute path to mass attenuation coefficients
        assert os.path.isfile(path_to_mass_att_coeff), "Mass attenuation coefficient file was not found"
        self.path_to_mass_att_coeff = path_to_mass_att_coeff


        # mu: Attenuation coefficient of the material
        # ro: Density of materials' wall
        # dose_conversion_factor: # Flux to Dose conversion factor
        # energy: energy of the source
        self.mu, self.energy, self.dose_conversion_factor = 0, 0, 0

        if energy != None: self.define_energy(energy)

        # Wall variables
        # Planes of the wall
        self.plane_normal = plane_normal
        # Planes Y-coordinate
        self.p0, self.p1 = 0, 0
        self.indices = self._define_wall_orientation()
        if plane_coords != None: self.define_wall_coords(plane_coords)
        
        # Coefficients for NN
        self.dist_source_tally = 0
        self.delta_t = 0
        self.angle = 0
        self.dist_source_shield = 0
        self.dist_shield_tally = 0
        self.theta = 0
        self.fi = 0
        self._slab_thickness = 0

# ********************************************************************
#                           Public members
# ********************************************************************

    def calculate_dose_direct_contribution(self, source:list[float], tally:list[float]) -> float:
        """Calulate the direct contribution to the dose at the tally

        Args:
            source (list[float]): source position
            tally (list[float]): tally position

        Returns:
            float: dose direct contribution
        """
        sourceToTally = np.array([tally[0]-source[0], tally[1]-source[1], tally[2]-source[2]])
        
        self.dist_source_tally = LA.norm(sourceToTally)
        self.delta_t, self.dist_source_shield, self.dist_shield_tally = self._wall_intersection(source, tally ,sourceToTally)
        self.angle = self._angle_between(self.plane_normal, sourceToTally)

        # POLAR COORDINATES
        self.theta, self.fi = self._get_polar_coordinates(sourceToTally)

        attenuation = math.exp(- self.mu * self.ro * self.delta_t) # t must be in cm
        uncoll_fi = 1/(4*math.pi*self.dist_source_tally**2) # [ gamma/cm^2 ]
        dose = self.dose_conversion_factor*uncoll_fi*attenuation

        return dose

    def get_coefficients(self):
        return {    'energy': self.energy,  'slab_thickness': self._slab_thickness,             'dist_source_tally': self.dist_source_tally, \
                    'delta_t':self.delta_t,  'angle':self.angle,                         'theta':self.theta,\
                    'fi':self.fi ,          'dist_source_shield':self.dist_source_shield,    'dist_shield_tally':self.dist_shield_tally,\
                    'mfp':self.delta_t * self.ro* self.mu }
    
    def define_wall_coords(self, plane_coord:list[float]):
        """Instantiates the coordinates of the planes that defines the wall

        Args:
            plane_coord (list[float]): the two plane values (default on Y-axis)
        """        
        assert len(plane_coord) == 2, "There must be two values for plane_coord"
        p0, p1 = plane_coord
        if(abs(p1)<abs(p0)):
            p0, p1 = self._swap(p0, p1)
        self.p0 = p0
        self.p1 = p1
        return
    
    def define_energy(self, energy:float):
        """Instantiate energy and energy related coeffs.
        In particular it defines: 1. energy [MeV], 2. Mass attenuation coefficiente, 
        3. Dose conversion factor

        Args:
            energy (float): _description_
        """        
        self.energy = energy
        #self.mu =  self.defineData(self.Mu_Energies, self.Mus)
        self.mu = self._define_mass_att_coeff()
        self.dose_conversion_factor = self._define_data(self.cv_energies, self.cvs)

    @property
    def slab_thickness(self):
        return self._slab_thickness
    
    @slab_thickness.setter
    def slab_thickness(self, thick:float):
        self._slab_thickness = thick

# ********************************************************************
#                           Private members
# ********************************************************************

    def _define_mass_att_coeff(self) -> float:
        """Calculates the mass attenuation coefficient for the material defined
        in the input manager at the energy self.energy.

        Args:

        Returns:
            float: the mass attenuation coefficient
        """
        path = self.path_to_mass_att_coeff
        tree = ET.parse(path)
        root = tree.getroot()
        MU = 0
        for i in range(0,len(self.atomic_number)):
            thismu = self._get_mass_att_coeff(self.atomic_number[i], root)
            MU += self.mass_fraction[i]*thismu
        return MU

    def _swap(self, t0, t1):
        """Changes t0 for t1 and vice-versa
        """
        temp = t1
        t1 = t0
        t0 = temp
        return t0, t1
    
    def _define_wall_orientation(self):
        """Defines if the wall surfaces are perpendicular to X, Y or Z

        Returns:
            np.array: list of axis logically ordered with respect to wall orienation. 
            In the array the 0 correspond to X, the 1 to Y and the 2 to Z.
            The first value of the list is the axis perpendicular to wall surfaces
        """        
        x_orientation = self._angle_between(self.plane_normal, np.array([1,0,0]))
        y_orientation = self._angle_between(self.plane_normal, np.array([0,1,0]))
        z_orientation = self._angle_between(self.plane_normal, np.array([0,0,1]))
        if x_orientation == 0 or x_orientation == 180:
            return np.array([0,1,2])
        elif y_orientation == 0 or y_orientation == 180:
            return np.array([1,0,2])
        elif z_orientation == 0 or z_orientation == 180:
            return np.array([2,0,1])
        else:
            raise ValueError("Only plane normals parallel to x, y or z axis are allowed")

    def _get_mass_att_coeff(self, Z:int, root):
        textE = root[Z-1][0].text.split()
        energies = [float(e) for e in textE]

        text_att_coeff = root[Z-1][1].text.split()
        mass_att_Coeff = [float(m) for m in text_att_coeff]

        mu = self._define_data(energies, mass_att_Coeff)
        return mu
    
    def _get_data(self, filepath, ind):
        fdata = open(filepath, encoding="utf8")
        read = False
        energy = []
        data = []
        for line in fdata:
            lsplit = line.split()
            if read:
                e = float(lsplit[0])
                mu_e = float(lsplit[ind])
                energy += [e]
                data += [mu_e]
            else:
                if len(lsplit) == 0:
                    read = True
        return energy, data

    def _define_data(self, energies:list[float], datas:list[float])->float:
        """Browse the energy bins until the bin in which falls self.energy.
        Then it perform an interpolation bewteen the datas for the two extreme of the bin. 

        Args:
            energies (list[float]): list of energy bins
            datas (list[float]): list of corresponding data bins

        Returns:
            float: interpolated data for energy=self.energy
        """

        downE = 0
        downMuE = 0
        upE = 0
        upMuE = 0
        for i in range(0, len(energies)):
            e = energies[i]
            mu_e = datas[i]
               
            if(e <= self.energy):
                downE = e
                downMuE = mu_e
            else:
                upE = e
                upMuE = mu_e
                break
        return self._interp(self.energy, downE, upE, downMuE, upMuE)

    def _interp(self, x, x1, x2, y1, y2):
        fract = (x - x1) / (x2 - x1)
        return y1 + (y2 - y1) * fract

    def _wall_intersection(self, source, tally,  v):

        intersection_0 = self._intersection_on_plane(self.p0, v, source)
        intersection_1 = self._intersection_on_plane(self.p1, v, source)
        intersection = np.array([intersection_0[0]-intersection_1[0], intersection_0[1]-intersection_1[1], intersection_0[2]-intersection_1[2]])
        
        return LA.norm(intersection), LA.norm(source-intersection_0), LA.norm(tally-intersection_1)

    def _intersection_on_plane(self, plane:float, v:np.array, p0:list[float])->np.array:
        """Founds an intersection between the vector defined from point 'p0' with direction 'v' and the 
        plane defined by the value 'plane' (the plane is always parallel to two axis, so its)

        Args:
            plane (float): _description_
            v (np.array): _description_
            p0 (list[float]): _description_

        Returns:
            np.array: point of the intersection
        """        

        # equation system
        #
        #  x = x0 + vx*t
        #  y = yo + vy*t
        #  z = z0 + vz*t
        #                            
        #  set y = self.py to find position on plane
        #  t = (y-y0)/vy 

        intersection = np.empty(3)

        t = (plane - p0[self.indices[0]])/v[self.indices[0]]
        y = p0[self.indices[0]] + v[self.indices[0]]*t
        x = p0[self.indices[1]] + v[self.indices[1]]*t
        z = p0[self.indices[2]] + v[self.indices[2]]*t

        intersection[self.indices[0]] = y
        intersection[self.indices[1]] = x
        intersection[self.indices[2]] = z

        return intersection

    def _unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def _angle_between(self, v1, v2):
        v1_u = self._unit_vector(v1)
        v2_u = self._unit_vector(v2)
        radians = round(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)),3)
        return math.degrees(radians)

    def _get_polar_coordinates(self, point):
        theta = np.arccos( point[2]/(self.dist_source_tally)  )
        fi = np.sign(point[1]) * np.arccos( point[0] / (LA.norm(np.array([ point[0], point[1] ]))) )
    
        return math.degrees(theta), math.degrees(fi)




class database_maker:
    """Read a raw MCNP file to extract input/output informations. The aim of this class is to create binary database file. 
    """    
    def __init__(self, inp_adm:input_admin, reader:raw_reader, 
                 mesh_dim:list[int], source:list[float], p0:float):
        """Read a raw MCNP file to extract input/output informations. From the reading process, two variables named 'database_input'
        and 'database_output' are exctracted. The aim of this class is to create binary database file.  

        Args:
            inp_adm (input_admin): a input administrator that calculates relevant values for the standard radiation configuration
            reader (raw_reader): a class that reads raw MCNP results files
            mesh_dim (list[int]): list containing the number of subdivisions on x,y and z of the mesh
            source (list[float]): list containing the x, y, z coordinates [cm] of the source
            p0 (float): plane coordinate value of the wall in the perpendicular axis [see input_admin class]
        """        

        self.mesh_dim = mesh_dim
        self.inp_adm = inp_adm
        self.reader = reader
        self.rad_source = source
        self.py0 = p0


        self.database_input = {}
        self.database_output = []
        self.database_errors = []

# ********************************************************************
#                           Public members
# ********************************************************************

    def read(self, filename:str, inputs:list[str], output:str, n_samples:int=None, max_error:float=None, save_errors:bool=False):
        """Reads a raw MCNP file (named 'filename') and extract the list of desidered inputs and the desired output.
        After the reading process, two variables named 'database_input' and 'database_output' are exctracted.

        Args:
            filename (str): name of the raw MCNP file
            inputs (list[str]): list of the desired inputs for the database
            output (str): desired output for the database
            n_samples (int, optional): a number of n_samples is exctracted; n_samples should be lower than the total
            number of samples. Defaults to None.
            max_error (float, optional): the MCNP values with error>max_error are not considered. Error is given as relative error, 
            spanning from 0 to 1. Defaults to None.
        """    
        self._check_compliance(inputs, output)

        # Get variables from filename
        energy = float(filename.split('_')[2])
        slab_thickness = float(filename.split('_')[1])
        delta_y = float(filename.split('_')[3])
        
        
        self.inp_adm.slab_thickness = slab_thickness 
        self.rad_source[1] = delta_y # y position of the source is the one present in filename

        # Define tallies mesh geometry
        py1 = self.py0 + slab_thickness
        origin = [-800, py1, 165]
        end = [800, 4000 + slab_thickness, 765]
        
        # Read values from binary files containing MCNP results
        self.reader.set_mesh(origin, end, self.mesh_dim)
        self.reader.binary_reader(filename)

        # Collect values
        #self.errors = self.reader.errors
        coord = self.reader.coordinates

        # Filter by errors
        if(max_error!=None):
            self.reader.filter(max_error)

        # Sample a sub-set of the inputs if required
        if n_samples != None:
            # Sample coordinates and their index 
            # if there's a max_error specification, sampling is random
            if (max_error!=None):
                ind_sample = random.sample(range(0, len(self.reader._doses)), n_samples)
            # if there's not a max_error specification, sampling is made trough latin hypercube sampling
            else:
                coord_ind_sample = self._LH_sampling(n_samples)
                ind_sample = self._get_index(coord_ind_sample)

            coord_sample = coord[ind_sample]

        else:
            coord_sample = coord


        # Set the energy and the wall in self.inp_adm
        self._retrieve_case(energy, [self.py0, py1])


        self.database_input = {key:[] for key in inputs}
        
        dose_direct_contribution = []
        inp_coeffs = []

        # Calculate direct contribution to dose and get inputs

        with Pool(5) as p:
            dose_direct_contribution, inp_coeffs = zip(*p.map(self._task, coord_sample.tolist()))

        for i in range(0, len(inp_coeffs)):
            n_key = 0    
            for key in self.database_input.keys():
                self.database_input[key].append( inp_coeffs[i][n_key] )
                n_key += 1

        
        # Set output
        if n_samples != None:
            if output.lower() == 'b':
                self.database_output = self.reader.doses[ind_sample]/np.array(dose_direct_contribution)
            else:
                self.database_output = self.reader.doses[ind_sample]
            if save_errors: 
                self.database_errors = self.reader.errors[ind_sample]
        else:
            if output.lower() == 'b':
                self.database_output = self.reader.doses/np.array(dose_direct_contribution)
            else:
                self.database_output = self.reader.doses
            if save_errors:
                self.database_errors = self.reader.errors

        return
    

    def save_to_binary(self, save_path:str, save_errors:bool):
        """Save the variables 'database_output' and 'database_input' to save_path.
        The variables are saved with this order: output, inputs[0], inputs[1] ...

        Args:
            save_path (str): absolute path were to save the database input/output info
        """        
        lst = [np.array(i[1]) for i in self.database_input.items()]
        if save_errors:
            lst.insert(0, np.array(self.database_errors))
        lst.insert(0, np.array(self.database_output))
        lst = np.concatenate(lst, axis=None)
        buffer = self._binary_converter(lst)
        with open(save_path, 'bw') as f:
            f.write(buffer)
        return

# ********************************************************************
#                           Private members
# ********************************************************************

    def _task(self, t):
        dose = self.inp_adm.calculate_dose_direct_contribution( self.rad_source, t )
        coeff = self.inp_adm.get_coefficients()
        lst_coeffs = []
        for key in self.database_input.keys():
            lst_coeffs.append(coeff[key])
        return dose, lst_coeffs

    def _LH_sampling(self, n_samples:int):
        sampler = qmc.LatinHypercube(d=3)
        low = [0,0,0]
        sample = sampler.integers(l_bounds=low, u_bounds=self.mesh_dim, n=n_samples)
        return sample
    
    def _get_index(self, samples:list[list[int]]):
        return [   (self.mesh_dim[1]*self.mesh_dim[2])*ind_coord[0] + self.mesh_dim[2]*ind_coord[1] + ind_coord[2] for ind_coord in samples ]

    def _retrieve_case(self, energy:float, plane_coords:list[float]):
        self.inp_adm.define_energy(energy)
        self.inp_adm.define_wall_coords(plane_coords)

    def _binary_converter(self,lst):
        lst = lst.tolist()
        return struct.pack('%sf' % len(lst), *lst)
    
    def _check_compliance(self, inputs:list[str], output:str):
        inp_ref = ['energy','slab_thickness', 'dist_source_tally', 'delta_t', 'angle','theta',\
                    'fi', 'dist_source_shield', 'dist_shield_tally', 'mfp']
        out_ref = ['b', 'dose']
        for i in inputs:
            if i not in inp_ref:
                raise Exception(f"The input {i} is not a valid input")

        if output.lower() not in out_ref:
            raise Exception(f"The output {output} is not a valid output")
            
    
