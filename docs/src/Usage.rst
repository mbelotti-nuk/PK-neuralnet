=====
Usage
=====

The pkdnn package has 4 functions to be used:
    * Database creation
    * Pkdnn training
    * Pkdnn prediction 
    * Export to NACARTE

Each function take as an input a '.yaml' file. An example of a '.yaml' file is available
for each function in the ``examples\`` folder present at the root of the package.


Database creation
+++++++++++++++++
   
   This module is used to convert raw MCNP meshtal files to database files that
   can be used to train a pkdnn.

.. toctree::
   :maxdepth: 2
   :caption: Parameters:  

   tables/inp_proc_example


.. code::

    >>> databasepknn.exe --config path\to\config\file


Pkdnn training
++++++++++++++

    This module is used to train a pkdnn from a list of database files.

.. toctree::
   :maxdepth: 2
   :caption: Parameters:

   tables/train_example

.. code::

    >>> trainpkdnn.exe --config path\to\config\file


Pkdnn prediction 
++++++++++++++++

   This module is used to make predicitons with a trained pkdnn from a database file.

.. toctree::
   :maxdepth: 2
   :caption: Parameters:

   tables/predict_example

.. code::

    >>> predictpkdnn.exe --config path\to\config\file



Export to NACARTE
+++++++++++++++++

   This module is used to export pkdnn in Pytorch format to a format compatible with NACARTE

.. toctree::
   :maxdepth: 2
   :caption: Parameters:

   tables/export_example


.. code::
     
    >>> savepknn.exe --config path\to\config\file
