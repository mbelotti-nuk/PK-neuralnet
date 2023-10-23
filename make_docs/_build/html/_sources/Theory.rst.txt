======
Theory
======

Point Kernel
++++++++++++

 The PK method is an analytical technique that calculates the photon radiation coming from a volume by dividing it in an ensemble of point sources (point kernels). 
 Each point kernel contribute to the dose at a measurement point x by considering: (i) a direct contribution, modelled as a ray, which attenuates exponentially 
 along the thickness (t) of the medium between the point kernel and the point, and (ii) an indirect contribution reaching x due to scattered radiation induced 
 by the medium, modelled by a Build-up factor B(E,t). The overall dose estimation at x is:

.. math:: D(x) = C(E) B(E,t) exp(-\mu t)/(4 \pi r^2).

Where C(E) is the dose to flux conversion factor, r is the distance between the source and the measurement point and \mu  is the attenuation coefficient of the medium.