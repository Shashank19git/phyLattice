"""
   
   phyLattice is a python package to impliment Hybrid Monte Carlo Simulation for Lattice gauge theories.
   for now "xy-model(2D)" and "QED(2D,3D,4D)" are implimented.
   This package uses pythorch tensors operation for computation.
   
   -> xy_package: impliments 2D xy-model with nearest neighbour interection. 
      Provides functions to plot energy, autocorrelation with number of updates 
      and functions to calculate Specific heat. Also one can generate configuration and save them in a folder. 
      

   ->QED_package: impliments Quantum Electrodynamics on 2D, 3D and 4D Lattice. To give freedom to user seperate functions are there to compute action, dirac operato and its derivative,
     inverse of a symmetric positive definet matrix, leapfrog updates and Hybrid Monte Carlo updates. 
"""