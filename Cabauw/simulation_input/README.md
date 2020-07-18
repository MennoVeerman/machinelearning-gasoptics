# RCEMIP simulation
Input for the Cabauw LES-simulation performed with DALES (*Heus et al. (2010). Formulation of the Dutch Atmospheric Large-Eddy simulation and overview of its applications, GMD.*; original code available at https://github.com/dalesteam/dales). For a description of the case, see e.g. *Pedruzo-Bagazoitia et al. (2017): Direct and Diffuse Radiation in the Shallow Cumulus–Vegetation System: Enhanced and Decreased Evapotranspiration Regimes, J. Hydrometeor.* and *Vila-Guerau de Arellano et al. (2014): Shallow cumulus rooted in photosynthesis, Geophys. Res. Lett.*

Intruction to run the case:
1. Obtain the Dutch Atmospheric Large-Eddy Simulation (DALES) code from the forked repository https://github.com/MennoVeerman/dales/tree/to4.2\_oldpoisson\_scalar\_tenstream and compile it
2. run the simulation, e.g. 
    srun -n 144./dales4 namoptions.000 >output.000









