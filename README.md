## Single Electron Radiation and Analysis Code
### SingleElectronRadiation.py
Contains functions to calculate the electric field and other related quantities from a single electron given its motion.  It can calculate electric fields in the near and far field both non-relativistically and relativistically.  Powers and power densities can be calculated for a range of input points and the total power ca
Takes electron motion paths in the form
[time, x, y, z, vx, vy, vz, ax, ay, az].

### InhomogeneityAnalysis.py
Uses SingleElectronRadiation.py to calculate the relativistic electric field (near+far), power density, power and integrated power on a point or an array of points representing, for example, an antenna.
