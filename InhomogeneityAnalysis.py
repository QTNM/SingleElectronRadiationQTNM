import numpy
import matplotlib.pyplot as pyplot
from SingleElectronRadiation import SingleElectronRadiation as SER
import pandas as pd
from scipy import fft
from scipy.signal import find_peaks
#pyplot.rcdefaults()
pyplot.rcParams.update({'font.size': 18})


StudyPrefix = "Bathtub"

# Define constants
# Antenna is just a single point here, but can take an array of positions
antennaPosition=numpy.array([0.05,0,0]) # metres
antennaAlignment=numpy.array([0,1,0])
pointArea = 2.62894943536*10**-6 # m^2, based on Lijie's notes
pi = numpy.pi
B = 1 # T
vacPerm = 8.8541878128*10**-12	# F/m - Farads per metre - Vacuum permittivity
c = 299792458 # Speed of light in m/s


# Import motion path
# Motion paths are csv-format files with columns of time,x,y,z,vx,vy,vz,ax,ay,az
particleTrajectory = pd.read_csv("ParticlePaths//BathtubFieldSolutionWAccel.txt",sep=',')
particleTrajectory = particleTrajectory.to_numpy()
nPoints = particleTrajectory.shape[0]
Times = particleTrajectory[:,0]
EPositions = particleTrajectory[:,1:4]
EVelocities = particleTrajectory[:,4:7]
EAccelerations = particleTrajectory[:,7:]
# antennaArray = numpy.tile(antennaPosition(nPoints,1)) # Repeats the antenna position in an array equal to the number of points, for convenience with later functions


# Calculate Field/Power density/Power
EFields = numpy.zeros((nPoints,3))
powerDensities = numpy.zeros(nPoints)
Powers = numpy.zeros(nPoints)
integratedPowers = numpy.zeros(nPoints)

wavelength = 299792458 / 27990000000 
# This is an approximation, the frequency comes from a previous run with no antenna power factor
# In reality the wavelength changes as the power changes, not sure how this can be known before knowing the power
# To get the wavelength you'd need the frequency, from Fourier transform which we only get after the power...

for i in range(nPoints):
    EFields[i] = SER.CalcRelFarEField(antennaPosition, Times[i], EPositions[i], EVelocities[i], EAccelerations[i])[0] + SER.CalcRelNearEField(antennaPosition, Times[i], EPositions[i], EVelocities[i], EAccelerations[i])[0]
    powerDensities[i] = SER.CalcPoyntingVectorMagnitude(numpy.linalg.norm(EFields[i]))
    cosDipoleToEmissionAngle = numpy.dot(EPositions[i]-antennaPosition,antennaAlignment)  \
    / ( numpy.linalg.norm(EPositions[i]-antennaPosition)*numpy.linalg.norm(antennaAlignment) )
    dipoleToEmissionAngle = numpy.arccos(cosDipoleToEmissionAngle)
    
    hertzianDipoleEffectiveArea = SER.HertzianDipoleEffectiveArea(wavelength,dipoleToEmissionAngle)

    Powers[i] = powerDensities[i]*hertzianDipoleEffectiveArea
    
# powerDensities,Powers,integratedPowers = SER.CalcIncidentPowerAntenna([antennaPoint],pointArea,Times,EPositions,EVelocities,EAccelerations)
integratedPowers = Powers

# Do FFT 
normalisedPowers = integratedPowers
FFTPowers = fft.rfft(normalisedPowers)
SampleSpacing = (Times.max()-Times.min())/nPoints
FFTFreqs = fft.rfftfreq(nPoints,d=SampleSpacing)
FFTPowers = numpy.abs(FFTPowers)
minPeakHeight = 0.01*10**-14 # Can choose a minimum height for the fft peaks to save in file
significantmaxima = find_peaks(FFTPowers,height = minPeakHeight)
print(significantmaxima[0])

FFTMaxima = open("%sFFTMaxima.txt" % StudyPrefix,"w")
for i in significantmaxima[0]:
    FFTMaxima.write(str(FFTFreqs[i]))
    FFTMaxima.write(",")
FFTMaxima.close()


# Make plots
figFFT,axFFT = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
axFFT.plot(FFTFreqs,numpy.abs(FFTPowers))
#axFFT.set_xscale("log")
axFFT.set_xlabel("Frequency (Hz)")
axFFT.set_ylabel("Amplitude")
figFFT.savefig("%sFourierPower.png" % StudyPrefix)

figEField,axEField = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
axEField.plot(Times,numpy.linalg.norm(EFields,axis=1))
axEField.set_xlabel("Time (s)")
axEField.set_ylabel("E Field Magnitude (V/m)")
figEField.savefig("%sEFieldGraph.png" % StudyPrefix)

figPowerDens,axPowerDens = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
axPowerDens.plot(Times,powerDensities)
axPowerDens.set_xlabel("Time (s)")
axPowerDens.set_ylabel("Power Density (W/m)")
figPowerDens.savefig("%sPowerDensity.png" % StudyPrefix)

figPower,axPower = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
axPower.plot(Times,Powers)
axPower.set_xlabel("Time (s)")
axPower.set_ylabel("Power (W)")
figPower.savefig("%sPower.png" % StudyPrefix)

figPowerDensShort,axPowerDensShort = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
axPowerDensShort.plot(Times[0:200],powerDensities[0:200])
axPowerDensShort.set_xlabel("Time (s)")
axPowerDensShort.set_ylabel("Power Density (W/m)")
figPowerDensShort.savefig("%sPowerDensityShort.png" % StudyPrefix)


figFFTSmall,axFFTSmall = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
axFFTSmall.plot(FFTFreqs,numpy.abs(FFTPowers))
axFFTSmall.set_xlabel("Frequency (Hz)")
axFFTSmall.set_ylabel("Amplitude")
axFFTSmall.set_xlim(0.26e11,0.29e11)
figFFTSmall.savefig("%sFourierPowerSmall.png" % StudyPrefix)