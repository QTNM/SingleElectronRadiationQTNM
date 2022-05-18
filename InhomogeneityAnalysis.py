import numpy
import matplotlib.pyplot as pyplot
from SingleElectronRadiation import SingleElectronRadiation as SER
import SignalProcessingFunctions as SPF
import SpectrumAnalyser
import pandas as pd
#  scipy import fft
from scipy.signal import find_peaks
from scipy.signal import spectrogram
pyplot.rcdefaults()
pyplot.rcParams.update({'font.size': 18})
pyplot.rcParams['agg.path.chunksize'] = 10000
pyplot.rcParams["figure.dpi"] = 100



# Define constants
# Antenna is just a single point here, but can take an array of positions
antennaPosition=numpy.array([0.02,0,0]) # metres
antennaAlignment=numpy.array([0,1,0])
pointArea = 2.62894943536*10**-6 # m^2, based on Lijie's notes
pi = numpy.pi
B = 1 # T
vacPerm = 8.8541878128*10**-12	# F/m - Farads per metre - Vacuum permittivity
c = 299792458 # Speed of light in m/s

# Import motion path
# Motion paths are csv-format files with columns of time,x,y,z,vx,vy,vz,ax,ay,az
StudyPrefix = "TestTinyTrap"


particleTrajectory = pd.read_csv("D:/Project Stuff/ParticlePaths/%sWAccel.txt" % StudyPrefix, sep=',')

particleTrajectory = particleTrajectory.to_numpy()



# Optional Shorten Path
# print(particleTrajectory.shape)
# particleTrajectory = particleTrajectory[0:round(particleTrajectory.shape[0]/20),:]

nPoints = particleTrajectory.shape[0]
Times = particleTrajectory[:,0]
EPositions = particleTrajectory[:,1:4]
EVelocities = particleTrajectory[:,4:7]
EAccelerations = particleTrajectory[:,7:]






# # Alternative numpy npz file load:
# with numpy.load('ParticlePaths//Flat Background 1.007 Trap//FlatBackgroundCoils_Z0.1_R0.05WAccel.txt') as particleTrajectory:
#     EPositions = particleTrajectory['x']
#     EVelocities = particleTrajectory['v']
  
# print(EPositions)
# print(particleTrajectory)
# cfl = 0.1
# nsteps = int(1e5 / cfl)
# fg = 169704864170.0153 # gyro frequency
# dt = 2.0 * numpy.pi * cfl / fg
# Times = numpy.linspace(0.0, dt*nsteps, nsteps+1)

# nPoints = len(EPositions)
# print(len(EPositions))
# print(len(Times))
# print(max(Times))

# EAccelerations = []
# print("init",EAccelerations)
# for i in range(len(EPositions)-1):
#     timeStep = Times[i+1]-Times[i]
#     initialVelocity = numpy.array( [EVelocities[i,0], EVelocities[i,1], EVelocities[i,2]] )
#     secondVelocity = numpy.array( [EVelocities[i+1,0], EVelocities[i+1,1], EVelocities[i+1,2]] )
#     acceleration = (secondVelocity-initialVelocity)/timeStep
#     if i == 0:
#         print("accel =",acceleration)
#     EAccelerations.append(acceleration)

# EAccelerations = numpy.array(EAccelerations)
# print("length accel:", len(EAccelerations))
# EPositions = EPositions[:len(EAccelerations)]
# EVelocities = EVelocities[:len(EAccelerations)]
# Times = Times[:len(EAccelerations)]
# nPoints-=1







# Calculate Field/Power density/Power
EFields = numpy.zeros((nPoints,3))
powerDensities = numpy.zeros(nPoints)
Powers = numpy.zeros(nPoints)
voltages = numpy.zeros(nPoints)

wavelength = 299792458 / 27.009368404e9


dipoleToEmissionAngles = []

for i in range(nPoints):
    EFields[i] = SER.CalcRelFarEField(antennaPosition, Times[i], EPositions[i], EVelocities[i], EAccelerations[i])[0]  + SER.CalcRelNearEField(antennaPosition, Times[i], EPositions[i], EVelocities[i], EAccelerations[i])[0]
    # EFields[i] = SER.CalcRelNearEField(antennaPosition, Times[i], EPositions[i], EVelocities[i], EAccelerations[i])[0]
    # powerDensities[i] = SER.CalcPoyntingVectorMagnitude(numpy.linalg.norm(EFields[i]))
    powerDensities[i] = SER.CalcPoyntingVectorMagnitude(numpy.linalg.norm(numpy.dot(EFields[i],antennaAlignment))) # I think maybe I missed accounting for the polarisation of the antenna when getting the power densities
    cosDipoleToEmissionAngle = numpy.dot(EPositions[i]-antennaPosition,antennaAlignment)  \
    / ( numpy.linalg.norm(EPositions[i]-antennaPosition)*numpy.linalg.norm(antennaAlignment) )
    dipoleToEmissionAngle = numpy.arccos(cosDipoleToEmissionAngle)
    dipoleToEmissionAngles.append(dipoleToEmissionAngle)
    
    hertzianDipoleEffectiveArea = SER.HertzianDipoleEffectiveArea(wavelength,dipoleToEmissionAngle)
    halfWaveDipoleEffectiveArea = SER.HalfWaveDipoleEffectiveArea(wavelength,dipoleToEmissionAngle)
    Powers[i] = powerDensities[i]*halfWaveDipoleEffectiveArea
    
    # Get the voltage from the E field
    # Radiation resistance of a half wavelength dipole antenna is ~73.2 ohms
    VectorEffectiveLength = numpy.dot(antennaAlignment, SER.HalfWaveDipoleEffectiveLength(wavelength))
    voltages[i] = numpy.dot(VectorEffectiveLength, EFields[i])
    
print("Mean Power:", numpy.mean(Powers))
print("RMS Voltage:", numpy.sqrt(numpy.mean(voltages**2)))
powersAndTimes = numpy.array([Times,Powers]).transpose()
print(powersAndTimes.shape)
numpy.savetxt(StudyPrefix+"Powers.txt",powersAndTimes,delimiter=",")

# Do FFT 
FFTFreqs, FFTPowers = SPF.FourierTransformPower(Powers, Times)

minPeakHeight = 0.01*10**-14 # Can choose a minimum height for the fft peaks to save in file
significantmaxima = find_peaks(FFTPowers,height = minPeakHeight)
print(significantmaxima[0])

FFTMaxima = open("%sFFTMaxima.txt" % StudyPrefix,"w")
for i in significantmaxima[0]:
    FFTMaxima.write(str(FFTFreqs[i]))
    FFTMaxima.write(",")
FFTMaxima.close()

spectrumAnalyserOutput = SpectrumAnalyser.FFTAnalyser(Powers, Times, lowFreqLimit=27.995e9, upperFreqLimit=27.015e9, nTimeSplits=10, nFreqBins=8)

# Voltage FFT for comparison
FFTFreqsVoltage, FFTVoltages = SPF.FourierTransformPower(voltages, Times)


# Make plots
figFFT,axFFT = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
axFFT.plot(FFTFreqs,numpy.abs(FFTPowers))
#axFFT.set_xscale("log")
axFFT.set_xlabel("Frequency (Hz)")
axFFT.set_ylabel("Amplitude")
figFFT.savefig("%sFourierPower.png" % StudyPrefix)


figFFTVolt, axFFTVolt = pyplot.subplots(nrows=1, ncols=1, figsize=[18,8])
axFFTVolt.plot(FFTFreqsVoltage, numpy.abs(FFTVoltages))
axFFTVolt.set_xlabel("Frequency (Hz)")
axFFTVolt.set_ylabel("Amplitude")
figFFTVolt.savefig("%sFourierVoltage.png" % StudyPrefix)




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
axPower.plot(Times,Powers*1e15, color='red')
axPower.set_xlabel("Time (s)")
axPower.set_ylabel("Power (fW)")
axPower.set_title("Power At Half Wavelength Dipole")
figPower.savefig("%sPower.png" % StudyPrefix)

figVoltage, axVoltage = pyplot.subplots(1, 1, figsize=[18,8])
axVoltage.plot(Times, voltages)
axVoltage.set_xlabel("Time (s)")
axVoltage.set_ylabel("Voltage (V)")

figVoltagePower, axVoltagePower = pyplot.subplots(1, 1, figsize=[18,8])
axVoltagePower.plot(Times[:], voltages[:]**2 *1e15 / (4*73.2))
axVoltagePower.set_ylabel("Power? (fW)?")



figXPos, axXPos = pyplot.subplots(nrows=1, ncols=1, figsize = [18,8])
axXPos.plot(Times,EPositions[0:,0], color='red')
axXPos.set_ylabel("XPos")

figYPos, axYPos = pyplot.subplots(nrows=1, ncols=1, figsize = [18,8])
axYPos.plot(Times,EPositions[0:,1], color='red')
axYPos.set_ylabel("YPos")

figZPos, axZPos = pyplot.subplots(nrows=1, ncols=1, figsize = [18,8])
axZPos.plot(Times,EPositions[0:,2], color='red')
axZPos.set_ylabel("ZPos")




figXVel, axXVel = pyplot.subplots(nrows=1, ncols=1, figsize = [18,8])
axXVel.plot(Times,EVelocities[0:,0], color='red')
axXVel.set_ylabel("XVel")
print(numpy.mean(EVelocities[0:,0]))

figYVel, axYVel = pyplot.subplots(nrows=1, ncols=1, figsize = [18,8])
axYVel.plot(Times,EVelocities[0:,1], color='red')
axYVel.set_ylabel("YVel")

figZVel, axZVel = pyplot.subplots(nrows=1, ncols=1, figsize = [18,8])
axZVel.plot(Times,EVelocities[0:,2], color='red')
axZVel.set_ylabel("ZVel")


figVelocities,axVelocities = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
axVelocities.plot(Times,numpy.linalg.norm(EVelocities,axis=1), color='red')
axVelocities.set_xlabel("Time (s)")
axVelocities.set_ylabel("Velocities (m/s)")
axVelocities.set_title("Velocities ")
figVelocities.savefig("%sVelocities.png" % StudyPrefix)

figAccelerations,axAccelerations = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
axAccelerations.plot(Times,numpy.linalg.norm(EAccelerations,axis=1), color='red')
axAccelerations.set_xlabel("Time (s)")
axAccelerations.set_ylabel("Accelerations (m/s)")
axAccelerations.set_title("Accelerations ")
figAccelerations.savefig("%sAccelerations.png" % StudyPrefix)

figAccelerationsSinAngle,axAccelerationsSinAngle = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
axAccelerationsSinAngle.plot(Times,numpy.linalg.norm(EAccelerations,axis=1)*numpy.sin(dipoleToEmissionAngles)*numpy.sin(dipoleToEmissionAngles), color='red')
axAccelerationsSinAngle.set_xlabel("Time (s)")
axAccelerationsSinAngle.set_ylabel("Accelerations (m/s)")
axAccelerationsSinAngle.set_title("Accelerations ")
figAccelerationsSinAngle.savefig("%sAccelerationsSinAngle.png" % StudyPrefix)







figPowerDensShort,axPowerDensShort = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
axPowerDensShort.plot(Times[0:200],powerDensities[0:200])
axPowerDensShort.set_xlabel("Time (s)")
axPowerDensShort.set_ylabel("Power Density (W/m)")
figPowerDensShort.savefig("%sPowerDensityShort.png" % StudyPrefix)


figFFTSmall,axFFTSmall = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
axFFTSmall.plot(FFTFreqs,numpy.abs(FFTPowers))
axFFTSmall.set_xlabel("Frequency (Hz)")
axFFTSmall.set_ylabel("Amplitude")
axFFTSmall.set_xlim(26.9e9,27.2e9)
figFFTSmall.savefig("%sFourierPowerSmall.png" % StudyPrefix)
print("Overall FFT Step size:", FFTFreqs[1]-FFTFreqs[0])
print("Overall FFT Step size check:", FFTFreqs[len(FFTFreqs)//2]-FFTFreqs[len(FFTFreqs)//2-1])



figFFTTimeDependent,axFFTTimeDependent = pyplot.subplots(nrows=1,ncols=1,figsize = [10,8])
axFFTTimeDependent.pcolormesh(spectrumAnalyserOutput[0],spectrumAnalyserOutput[1], numpy.transpose(spectrumAnalyserOutput[2]))
axFFTTimeDependent.set_xlabel("Time (s)")
axFFTTimeDependent.set_ylabel("Frequency (Hz)")
# axFFTTimeDependent.scatter(TimesBoundaries,FFTFreqsSplit)
#for xe, ye in zip(TimesBoundaries, FFTFreqsSplit):
#    axFFTTimeDependent.scatter([xe] * len(ye), ye)

samplingRate = (Times[len(Times)//2]-Times[len(Times)//2-1])**-1
print("Sample rate:", samplingRate)
spectrogram = spectrogram(Powers, samplingRate, window=("tukey",0.25), nperseg = len(Times)//10, noverlap=0)
print(spectrogram)
figSpectrogram, axSpectrogram = pyplot.subplots(nrows=1, ncols=1, figsize = [10,8])
axSpectrogram.pcolormesh(spectrogram[1], spectrogram[0], spectrogram[2], shading='auto')
axSpectrogram.set_ylabel('Frequency [Hz]')
axSpectrogram.set_xlabel('Time [sec]')
axSpectrogram.set_ylim(27.0e9,27.02e9)
#axSpectrogram.set_ylim(26e9,28e9)






