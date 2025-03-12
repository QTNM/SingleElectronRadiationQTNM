import numpy
import matplotlib.pyplot as pyplot
from SingleElectronRadiation import SingleElectronRadiation as SER
import SignalProcessingFunctions as SPF
import SpectrumAnalyser
import pandas as pd
from scipy.optimize import curve_fit
#  scipy import fft
from scipy.signal import find_peaks
from scipy.signal import spectrogram
from bisect import bisect_left
pyplot.rcdefaults()
pyplot.rcParams.update({'font.size': 18})
pyplot.rcParams['agg.path.chunksize'] = 10000
pyplot.rcParams["figure.dpi"] = 100




def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    
    https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511
    """
    print(type(myList))
    print(type(myNumber))
    
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if (after - myNumber) < (myNumber - before):
        return after
    else:
        return before
    
    
    
    
def sineWaveForFit(x, amp, freq, phase):
    valueOut = amp * numpy.sin(2 * numpy.pi * freq * x + phase)
    return valueOut

# Define constants
# Antenna is just a single point here, but can take an array of positions
antennaPosition=numpy.array([0.02,0.0,0]) # metres
antennaAlignment=numpy.array([0,1,0])
pointArea = 2.62894943536*10**-6 # m^2, based on Lijie's notes
pi = numpy.pi
B = 1 # T
vacPerm = 8.8541878128*10**-12	# F/m - Farads per metre - Vacuum permittivity
c = 299792458 # Speed of light in m/s

# Import motion path
# Motion paths are csv-format files with columns of time,x,y,z,vx,vy,vz,ax,ay,az
StudyPrefix = "Boris_89deg_Bathtub_Solution"


particleTrajectory = pd.read_csv("D:/Project Stuff/ParticlePaths/For Thesis/%s.txt" % StudyPrefix, sep=',')

particleTrajectory = particleTrajectory.to_numpy()[:len(particleTrajectory)//1]
# Change this to not have as much processing
# Bear in mind that this will affect the fourier transform's precision


# Optional Shorten Path
# print(particleTrajectory.shape)
# particleTrajectory = particleTrajectory[0:round(particleTrajectory.shape[0]/20),:]


##### Option to restrict electron motion so that it completes a whole number of oscillations and ends where it started #####
# Need to:
# Find where the electron crosses the centre of the trap (z=0)
# Pick a starting point (1st or second crossing)
# Move a whole number of oscillations (ensures no asymmetry in fields experienced, start/end positions, etc.)
# Bear in mind that it is 2 crossings per oscillation
# Keep these and send them to the rest of the processing.

# indicesOfCrossings = numpy.where( numpy.diff( numpy.sign( particleTrajectory[:,3] ) ) )[0] # z axis for crossing the centre of the trap
# print(indicesOfCrossings)
# # Take the start and stop indices from this, accounting for 2 crossings per axial oscillation.
# if len(indicesOfCrossings) > 2:
#     print("Trimming path to include only full oscillations axially")
#     startIndex = indicesOfCrossings[0]
#     if len(indicesOfCrossings) % 2 == 1: # If the number of crossings is odd, then take the last one
#         endIndex = indicesOfCrossings[-1]
#     else:
#         endIndex = indicesOfCrossings[-2] # If the number is even, take the second to last one to ensure complete oscillations
#     particleTrajectory = particleTrajectory[startIndex:endIndex]

# else:
#     print("Fewer than three crossings, take the whole file")




nPoints = particleTrajectory.shape[0]
Times = particleTrajectory[:,0]
EPositions = particleTrajectory[:,1:4]
EVelocities = particleTrajectory[:,4:7]
EAccelerations = particleTrajectory[:,7:]
print("Max z:", max(EPositions[:,2]))

## Check cyclotron radius at start and end
cyclotronRadiusAtStart = numpy.amax(EPositions[0:len(EPositions)//100])
print("Cyclotron radius at start:",numpy.amax(EPositions[0:len(EPositions)//100]))
print("Cyclotron radius at end:",numpy.amax(EPositions[len(EPositions)//100*99:]))
print("Electron position at start:",EPositions[0,0:])

# Get pitch angle assuming x and y B field components are small so all B is in z direction (which is true far from the coils)
pitchAngles = []
for i in range(nPoints):
    pitchAngles.append( numpy.arccos( EVelocities[i,2]/numpy.linalg.norm(EVelocities[i]) ) / pi * 180)
print(pitchAngles[0:6])



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
    powerDensities[i] = SER.CalcPoyntingVectorMagnitude(numpy.linalg.norm(numpy.dot(EFields[i],antennaAlignment))) 
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
FFTFreqs, FFTPowers = SPF.FourierTransformWaveform(Powers, Times)

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
FFTFreqsVoltage, FFTVoltages = SPF.FourierTransformWaveform(voltages, Times)

# Filter voltage test for high frequency background noise
# Centre is at 26,880,361,445.78Hz
centralFrequencyOfSignal = 26880361445.78 # Hz
sampleRate = ((Times[-1]-Times[0])/nPoints)**-1
backgroundBandPass = SPF.ButterBandPass(cutoffLow=centralFrequencyOfSignal-500000,
                                        cutoffHigh=centralFrequencyOfSignal+500000,
                                        order = 3,
                                        sampleRate = sampleRate)
filteredVoltage = backgroundBandPass.ApplyFilter(voltages)
RMSVoltageInSignalRegion = numpy.sqrt( numpy.mean( filteredVoltage**2 ) )
print("RMS filtered Voltage in Signal Region:", RMSVoltageInSignalRegion)
k_B = 1.38064852e-23
temperature = 8 # K   
resistance = 73.2
antennaBandwidth = 1e6
print("Thermal noise RMS:", numpy.sqrt(4*k_B*temperature*resistance*antennaBandwidth))





## Check voltage at start and end
print("Voltage max at start:",numpy.amax(voltages[0:len(voltages)//100]))
print("Voltage min at start:",numpy.amin(voltages[0:len(voltages)//100]))


FFTVoltagePeakIndices = find_peaks(FFTVoltages, height=0.001)[0]
FFTVoltagePeaks = []
for i in range(len(FFTVoltagePeakIndices)):
    FFTVoltagePeaks.append(FFTFreqsVoltage[FFTVoltagePeakIndices[i]])
print(FFTVoltagePeaks)
peakSignal = take_closest(FFTVoltagePeaks, 18.910e9)
print("Peak signal:",peakSignal)
ButterBandPassFilter = SPF.ButterBandPass(peakSignal-0.5e6, peakSignal+0.5e6, order=5, sampleRate=sampleRate)
filteredSignalAroundPeak = ButterBandPassFilter.ApplyFilter(voltages)
print("Filtered Signal RMS Voltage",SPF.GetRMS(filteredSignalAroundPeak[len(voltages)//3:]))





checkTimes = [5e-8,5e-8+2e-10]

checkTimes = [0,4.5e-7]
#checkTimes = [0,2e-10]

# Make plots
figFFT,axFFT = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
axFFT.plot(FFTFreqs,numpy.abs(FFTPowers))
#axFFT.set_xscale("log")
axFFT.set_xlabel("Frequency (Hz)")
axFFT.set_ylabel("Amplitude")
axFFT.set_title("Fourier Transform of Power over Time")
figFFT.savefig("%sFourierPower.png" % StudyPrefix)


figFFTVolt, axFFTVolt = pyplot.subplots(nrows=1, ncols=1, figsize=[16,8])
axFFTVolt.plot(FFTFreqsVoltage, numpy.abs(FFTVoltages))
axFFTVolt.set_xlabel("Frequency (Hz)")
axFFTVolt.set_ylabel("Amplitude")
axFFTVolt.set_title("Fourier Transform of Voltage over Time")
axFFTVolt.set_xlim(-1e9,2e11)
figFFTVolt.savefig("%sFourierVoltage.png" % StudyPrefix)

figFFTVoltLog, axFFTVoltLog = pyplot.subplots(nrows=1, ncols=1, figsize=[16,8])
axFFTVoltLog.plot(FFTFreqsVoltage, numpy.abs(FFTVoltages))
axFFTVoltLog.set_xlabel("Frequency (Hz)")
axFFTVoltLog.set_ylabel("Amplitude")
axFFTVoltLog.set_title("Fourier Transform of Voltage over Time")
axFFTVoltLog.set_xlim(-1e9,2e11)
axFFTVoltLog.set_yscale("log")
figFFTVoltLog.savefig("%sFourierVoltage.png" % StudyPrefix)


figFFTVoltSmall, axFFTVoltSmall = pyplot.subplots(1, 1, figsize=[16,8])
axFFTVoltSmall.plot(FFTFreqsVoltage, numpy.abs(FFTVoltages))
axFFTVoltSmall.set_xlabel("Frequency (Hz)")
axFFTVoltSmall.set_ylabel("Amplitude")
axFFTVoltSmall.set_title("Fourier Transform of Voltage over Time")
axFFTVoltSmall.set_xlim(18.8e9,19.05e9)
figFFTVolt.savefig("%sFourierVoltageSmall.png" % StudyPrefix)




figFFTVoltageSideband, axFFTVoltageSideband = pyplot.subplots(1, 1, figsize=[16,8])
axFFTVoltageSideband.plot(FFTFreqsVoltage, numpy.abs(FFTVoltages))
axFFTVoltageSideband.set_xlabel("Frequency (Hz)")
axFFTVoltageSideband.set_ylabel("Amplitude")
axFFTVoltageSideband.set_title("Fourier Transform of Voltage over Time")
axFFTVoltageSideband.set_xlim(18.90e9,18.905e9)
figFFTVolt.savefig("%sFourierVoltageSmall.png" % StudyPrefix)


figEField,axEField = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
axEField.plot(Times,numpy.linalg.norm(EFields,axis=1))
axEField.set_xlabel("Time (s)")
axEField.set_ylabel("E Field Magnitude (V/m)")
axEField.set_xlim(checkTimes[0],checkTimes[1])
figEField.savefig("%sEFieldGraph.png" % StudyPrefix)
axEField.set_xlim(0,3e-10)


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
axVoltage.plot(Times, -1*voltages)
axVoltage.set_xlabel("Time (s)")
axVoltage.set_ylabel("Voltage (V)")
axVoltage.set_xlim(checkTimes[0],checkTimes[1])

figVoltagePower, axVoltagePower = pyplot.subplots(1, 1, figsize=[18,8])
axVoltagePower.plot(Times[:], voltages[:]**2 *1e15 / (4*73.2))
axVoltagePower.set_ylabel("Power? (fW)?")


xAmplitude = max(EPositions[0:,0])
yAmplitude = max(EPositions[0:,1])
sineWaveFrequency = peakSignal-0.5e6
#sineWaveX = xAmplitude * numpy.sin(2*numpy.pi * sineWaveFrequency * Times)
#sineWaveY = yAmplitude * numpy.cos(2*numpy.pi * sineWaveFrequency * Times)

rangeForFit = slice(0,400)

sineWaveXParams, sineWaveXMatrix = curve_fit(sineWaveForFit, Times[rangeForFit], EPositions[rangeForFit,0], p0=[max(EPositions[rangeForFit,0]), sineWaveFrequency, 0])

sineWaveYParams, sineWaveYMatrix = curve_fit(sineWaveForFit, Times[rangeForFit], EPositions[rangeForFit,1], p0=[max(EPositions[rangeForFit,0]), sineWaveFrequency, 3*numpy.pi/2])

rangeForFit = slice(0,-1)

sineWaveX = sineWaveXParams[0] * numpy.sin(2*numpy.pi * sineWaveXParams[1] * Times[rangeForFit] + sineWaveXParams[2])
sineWaveY = sineWaveYParams[0] * numpy.sin(2*numpy.pi * sineWaveYParams[1] * Times[rangeForFit] + sineWaveYParams[2])


sineWaveZ = numpy.zeros(len(Times[rangeForFit]))
residualsX = EPositions[rangeForFit,0] - sineWaveX
residualsY = EPositions[rangeForFit,1] - sineWaveY
residualsZ = EPositions[rangeForFit,2] - sineWaveZ


figXPos, axXPos = pyplot.subplots(nrows=1, ncols=1, figsize = [18,8])
axXPos.plot(Times,EPositions[0:,0], color='red')
axXPos.set_xlabel("Time (s)")
axXPos.set_ylabel("X position (m)")
axXPos.set_xlim(checkTimes[0],checkTimes[1])

figYPos, axYPos = pyplot.subplots(nrows=1, ncols=1, figsize = [18,8])
axYPos.plot(Times,EPositions[0:,1], color='red')
axYPos.set_xlabel("Time (s)")
axYPos.set_ylabel("Y position (m)")
axYPos.set_xlim(checkTimes[0],checkTimes[1])

figZPos, axZPos = pyplot.subplots(nrows=1, ncols=1, figsize = [18,8])
axZPos.plot(Times,EPositions[0:,2], color='red')
axZPos.set_xlabel("Time (s)")
axZPos.set_ylabel("Z position (m)")
axZPos.set_xlim(checkTimes[0],checkTimes[1])



figResiduals, axResiduals = pyplot.subplots(3, 1, figsize=[16,13])
axResiduals[0].plot(Times[rangeForFit], residualsX)
axResiduals[0].set_xlabel("Time (s)")
axResiduals[0].set_ylabel("x Residual (m)")
axResiduals[1].plot(Times[rangeForFit], residualsY)
axResiduals[1].set_xlabel("Time (s)")
axResiduals[1].set_ylabel("y Residual (m)")
axResiduals[2].plot(Times[rangeForFit], residualsZ)
axResiduals[0].set_title("Residuals")
axResiduals[2].set_xlabel("Time (s)")
axResiduals[2].set_ylabel("z Residual (m)")



# Set up bigger residuals plots
figResidualsBig = pyplot.figure(80, figsize=[16,16])

frame1=figResidualsBig.add_axes((.1,.8,.7,.2))
#left, bottom, width, height [units are fraction of the image frame]
frame1.plot(Times[rangeForFit],sineWaveX,'b')
frame1.plot(Times[rangeForFit],EPositions[rangeForFit,0],'r')
frame1.set_xticklabels([]) #Remove x-tic labels for the first frame

#Residual plot
frame2=figResidualsBig.add_axes((.1,.7,.7,.1))        
frame2.plot(Times[rangeForFit],residualsX*10**12,'b')
frame2.set_xlabel("Time (s)")
frame1.set_ylabel("x (m)")
frame2.set_ylabel(r"$x_{res}$ ($\times 10^{-12}$m)")


frame3 = figResidualsBig.add_axes((.1,.45,.7,.2))
frame3.plot(Times[rangeForFit],sineWaveY,'b')
frame3.plot(Times[rangeForFit],EPositions[rangeForFit,1],'r')
frame3.set_xticklabels([]) #Remove x-tic labels for the first frame

frame4 = figResidualsBig.add_axes((.1,.35,.7,.1))  
frame4.plot(Times[rangeForFit],residualsY*10**7,'b')
frame4.set_xlabel("Time (s)")
frame3.set_ylabel("y (m)")
frame4.set_ylabel(r"$y_{res}$ ($\times 10^{-7}$m)")



frame5 = figResidualsBig.add_axes((.1,.1,.7,.2))
frame5.plot(Times[rangeForFit],sineWaveZ,'b')
frame5.plot(Times[rangeForFit],EPositions[rangeForFit,2]*10**18,'r')
#frame5.set_xticklabels([]) #Remove x-tic labels for the first frame
#frame6 = figResidualsBig.add_axes((.1,.1,.7,.1))  
#frame6.plot(Times[rangeForFit],residualsZ,'b')
frame5.set_xlabel("Time (s)")
frame5.set_ylabel(r"z ($\times 10^{-18}$m)")
#frame6.set_ylabel(r"$z_{res}$ (m)")




# figXVel, axXVel = pyplot.subplots(nrows=1, ncols=1, figsize = [18,8])
# axXVel.plot(Times,EVelocities[0:,0], color='red')
# axXVel.set_ylabel("XVel")
# axXVel.set_xlim(checkTimes[0],checkTimes[1])
# #print(numpy.mean(EVelocities[0:,0]))

# figYVel, axYVel = pyplot.subplots(nrows=1, ncols=1, figsize = [18,8])
# axYVel.plot(Times,EVelocities[0:,1], color='red')
# axYVel.set_ylabel("YVel")
# axXVel.set_xlim(checkTimes[0],checkTimes[1])

# figZVel, axZVel = pyplot.subplots(nrows=1, ncols=1, figsize = [18,8])
# axZVel.plot(Times,EVelocities[0:,2], color='red')
# axZVel.set_ylabel("ZVel")
# axXVel.set_xlim(checkTimes[0],checkTimes[1])


# figVelocities,axVelocities = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
# axVelocities.plot(Times,numpy.linalg.norm(EVelocities,axis=1), color='red')
# axVelocities.set_xlabel("Time (s)")
# axVelocities.set_ylabel("Velocities (m/s)")
# axVelocities.set_title("Velocities ")
# figVelocities.savefig("%sVelocities.png" % StudyPrefix)

# figAccelerations,axAccelerations = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
# axAccelerations.plot(Times,numpy.linalg.norm(EAccelerations,axis=1), color='red')
# axAccelerations.set_xlabel("Time (s)")
# axAccelerations.set_ylabel("Accelerations (m/s)")
# axAccelerations.set_title("Accelerations ")
# figAccelerations.savefig("%sAccelerations.png" % StudyPrefix)

# figAccelerationsSinAngle,axAccelerationsSinAngle = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
# axAccelerationsSinAngle.plot(Times,numpy.linalg.norm(EAccelerations,axis=1)*numpy.sin(dipoleToEmissionAngles)*numpy.sin(dipoleToEmissionAngles), color='red')
# axAccelerationsSinAngle.set_xlabel("Time (s)")
# axAccelerationsSinAngle.set_ylabel("Accelerations (m/s)")
# axAccelerationsSinAngle.set_title("Accelerations ")
# figAccelerationsSinAngle.savefig("%sAccelerationsSinAngle.png" % StudyPrefix)

# figPitchAngle, axPitchAngle = pyplot.subplots(1, 1, figsize=[16,8])
# axPitchAngle.plot(Times, pitchAngles)
# axPitchAngle.set_xlabel("Time (s)")
# axPitchAngle.set_ylabel("Pitch Angle (deg)")





# figPowerDensShort,axPowerDensShort = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
# axPowerDensShort.plot(Times[0:200],powerDensities[0:200])
# axPowerDensShort.set_xlabel("Time (s)")
# axPowerDensShort.set_ylabel("Power Density (W/m)")
# figPowerDensShort.savefig("%sPowerDensityShort.png" % StudyPrefix)


# figFFTSmall,axFFTSmall = pyplot.subplots(nrows=1,ncols=1,figsize=[18,8])
# axFFTSmall.plot(FFTFreqs,numpy.abs(FFTPowers))
# axFFTSmall.set_xlabel("Frequency (Hz)")
# axFFTSmall.set_ylabel("Amplitude")
# axFFTSmall.set_xlim(26.99e9,27.01e9)
# axFFTSmall.set_title("FFT Zoomed In")
# figFFTSmall.savefig("%sFourierPowerSmall.png" % StudyPrefix)
# print("Overall FFT Step size:", FFTFreqs[1]-FFTFreqs[0])
# print("Overall FFT Step size check:", FFTFreqs[len(FFTFreqs)//2]-FFTFreqs[len(FFTFreqs)//2-1])



# figFFTTimeDependent,axFFTTimeDependent = pyplot.subplots(nrows=1,ncols=1,figsize = [10,8])
# axFFTTimeDependent.pcolormesh(spectrumAnalyserOutput[0],spectrumAnalyserOutput[1], numpy.transpose(spectrumAnalyserOutput[2]))
# axFFTTimeDependent.set_xlabel("Time (s)")
# axFFTTimeDependent.set_ylabel("Frequency (Hz)")
# # axFFTTimeDependent.scatter(TimesBoundaries,FFTFreqsSplit)
# #for xe, ye in zip(TimesBoundaries, FFTFreqsSplit):
# #    axFFTTimeDependent.scatter([xe] * len(ye), ye)

# samplingRate = (Times[len(Times)//2]-Times[len(Times)//2-1])**-1
# print("Sample rate:", samplingRate)
# spectrogram = spectrogram(Powers, samplingRate, window=("tukey",0.25), nperseg = len(Times)//10, noverlap=0)
# print(spectrogram)
# figSpectrogram, axSpectrogram = pyplot.subplots(nrows=1, ncols=1, figsize = [10,8])
# axSpectrogram.pcolormesh(spectrogram[1], spectrogram[0], spectrogram[2], shading='auto')
# axSpectrogram.set_ylabel('Frequency [Hz]')
# axSpectrogram.set_xlabel('Time [sec]')
# #axSpectrogram.set_ylim(27.0e9,27.02e9)
# #axSpectrogram.set_ylim(26e9,28e9)






