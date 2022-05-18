import numpy
import SpectrumAnalyser
import matplotlib.pyplot as pyplot
import SignalProcessingFunctions as SPF
import SingleElectronRadiation.SingleElectronRadiation as SER
import pandas as pd
import sys

import VelocityCalculationFrequencyGradient as VCFG
import LarmorPowerFrequencyCalculation as LPFC
#pyplot.rcdefaults()
pyplot.rcParams.update({'font.size': 18})




def GenerateAnalyticCyclotronMotion(BFieldMagnitude, plotGraphs = False):    
    # Simulate a 90 degree electron with its power loss analytically and use that signal for radiation.
    powerLoss = 0
    initialZPos = 0 
    initialEnergy = 18600*SER.ECharge
    antennaPosition = numpy.array([0.02,0,0]) # metres
    antennaAlignment = numpy.array([0,1,0])
    initialFrequency = VCFG.CalculateFrequency(initialEnergy/SER.ECharge, BFieldMagnitude)
    wavelength = 299792458 / initialFrequency
    downmixingFrequency = 27.00000e9
    downmixPhase = numpy.pi*0.5    
    
    signalVoltage = []
    eKEs = []
    electronXPos = []
    electronYPos = []
    electronZPos = []
    powers = numpy.zeros(len(times))
    angFreqs = []
    
    for i in range(len(times)):
        eKE = 18600*SER.ECharge-powerLoss
        eKEs.append(eKE)        
        angFreq = SER.CalcCyclotronFreqWithPowerLoss(powerLoss, B)
        angFreqs.append(angFreq)
        cyclotronAmplitude = SER.CalcCyclotronAmpWithPowerLoss(powerLoss, B)
        EPosition, EVelocity, EAcceleration = SER.CalcElectronCircularMotion(times[i], angFreq, cyclotronAmplitude, initialZPos)
        
        EField = SER.CalcRelFarEField(antennaPosition, times[i], EPosition, EVelocity, EAcceleration)[0]  + SER.CalcRelNearEField(antennaPosition, times[i], EPosition, EVelocity, EAcceleration)[0]
        vectorEffectiveLength = numpy.dot(antennaAlignment, SER.HalfWaveDipoleEffectiveLength(wavelength))
        highFrequencyVoltage = numpy.dot(vectorEffectiveLength, EField)
        downmixingSin = numpy.sin(2*numpy.pi * times[i] * downmixingFrequency + downmixPhase)
        lowFrequencyVoltage = highFrequencyVoltage * downmixingSin
        signalVoltage.append( lowFrequencyVoltage )
        
        larmorPower = SER.CalcLarmorPower(EAcceleration)
        powerLoss += larmorPower * timeStep
        
        electronXPos.append(EPosition[0])
        electronYPos.append(EPosition[1])
        electronZPos.append(EPosition[2])

        wavelength = 299792458 / VCFG.CalculateFrequency(eKE, BFieldMagnitude)
        
        powerDensity = SER.CalcPoyntingVectorMagnitude(numpy.linalg.norm(numpy.dot(EField,antennaAlignment)))
        cosDipoleToEmissionAngle = numpy.dot(EPosition-antennaPosition,antennaAlignment)  \
                                          / ( numpy.linalg.norm(EPosition-antennaPosition)*numpy.linalg.norm(antennaAlignment) )
        dipoleToEmissionAngle = numpy.arccos(cosDipoleToEmissionAngle)
        halfWaveDipoleEffectiveArea = SER.HalfWaveDipoleEffectiveArea(wavelength,dipoleToEmissionAngle)
        powers[i] = powerDensity*halfWaveDipoleEffectiveArea
        
    signalVoltage = numpy.array(signalVoltage)    
    
    if plotGraphs == True:
        figEXPos, axEXPos = pyplot.subplots(3, 1, figsize=[18,24])
        axEXPos[0].plot(times, electronXPos)
        axEXPos[0].set_xlabel("time (s)")
        axEXPos[0].set_ylabel("x (m)")
        axEXPos[1].plot(times, electronYPos)
        axEXPos[1].set_xlabel("time (s)")
        axEXPos[1].set_ylabel("y (m)")
        axEXPos[2].plot(times, electronZPos)
        axEXPos[2].set_xlabel("time (s)")
        axEXPos[2].set_ylabel("z (m)")
        
        figAngFreq, axAngFreq = pyplot.subplots(1, 1, figsize=[18,8])
        axAngFreq.plot(times, numpy.array(angFreqs)/(2*numpy.pi))
        axAngFreq.set_xlabel("time (s)")
        axAngFreq.set_ylabel("Frequency (Hz)")
        
        figeKE, axeKE = pyplot.subplots(1, 1, figsize=[18,8])
        axeKE.plot(times, numpy.array(eKEs)/SER.ECharge)
        axeKE.set_xlabel("time (s)")
        axeKE.set_ylabel("Electron KE (eV)")
        
        figCleanVoltage, axCleanVoltage = pyplot.subplots(1, 1, figsize=[18,8])
        axCleanVoltage.plot(times,signalVoltage)
        axCleanVoltage.set_xlabel("time (s)")
        axCleanVoltage.set_ylabel("Voltage (V)")
    
    print("Initial KE:",eKEs[0]/SER.ECharge)
    print("Final KE:", eKEs[-1]/SER.ECharge)
    print("Max time:", maxTime)
    print("B:",B)
    frequencyGradient = VCFG.CalculateFrequencyGradient(initialKEeV = eKEs[0]/SER.ECharge,
                                                        finalKEeV = eKEs[-1]/SER.ECharge,
                                                        totalTime = maxTime,
                                                        B = B)
    print("\nFrequency Gradient:",frequencyGradient,"\n")
    return signalVoltage, eKEs, electronXPos, electronYPos, electronZPos, powers, angFreqs, frequencyGradient




def GeneratePlainSine(times, amplitude, baseFrequency):
    plainSin = amplitude*numpy.sin(2*numpy.pi*baseFrequency*times)
    return plainSin

def GenerateChirpSine(times, amplitude, baseFrequency, chirpNewFreq):
    chirpRate = (chirpNewFreq-baseFrequency)/(times[-1]-times[0])
    chirpFreqOverTime = chirpRate/2 * times + baseFrequency 
    chirpSinVoltage = amplitude*numpy.sin(2*numpy.pi*chirpFreqOverTime*times)   
    return chirpSinVoltage


def AddEmptyPaddingToSignal(times, signal):
    preSignalTimes = numpy.arange(-1e-4, 0, step=timeStep)
    postSignalTimes = numpy.arange(maxTime, maxTime+1e-4, step=timeStep)
    emptyPreSignal = numpy.zeros(len(preSignalTimes))
    emptyPostSignal = numpy.zeros(len(postSignalTimes))
    times = numpy.insert(times, 0, preSignalTimes)
    times = numpy.append(times, postSignalTimes)
    signal = numpy.insert(signal, 0, emptyPreSignal)
    signal = numpy.append(signal, emptyPostSignal)
    return signal


def ApplyFFTAnalyser(times, signal, lowSearchFreq, highSearchFreq, nTimeSplits, nFreqBins):
    ### FFT Analyser Testing ###
    spectrumAnalyserOutput = SpectrumAnalyser.FFTAnalyser(signal, times, lowSearchFreq, highSearchFreq, nTimeSplits=nTimeSplits, nFreqBins=nFreqBins)
    zFFT = spectrumAnalyserOutput[2]
    
    figFFTTimeDependent,axFFTTimeDependent = pyplot.subplots(nrows=1,ncols=1,figsize = [12,8])
    im = axFFTTimeDependent.pcolormesh(spectrumAnalyserOutput[0],spectrumAnalyserOutput[1], numpy.transpose(zFFT))
    axFFTTimeDependent.set_xlabel("Time (s)")
    axFFTTimeDependent.set_ylabel("Frequency (Hz)")
    colorbar = figFFTTimeDependent.colorbar(im, ax=axFFTTimeDependent)
    colorbar.ax.set_ylabel("Fourier Transform Magnitude (V)")
    figFFTTimeDependent.tight_layout()
    return 0




def ApplySweepAnalyser(times, signal, sampleRate, sweepStep, lowSearchFreq, highSearchFreq, intermediateFreq, freqResolution):
    ### Heterodyne Analyser Testing ###
    nSweeps = (highSearchFreq - lowSearchFreq) / sweepStep * nTimeSplits
    sweepTime = len(times) // nSweeps / sampleRate
    HeterodyneTest = SpectrumAnalyser.HeterodyneAnalyser(signal,
                                                         times,
                                                         sampleRate,
                                                         nTimeSplits,
                                                         lowSearchFreq=lowSearchFreq,
                                                         highSearchFreq=highSearchFreq,
                                                         sweepStep=sweepStep,
                                                         sweepTime=sweepTime,
                                                         intermediateFreq=intermediateFreq,
                                                         freqResolution=freqResolution,
                                                         showBandPassResponse=False)
    
    x = HeterodyneTest[0]
    y = HeterodyneTest[1]
    z = HeterodyneTest[2]
    
    figSweepTimeDependent,axSweepTimeDependent = pyplot.subplots(nrows=1,ncols=1,figsize = [12,8])
    im = axSweepTimeDependent.pcolormesh(x, y, numpy.transpose(z), shading='auto')
    axSweepTimeDependent.set_xlabel("Time (s)")
    axSweepTimeDependent.set_ylabel("Frequency (Hz)")
    figSweepTimeDependent.colorbar(im, ax=axSweepTimeDependent)
    figSweepTimeDependent.tight_layout()
    return 0



def ApplyParallelLockInAmplifiers(times, signal, sampleRate, lowSearchFreq, highSearchFreq, lowPassCutoff):
    ### Parallel Lock-In Amplifier Testing ###
    yFrequencyBounds, lockInAmpOutputs = SpectrumAnalyser.parallelLockInAmplifiers(signal,
                                                                                   times,
                                                                                   lowFrequency = lowSearchFreq,
                                                                                   highFrequency = highSearchFreq,
                                                                                   lowPassCutoff = lowPassCutoff,
                                                                                   sampleRate = sampleRate)
    
    figLIATest, axLIATest = pyplot.subplots(1, 1, figsize=[12,8])
    im = axLIATest.pcolormesh(times, yFrequencyBounds, lockInAmpOutputs, shading='auto')
    axLIATest.set_xlabel("Time (s)")
    axLIATest.set_ylabel("Frequency (Hz)")
    figLIATest.colorbar(im, ax=axLIATest)
    figLIATest.tight_layout()



def ApplySingleLIAWithChirp(times, signal, sampleRate, startFrequency, frequencyGradient, lowPassCutoff):
    ### Single LIA chirp reference ###
    SpectrumAnalyser.SequentialSweepingLockInAmplifier(signal = signal,
                                              times = times,
                                              startFrequency = startFrequency,
                                              frequencyGradient = frequencyGradient,
                                              lowPassCutoff = lowPassCutoff,
                                              sampleRate = sampleRate,
                                              showGraphs = True)
    return 0


def ApplyMultipleLIAWithDifferentStartFrequencies(times, signal, sampleRate, actualStartFreq, frequencyGradient):    
    # Try multiple different start frequencies and graph on same plot
    startFrequencies = numpy.linspace(start = actualStartFreq-20000,
                                      stop = actualStartFreq+20000,
                                      num = 500)
    lockInVoltages = []
    for i in range(len(startFrequencies)):
        lockInVoltages.append(SpectrumAnalyser.SweepingLockInAmplifier(signal = signal,
                                                                        times = times,
                                                                        startFrequency = startFrequencies[i],
                                                                        frequencyGradient = frequencyGradient,
                                                                        lowPassCutoff = 0.7e4,
                                                                        sampleRate = sampleRate))
    
    figMultiStartFreq, axMultiStartFreq = pyplot.subplots(1, 1, figsize=[18,8])
    for i in range(len(startFrequencies)):
        axMultiStartFreq.plot(times, lockInVoltages[i], label=str(startFrequencies[i]-actualStartFreq))
    axMultiStartFreq.set_xlabel("Time (s)")
    axMultiStartFreq.set_ylabel("Voltage (V)")
    axMultiStartFreq.legend()
    
    
    means = []
    for i in range(len(lockInVoltages)):
        means.append(numpy.mean(lockInVoltages[i]))
    
    figFrequencyDifferences, axFrequencyDifferences = pyplot.subplots(1, 1, figsize=[18,8])
    axFrequencyDifferences.plot(startFrequencies-actualStartFreq, means)
    axFrequencyDifferences.set_xlabel("Difference from Actual Frequency (Hz)")
    axFrequencyDifferences.set_ylabel("Mean Voltage (V)")
    return 0




def ApplyMultiLIAWithDifferentGradients(times, signal, sampleRate):
    # Try multiple different frequency gradients
    startFrequency = actualStartFreq-7000
    frequencyGradients = numpy.linspace(start = actualFrequencyGradient-2e7,
                                        stop = actualFrequencyGradient+7e7,
                                        num = 51)
    lockInVoltages = []
    for i in range(len(frequencyGradients)):
        lockInVoltages.append(SpectrumAnalyser.SweepingLockInAmplifier(signal = signal,
                                                                        times = times,
                                                                        startFrequency = startFrequency,
                                                                        frequencyGradient = frequencyGradients[i],
                                                                        lowPassCutoff = 0.7e4,
                                                                        sampleRate = sampleRate))
    means = []
    for i in range(len(lockInVoltages)):
        means.append(numpy.mean(lockInVoltages[i]))
    
    figGradientDifferences, axGradientDifferences = pyplot.subplots(1, 1, figsize=[18,8])
    axGradientDifferences.plot(frequencyGradients-actualFrequencyGradient, means)
    axGradientDifferences.set_xlabel("Difference from Actual Gradient (Hz)")
    axGradientDifferences.set_ylabel("Mean Voltage (V)")
    return 0




def CreateTriggerValuesPlot(frequencyGradients, startFrequencies, triggerValues, triggerLabel, triggerMax, triggerMin):
    frequencyGradientLabels = [] 
    startFrequencyLabels = []
        
    for i in range(len(frequencyGradients)):
        frequencyGradientLabels.append( str( '{:.2e}'.format(frequencyGradients[i])))
    for i in range(len(startFrequencies)):
        startFrequencyLabels.append( str( '{:.4e}'.format(startFrequencies[i])))
        
    print(frequencyGradientLabels)
    print(startFrequencyLabels)
    
    figTriggerValues, axTriggerValues = pyplot.subplots(1, 1, figsize=[12,12])
    im = axTriggerValues.imshow(numpy.transpose(triggerValues), origin='lower', cmap='Greys', vmin=triggerMin, vmax=triggerMax) # plot the transpose to keep axes consistent
    axTriggerValues.set_xlabel("Frequency Gradient (Hz/s)")
    axTriggerValues.set_ylabel("Start Frequency (Hz)")
    axTriggerValues.set_xticks( numpy.arange(0, len(frequencyGradients), dtype='int')[::5] )
    axTriggerValues.set_yticks( numpy.arange(0, len(startFrequencies), dtype='int')[::5] )
    axTriggerValues.set_xticklabels(frequencyGradientLabels[::5], rotation=90)
    axTriggerValues.set_yticklabels(startFrequencyLabels[::5])
    colorbar = figTriggerValues.colorbar(im)
    colorbar.ax.set_ylabel(triggerLabel)
    return 0




def SweepReferenceGradientVsStartFreqPlots():
    # Combined start frequency and frequency gradient plot
    startFrequencies = numpy.linspace(start = actualStartFreq-40000,
                                      stop = actualStartFreq+40000,
                                      num = 51)
    frequencyGradients = numpy.linspace(start = 3.46e8,
                                        stop = 3.61e8,
                                        num = 51)
    
    lockInVoltages = numpy.empty( (len(frequencyGradients), len(startFrequencies)), dtype="object" )
    for i in range(len(frequencyGradients)):
        for j in range(len(startFrequencies)):
            lockInVoltages[i,j] = SpectrumAnalyser.SweepingLockInAmplifier(signal = noisySin,
                                                                            times = times,
                                                                            startFrequency = startFrequencies[j],
                                                                            frequencyGradient = frequencyGradients[i],
                                                                            lowPassCutoff = 0.7e4,
                                                                            sampleRate = sampleRate) 
    meanVoltages = numpy.zeros(lockInVoltages.shape)
    for i in range(len(frequencyGradients)):
        for j in range(len(startFrequencies)):
            meanVoltages[i,j] = numpy.mean(lockInVoltages[i,j])
    
    
    triggerLabel = "Mean Voltage (V)"
    triggerMax = numpy.amax(meanVoltages)
    triggerMin = numpy.amin(meanVoltages)
    
    CreateTriggerValuesPlot(frequencyGradients, startFrequencies, meanVoltages, triggerLabel, triggerMax, triggerMin)
    
    # Repeat with just noise
    noiseVoltages = numpy.empty( (len(frequencyGradients), len(startFrequencies)), dtype="object" )
    for i in range(len(frequencyGradients)):
        for j in range(len(startFrequencies)):
            noiseVoltages[i,j] = SpectrumAnalyser.SweepingLockInAmplifier(signal = noise,
                                                                          times = times,
                                                                          startFrequency = startFrequencies[j],
                                                                          frequencyGradient = frequencyGradients[i],
                                                                          lowPassCutoff = 0.7e4,
                                                                          sampleRate = sampleRate) 
    meanNoiseVoltages = numpy.zeros(noiseVoltages.shape)
    for i in range(len(frequencyGradients)):
        for j in range(len(startFrequencies)):
            meanNoiseVoltages[i,j] = numpy.mean(noiseVoltages[i,j])
    
    CreateTriggerValuesPlot(frequencyGradients, startFrequencies, meanNoiseVoltages, triggerLabel, triggerMax, triggerMin)

    return 0


def TrackingFrequencyTest(signal, times):
    startFrequency = actualStartFreq-35800-1e5
    frequencyGradient = actualFrequencyGradient*2
    lowPassCutoff = 0.7e4

    kineticEnergyJ = 18600 * SER.ECharge
    initialCyclotronAngFreq = LPFC.AngularFrequency(kineticEnergyJ, B)
    gamma = LPFC.CalculateGamma(kineticEnergyJ)
    pitchAngleDeg = 89
    perpendicularVelocity = LPFC.CalcVelocity(gamma) * numpy.sin(pitchAngleDeg / 180 * numpy.pi)
    print("Perpendicular velocity:", perpendicularVelocity)
    print("B:", B)
    print("Gamma:", gamma)
    print("Initial cyclotron Freq:", initialCyclotronAngFreq/(2*numpy.pi))
    cyclotronRadius = LPFC.CyclotronRadius(perpendicularVelocity, B, gamma, initialCyclotronAngFreq)
    print("Cyclotron radius:", cyclotronRadius)
    emittedLarmorPower = LPFC.LarmorPower( LPFC.CalcAcceleration(initialCyclotronAngFreq, cyclotronRadius) )
    print("Larmor power:", emittedLarmorPower)
    print(downmixingFrequency)
    downmixedInitialAngularFrequency = initialCyclotronAngFreq - (initialCyclotronAngFreq - 2.358e7*2*numpy.pi) - 3.5e6*2*numpy.pi - actualFrequencyGradient*-1e-3*2*numpy.pi - 9e4*2*numpy.pi
    print("Initial frequency:", downmixedInitialAngularFrequency/(2*numpy.pi))
    
    quadratureMagnitudes, quadraturePhases = SpectrumAnalyser.SweepingLockInAmplifierWithPhaseAdjust(signal,
                                                                                                     times,
                                                                                                     initialCyclotronAngFreq,
                                                                                                     downmixedInitialAngularFrequency,
                                                                                                     emittedLarmorPower,
                                                                                                     gamma,
                                                                                                     lowPassCutoff,
                                                                                                     sampleRate)
    
    figQuadratureMagPhase, axQuadratureMagPhase = pyplot.subplots(2, 1, figsize = [18,16])
    axQuadratureMagPhase[0].plot(times, quadratureMagnitudes)
    axQuadratureMagPhase[1].plot(times, quadraturePhases)
    return 0



def GetSignalVoltageFromMotion(FileName, BFieldMagnitude, plotGraphs = False):
    underSampledMotion = pd.read_csv("D:/Project Stuff/ParticlePaths/%s.txt" % FileName,sep=',', header=None, skiprows=(lambda x: x % 1000)).to_numpy()
    times = underSampledMotion[:,0]
    print("Data length from file:",len(times))    
    
    initialEnergy = 18600*SER.ECharge
    antennaPosition = numpy.array([0.02,0,0]) # metres
    antennaAlignment = numpy.array([0,1,0])
    initialFrequency = VCFG.CalculateFrequency(initialEnergy/SER.ECharge, BFieldMagnitude)
    
    wavelength = 299792458 / initialFrequency
    downmixingFrequency = 27.00000e9
    downmixPhase = numpy.pi*0.5 
    signalVoltage = []
    for i in range(len(times)):
        EPosition = numpy.array([ underSampledMotion[i,1], underSampledMotion[i,2], underSampledMotion[i,3] ])
        EVelocity = numpy.array([ underSampledMotion[i,4], underSampledMotion[i,5], underSampledMotion[i,6] ])
        EAcceleration = numpy.array([ underSampledMotion[i,7], underSampledMotion[i,8], underSampledMotion[i,9] ])
        EField = SER.CalcRelFarEField(antennaPosition, times[i], EPosition, EVelocity, EAcceleration)[0]  + SER.CalcRelNearEField(antennaPosition, times[i], EPosition, EVelocity, EAcceleration)[0]
    
        vectorEffectiveLength = numpy.dot(antennaAlignment, SER.HalfWaveDipoleEffectiveLength(wavelength))
        highFrequencyVoltage = numpy.dot(vectorEffectiveLength, EField)
        downmixingSin = numpy.sin(2*numpy.pi * times[i] * downmixingFrequency + downmixPhase)
        lowFrequencyVoltage = highFrequencyVoltage
        signalVoltage.append( lowFrequencyVoltage )
    
    return times, numpy.array(signalVoltage)
    
def downmix(times, signalVoltage, frequency):
    downmixSin = numpy.sin(times * 2 * numpy.pi * frequency)
    return signalVoltage*downmixSin



if __name__ == "__main__":
    # Initialise Parameters    
    sampleRate = 200e6
    timeStep = 1/sampleRate
    maxTime = 1e-3   # seconds
    times = numpy.arange(0, maxTime, step=timeStep)
    nTimeSplits = 10
    print("Data length:",len(times))

    resistance = 73.2
    B = 0.9951 # Centre of harmonic trap simulated
    
    
    lowSearchFreq = 20e6
    highSearchFreq = 21e6
    
    if lowSearchFreq >= highSearchFreq:
        print("ERROR: low search frequency is greater than or equal to high search frequency")
        sys.exit()

    
    # Create signal from file or load it if it exists
    
    FileName = "Harmonic/89degHarmonicWAccel"
    
    loadSignalFromFile = True
    if loadSignalFromFile==True:
        signalVoltage = pd.read_csv("D:/Project Stuff/ParticlePaths/%s_SignalVoltage.txt" % FileName,sep=',', header=None, skiprows=(lambda x: x % 1000)).to_numpy()
        times = signalVoltage[:,0]
        signalVoltage = signalVoltage[:,1]
        print("Data length from file:",len(times)) 
    else:
        times, signalVoltage = GetSignalVoltageFromMotion(FileName, BFieldMagnitude = B)
        with open("D:/Project Stuff/ParticlePaths/%s_SignalVoltageGen.txt" % FileName,'a') as fileOutput:     
            for i in range(len(times)):   
                fileOutput.write(str(times[i]) + "," + 
                                 str(signalVoltage[i]) + "\n" )
    
    
    # Actual start frequency and frequency gradient determined from looking at signal
    actualStartFreq = 20350000 - 2.1e4
    actualFrequencyGradient = 350000/0.001 + 0.8e7 
    
    # Add some noise to the data 
    k_B = 1.38064852e-23
    temperature = 4 # K
    antennaBandwidth = 1e6
    antennaLowCutoff = 20e6
    kTdB = k_B * temperature * antennaBandwidth 
    print("kTdB:", kTdB)
    
    noise = SPF.GenerateVoltageNoise(sampleRate = sampleRate,
                                     numSamples = len(times),
                                     temp = temperature,
                                     resistance = resistance,
                                     antennaLowCutoff = antennaLowCutoff,
                                     antennaBandwidth = antennaBandwidth)
    
    if len(noise) < len(times):
        signalVoltage = signalVoltage[:len(noise)]
        times = times[:len(noise)]
    elif len(times) < len(noise):
        noise = noise[:len(times)]

    print("Check RMS of signal before and after downmixing")
    RMSBeforeDownmixing = numpy.sqrt(numpy.mean(signalVoltage**2))
    print("Before:", RMSBeforeDownmixing)
    downmixingFrequency = 26.86e9
    signalVoltage = downmix(times, signalVoltage, downmixingFrequency)
    RMSAfterDownmixing = numpy.sqrt(numpy.mean(signalVoltage**2))
    print("After:", RMSAfterDownmixing)
    downmixingRMSFactor = RMSBeforeDownmixing / RMSAfterDownmixing
    signalVoltage *= downmixingRMSFactor
    print("After fix:", numpy.sqrt(numpy.mean(signalVoltage**2)))
    
    # Add additional noise before signal starts
    addExtraNoise = False
    if addExtraNoise == True:
        preSignalNoiseTimes = numpy.arange(-1e-3,0,step = timeStep)
        print("Length of additional times:",len(preSignalNoiseTimes))
        preSignalNoise = SPF.GenerateVoltageNoise(sampleRate = sampleRate,
                                                  numSamples = len(preSignalNoiseTimes),
                                                  temp = temperature,
                                                  resistance = resistance,
                                                  antennaLowCutoff = antennaLowCutoff,
                                                  antennaBandwidth = antennaBandwidth)

        times = numpy.insert(times, 0, preSignalNoiseTimes)
        noise = numpy.insert(noise, 0, preSignalNoise)

        signalZeros = numpy.zeros(len(preSignalNoise))
        signalVoltage = numpy.insert(signalVoltage,0,signalZeros)
    
    
    
    noisySin = signalVoltage + noise

    
    
    print("\nCHECK: Check voltage^2 and power ratios are the same for the noise:")
    print("AntennaV^2/noiseV^2:", numpy.mean(signalVoltage**2)/numpy.mean(noise**2))
 
    print("Noise RMS:", numpy.sqrt(numpy.mean(noise**2)))
    print("Noise mean power:",(numpy.mean(noise**2)/(4*73.2))) 
    print("Signal RMS:", numpy.sqrt(numpy.mean(signalVoltage**2)))
    print("Signal power (RMS):", numpy.sqrt( (numpy.mean( ((signalVoltage**2)/(4*73.2))**2) ) ))
    print("SNR:", numpy.sqrt(numpy.mean(signalVoltage**2)) / numpy.sqrt(numpy.mean(noise**2)))



        
    ApplyFFTAnalyser(times, signalVoltage, lowSearchFreq, highSearchFreq, nTimeSplits, nFreqBins=40)

    # First use a single LIA with chirp frequency but no tracking
    # ApplySingleLIAWithChirp(times,
    #                         noisySin,
    #                         sampleRate,
    #                         startFrequency=actualStartFreq-3e5 - actualFrequencyGradient*-1e-3 - 9e4,
    #                         frequencyGradient=actualFrequencyGradient,
    #                         lowPassCutoff=7e3
    #                         )
    
    # TrackingFrequencyTest(noisySin, times)
    SweepReferenceGradientVsStartFreqPlots()




















