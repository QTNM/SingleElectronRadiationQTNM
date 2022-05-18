import SignalProcessingFunctions as SPF
import numpy
import matplotlib.pyplot as pyplot
from scipy import signal as ScipySignal
import LockInAmplifierObjectOriented as LIA
# from memory_profiler import profile

eMass = 9.1093837015e-31 # kg

def FFTAnalyser(Powers, Times, lowFreqLimit, upperFreqLimit, nTimeSplits, nFreqBins):
    """Perform an FFT on signal (Powers) and create a spectrogram formed of FFTs of each time split."""
    print("\n    FFT Analyser")
    # Make frequency/time/amplitude graph
    nSplits = nTimeSplits
    PowersSplit = numpy.array_split(Powers,nSplits)
    TimesSplit = numpy.array_split(Times,nSplits)
    FFTFreqsSplit = numpy.zeros(nSplits,dtype=object)
    FFTPowersSplit = numpy.zeros(nSplits,dtype=object)
    TimesBoundaries = []
    for i in range(nSplits):
        # For each time slice, perform the FFT and store the powers and frequencies
        FFTFreqsSplit[i],FFTPowersSplit[i] = SPF.FourierTransformPower(PowersSplit[i],TimesSplit[i])
        TimesBoundaries.append(TimesSplit[i][0])
        
    print("FFT Step size:", FFTFreqsSplit[0][1]-FFTFreqsSplit[0][0])     

    xAxisTimes = []
    for i in range(len(TimesSplit)):
        xAxisTimes.append(TimesSplit[i][0])   

    nBoxes = nFreqBins
    boundarySeparation = (upperFreqLimit-lowFreqLimit)/nBoxes
    yFrequencyBounds = numpy.arange(start=lowFreqLimit, stop=upperFreqLimit+boundarySeparation, step=boundarySeparation)
    
    sampleRate = (Times[len(Times)//2]-Times[len(Times)//2-1])**-1
    print("Sample Rate:",sampleRate)
    nyquistFreq = sampleRate/2
    FFTSplitLength = FFTPowersSplit[0].shape
    FFTStepSize = nyquistFreq/FFTSplitLength
    print("Frequency Resolution:", boundarySeparation)
    if boundarySeparation < FFTStepSize:
        print("WARNING: Frequency bins are smaller than FFT step size.")
    
    
    freqAmpDictionaries = []
    # Create a list of dictionaries with freq:Amplitude pairs for each time step
    for i in range(len(xAxisTimes)):
        freqAmpZip = zip(FFTFreqsSplit[i],FFTPowersSplit[i])
        freqAmpDict = dict(freqAmpZip)
        freqAmpDictionaries.append(freqAmpDict)
    
    totalAmplitudesBounded = numpy.zeros((len(xAxisTimes),len(yFrequencyBounds)-1))
    
    for tIndex in range(len(xAxisTimes)): # for each timestep
        for i in range(len(yFrequencyBounds)-1): # for each bin
            lowerBound = yFrequencyBounds[i]
            upperBound = yFrequencyBounds[i+1]
            maxPowerInThisBin = 0
            for key in freqAmpDictionaries[tIndex]: # For each frequency in the fft
                if key<upperBound and key>lowerBound: # If the frequency is in the range of the bin
                    if freqAmpDictionaries[tIndex][key] > maxPowerInThisBin:
                        # If the power for the current frequency is larger than the current max in this bin, replace it.
                        maxPowerInThisBin = freqAmpDictionaries[tIndex][key]
            # Once all this is done for the bin, make the value of the bin equal to the max
            totalAmplitudesBounded[tIndex,i] += maxPowerInThisBin
    
    xAxisTimes = numpy.array(xAxisTimes)
    if len(xAxisTimes) > 1:
        xAxisTimes = numpy.append(xAxisTimes,xAxisTimes[-1]+(xAxisTimes[-1]-xAxisTimes[-2]))
    else:
        xAxisTimes = numpy.append(xAxisTimes,TimesSplit[-1][-1])
    
    print("")
    return (xAxisTimes,yFrequencyBounds,totalAmplitudesBounded)


def ApplyFFTAnalyser(times, signal, lowSearchFreq, highSearchFreq, nTimeSplits, nFreqBins):
    ### FFT Analyser Testing ###
    spectrumAnalyserOutput = FFTAnalyser(signal, times, lowSearchFreq, highSearchFreq, nTimeSplits=nTimeSplits, nFreqBins=nFreqBins)
    zFFT = spectrumAnalyserOutput[2]
    
    figFFTTimeDependent,axFFTTimeDependent = pyplot.subplots(nrows=1,ncols=1,figsize = [12,8])
    im = axFFTTimeDependent.pcolormesh(spectrumAnalyserOutput[0],spectrumAnalyserOutput[1], numpy.transpose(zFFT))
    axFFTTimeDependent.set_xlabel("Time (s)")
    axFFTTimeDependent.set_ylabel("Frequency (Hz)")
    figFFTTimeDependent.colorbar(im, ax=axFFTTimeDependent)
    figFFTTimeDependent.tight_layout()
    return 0







def HeterodyneAnalyser(signal, times, sampleRate,nTimeSplits, lowSearchFreq, highSearchFreq, sweepStep, sweepTime, intermediateFreq, freqResolution, showBandPassResponse=False):
    """Heterodyne (sweep) analyser, sweeps through frequencies at each timestep, mixing with the signal and producing output, similar to older spectrum analysers."""
    print("\n    Heterodyne Analyser")
    
    if freqResolution/sweepStep < 10:
        print("WARNING: Sweep step is not at least 10x smaller than frequency resolution, results may be unreliable.")
        print("Current sweep step:", sweepStep)
        print("Current frequency resolution:", freqResolution)
    
    pointsPerSweep = numpy.floor(sampleRate*sweepTime).astype('int')
    print("Sample rate:",sampleRate)
    print("Points per sweep:",pointsPerSweep)

    lowSweepFreq = lowSearchFreq+intermediateFreq
    highSweepFreq = highSearchFreq+intermediateFreq

    if highSearchFreq-lowSearchFreq > 2*intermediateFreq:
        print("WARNING: Scanning range is larger than 2*intermediate frequency, image frequencies will be present.")

    sweepFreqs = numpy.arange(lowSweepFreq, highSweepFreq, step=sweepStep)
    numSweepFreqs = len(sweepFreqs)
    sweepAmplitude = 1
    
    totalPointsUsed = pointsPerSweep*numSweepFreqs * nTimeSplits
    
    print("Total points needed for sweep:",totalPointsUsed)
    if totalPointsUsed > len(signal):
        print("ERROR: Number of points needed for sweep is greater than length of signal.")
        return 0
    
    signalUsedForAnalysis = signal[:totalPointsUsed]
    timesUsedForAnalysis = times[:totalPointsUsed]
    
    bandpassOrder = 1
    intermediateFreqBandPass = SPF.ButterBandPass(cutoffLow = intermediateFreq-freqResolution/2,
                                                  cutoffHigh = intermediateFreq+freqResolution/2,
                                                  order = bandpassOrder,
                                                  sampleRate = sampleRate)
    
    if showBandPassResponse == True: 
        graphMultiSos = intermediateFreqBandPass.secondOrderSections
        worN = numpy.logspace(numpy.log10(1),numpy.log10(highSearchFreq),num=int(1e7))*2*numpy.pi/sampleRate
        # Plot the frequency response of the Butterworth filter.
        butterFilterFig,butterFilterAx = pyplot.subplots(nrows=1, ncols=1, figsize=[10,8])
        w, h = ScipySignal.sosfreqz(graphMultiSos, worN=worN)
        butterFilterAx.plot(0.5*sampleRate*w/numpy.pi, numpy.abs(h))
        butterFilterAx.axhline(1/numpy.sqrt(2), color='k')
        butterFilterAx.set_title("Bandpass Filter Frequency Response")
        butterFilterAx.set_xlabel('Frequency (Hz)')
        # butterFilterAx.set_xscale("log")
        butterFilterAx.set_ylabel("Response")
        butterFilterAx.set_xlim(intermediateFreq-intermediateFreq/10,intermediateFreq+intermediateFreq/10)
        butterFilterAx.legend()
        butterFilterAx.grid()
    
    # Next bin the frequencies for the display and use the detector types to produce the outputs.
    frequencyBinEdges = numpy.arange(lowSweepFreq-intermediateFreq, highSweepFreq-intermediateFreq, step = freqResolution)
    frequencyBinEdges = numpy.append(frequencyBinEdges, frequencyBinEdges[-1]+freqResolution)
    numBins = len(frequencyBinEdges)-1
    # Need to use the detector now to turn the frequency amplitudes from an array with many values
    # to an array with length as long as frequencyBinEdges-1
    frequenciesPerBin = numSweepFreqs//numBins
        
    # Here things need to be split into separate timesteps
    timeSegments = numpy.array_split(timesUsedForAnalysis, nTimeSplits)
    signalSegments = numpy.array_split(signalUsedForAnalysis, nTimeSplits)
    displayValues = []
     
    for t in range(nTimeSplits):

        currentTimeSplit = numpy.split(timeSegments[t], numSweepFreqs) # Take the current time segment and split it into separate chunks for each sweep frequency
        signalSplit = numpy.split(signalSegments[t], numSweepFreqs) # Take the signal for the time segment and split it into chunks for each sweep frequency
    
        mixedSinAmplitudes = [] # These will be the amplitudes for the current timestep
        
        for i in range(numSweepFreqs):
            # For each sweep frequency, mix the generated local oscillator with the signal at the corresponding times
            currentTimes = currentTimeSplit[i]
            currentSignal = signalSplit[i]
            currentSweepSin = sweepAmplitude*numpy.sin(2*numpy.pi*sweepFreqs[i]*currentTimes)
            currentMixedSin = currentSweepSin*currentSignal # Should be at the intermediate frequency and some higher frequency
            filteredMixedSin = intermediateFreqBandPass.ApplyFilter(currentMixedSin)
            
            currentSweepCos = sweepAmplitude*numpy.cos(2*numpy.pi*sweepFreqs[i]*currentTimes)
            currentMixedCos = currentSweepCos * currentSignal
            filteredMixedCos = intermediateFreqBandPass.ApplyFilter(currentMixedCos)

            filteredMixedSin = numpy.sqrt(filteredMixedCos**2 + filteredMixedSin**2)
            
            intermediateEnvelopeMax = max(abs(filteredMixedSin))
            
            mixedSinAmplitudes.append(intermediateEnvelopeMax)
    
        currentTimeStepDisplayValues = []
        
        for i in range(numBins):
            # Start with a max peak detector type
            lowIndex = i*frequenciesPerBin
            highIndex =(i+1)*frequenciesPerBin
            currentTimeStepDisplayValues.append( max(mixedSinAmplitudes[lowIndex:highIndex]) )
        displayValues.append(currentTimeStepDisplayValues)
        
    displayValues = numpy.array(displayValues)
    print("displayValues shape:", displayValues.shape)
    
    xAxisTimes = [] 
    for i in range(len(timeSegments)):
        xAxisTimes.append(timeSegments[i][0]) 
    
    if len(xAxisTimes) > 1:
        xAxisTimes.append(xAxisTimes[-1]+xAxisTimes[-1]-xAxisTimes[-2])
    else:
        xAxisTimes.append(times[-1])
    
    print("")
    return (xAxisTimes, frequencyBinEdges, displayValues)
    
    
    
    
    

def parallelLockInAmplifiers(signal, times, lowFrequency, highFrequency, lowPassCutoff, sampleRate):
    """Multiple lock in amplifiers, each scanning a particular frequency range simultaneously."""
    print("\n    Lock-In Amplifier Analyser")
    # A set of lock-in amplifiers each scanning for one frequency
    numberOfLockInAmps = 10
    lockInAmpOutputs = []
    lowPassOrder = 5
    lowPassFilter = SPF.ButterLowPass(lowPassCutoff, lowPassOrder, sampleRate)
    for i in range(numberOfLockInAmps):
        frequency = lowFrequency + i*(highFrequency-lowFrequency)/numberOfLockInAmps
        reference = numpy.sin(2*numpy.pi * frequency * times)
        referencePhaseShifted = numpy.cos(2*numpy.pi * frequency * times)
        lockInAmplifier = LIA.LockInAmp(signal, reference, referencePhaseShifted, lowPassFilter)
        quadratureMagnitude, quadraturePhase = lockInAmplifier.ProcessQuadrature()
        lockInAmpOutputs.append(quadratureMagnitude) 
     
    # Make the frequency boundaries for the plot
    yFrequencyBounds = numpy.arange(start=lowFrequency, stop=highFrequency, step=(highFrequency-lowFrequency)/numberOfLockInAmps)
    yFrequencyBounds = numpy.append(yFrequencyBounds,highFrequency)
        
    return yFrequencyBounds, lockInAmpOutputs
    
    
    
def SweepingLockInAmplifier(signal, times, startFrequency, frequencyGradient, lowPassCutoff, sampleRate, showGraphs=False):
    """A normal lock in amplifier but with a sweeping reference signal."""
    # print("\n    Lock-In Amplifier (Chirp Reference) Analyser")   
    timeLength = times[-1]-times[0]
    lockInAmpOutputs = []
    lowPassOrder = 5
    lowPassFilter = SPF.ChebyLowPass(lowPassCutoff, lowPassOrder, sampleRate)     
    
    finalFrequency = startFrequency + frequencyGradient * timeLength
    chirpFrequencyCoefficient = (finalFrequency - startFrequency) * times/(2*timeLength) + startFrequency # ((f1-f0)t/2T+f0)
    reference = numpy.sin(2*numpy.pi * chirpFrequencyCoefficient * times) 
    referencePhaseShifted = numpy.cos(2*numpy.pi * chirpFrequencyCoefficient * times)
    
    lockInAmplifier = LIA.LockInAmp(signal, reference, referencePhaseShifted, lowPassFilter)
    quadratureMagnitude, quadraturePhase = lockInAmplifier.ProcessQuadrature()
    lockInAmpOutputs.append(quadratureMagnitude)
    
    if showGraphs==True:
        figLIATest, axLIATest = pyplot.subplots(1,1, figsize=[18,8])
        axLIATest.plot(times,quadratureMagnitude)
        axLIATest.set_xlabel("Time (s)")
        axLIATest.set_ylabel("Voltage (V)")
        
        figRef, axRef = pyplot.subplots(1,1, figsize=[18,8])
        axRef.plot(times[:100], reference[:100])
        axRef.set_xlabel("Time (s)")
        axRef.set_ylabel("Reference Amplitude")
        axRef.set_title("Reference Start")
        
        figRefEnd, axRefEnd = pyplot.subplots(1,1, figsize=[18,8])
        axRefEnd.plot(times[-100:], reference[-100:])
        axRefEnd.set_xlabel("Time (s)")
        axRefEnd.set_ylabel("Reference Amplitude")
        axRefEnd.set_title("Reference End")
    
        # Check FFT to see if sweep is correct
        spectrumAnalyserOutput = FFTAnalyser(reference+signal, times, 2.00e7, 2.10e7, 10, 40)
        zFFT = spectrumAnalyserOutput[2]
        figFFTTimeDependent,axFFTTimeDependent = pyplot.subplots(nrows=1,ncols=1,figsize = [12,8])
        im = axFFTTimeDependent.pcolormesh(spectrumAnalyserOutput[0],spectrumAnalyserOutput[1], numpy.transpose(zFFT))
        axFFTTimeDependent.set_xlabel("Time (s)")
        axFFTTimeDependent.set_ylabel("Frequency (Hz)")
        figFFTTimeDependent.colorbar(im, ax=axFFTTimeDependent)
        figFFTTimeDependent.tight_layout()
    return quadratureMagnitude


def SequentialSweepingLockInAmplifier(signal, times, startFrequency, frequencyGradient, lowPassCutoff, sampleRate, showGraphs=False):
    timeLength = times[-1]-times[0]
    finalFrequency = startFrequency + frequencyGradient * timeLength
    
    rootMinusOne = complex(0,1)
    phaseOffset = 0
    
    chirpConstant = 2*numpy.pi*frequencyGradient # ((f1-f0)t/2T+f0)
    complexReference = numpy.exp(rootMinusOne * (phaseOffset + startFrequency*2*numpy.pi * times + chirpConstant*times**2/2)) 
    reference = numpy.imag(complexReference)
    referencePhaseShifted = numpy.real(complexReference)
    lowPassOrder = 5
    lowPassFilter = SPF.ChebyLowPass(lowPassCutoff, lowPassOrder, sampleRate) 
    
    quadratureMagnitudes = []
    quadraturePhases = []
    
    chunkSize = len(times)//10000
    for i in range(len(times)//chunkSize):
        j = chunkSize*i    
        if j==0:
            filterStateInPhase = [[0,0],[0,0],[0,0]] # lowPassFilter.GetInitialFilterState()
            filterStateOutOfPhase = [[0,0],[0,0],[0,0]] # lowPassFilter.GetInitialFilterState()    
        # Get lock-in amp output
        lockInAmplifier = LIA.LockInAmp(signal[j:j+chunkSize], reference[j:j+chunkSize], referencePhaseShifted[j:j+chunkSize], lowPassFilter)
        quadratureMagnitude, quadraturePhase, filterStateInPhase, filterStateOutOfPhase = lockInAmplifier.ProcessQuadratureSequential(filterStateInPhase, filterStateOutOfPhase)
        quadratureMagnitudes.extend(quadratureMagnitude)
        quadraturePhases.extend(quadraturePhase)

    if showGraphs==True:
        figLIATest, axLIATest = pyplot.subplots(1,1, figsize=[18,8])
        axLIATest.plot(times, quadratureMagnitudes)
        axLIATest.set_xlabel("Time (s)")
        axLIATest.set_ylabel("Voltage (V)")

    spectrumAnalyserOutput = FFTAnalyser(reference,
                                         times,
                                         lowFreqLimit=20e6,
                                         upperFreqLimit=21e6,
                                         nTimeSplits=10,
                                         nFreqBins=40)
    zFFT = spectrumAnalyserOutput[2]
    figFFTTimeDependent,axFFTTimeDependent = pyplot.subplots(nrows=1,ncols=1,figsize = [12,8])
    im = axFFTTimeDependent.pcolormesh(spectrumAnalyserOutput[0],spectrumAnalyserOutput[1], numpy.transpose(zFFT))
    axFFTTimeDependent.set_xlabel("Time (s)")
    axFFTTimeDependent.set_ylabel("Frequency (Hz)")
    figFFTTimeDependent.colorbar(im, ax=axFFTTimeDependent)
    figFFTTimeDependent.tight_layout()
    axFFTTimeDependent.set_title("FFT of Reference Signal")

    return quadratureMagnitudes







def SweepingLockInAmplifierWithPhaseAdjust(signal, times, initialCyclotronAngFreq, downmixedCyclotronAngFreq, emittedPower, gamma, lowPassCutoff, sampleRate):
    """Phase tracking lock in amplifier (in progress)"""
    print("\n    Lock-In Amplifier (Tracking Chirp Reference) Analyser")   
    timeLength = times[-1]-times[0] 
    timestep = times[1]-times[0]
    lowPassOrder = 5
    lowPassFilter = SPF.ChebyLowPass(lowPassCutoff, lowPassOrder, sampleRate)   
    
    rootMinusOne = complex(0,1)
    phaseOffset = 0
    chirpConstant = ( initialCyclotronAngFreq*emittedPower / (eMass*gamma) ) / 299792458**2
    print("Chirp constant:", chirpConstant)

    complexReference = numpy.exp(rootMinusOne * (phaseOffset + downmixedCyclotronAngFreq * times + chirpConstant*times**2/2)) 
    reference = numpy.real(complexReference)
    referencePhaseShifted = numpy.imag(complexReference)
    
    quadratureMagnitudes = []
    quadraturePhases = []
    
    chunkSize = len(times)//10000
    
    usedReference = []
    chirpConstants = [chirpConstant]
    for i in range(len(times)//chunkSize):
        j = chunkSize*i
        usedReference.extend(reference[j:j+chunkSize])
        if j==0:
            filterStateInPhase = [[0,0],[0,0],[0,0]]# lowPassFilter.GetInitialFilterState()
            filterStateOutOfPhase = [[0,0],[0,0],[0,0]]# lowPassFilter.GetInitialFilterState()

        # Get lock-in amp output
        lockInAmplifier = LIA.LockInAmp(signal[j:j+chunkSize], reference[j:j+chunkSize], referencePhaseShifted[j:j+chunkSize], lowPassFilter)
        quadratureMagnitude, quadraturePhase, filterStateInPhase, filterStateOutOfPhase = lockInAmplifier.ProcessQuadratureSequential(filterStateInPhase, filterStateOutOfPhase)

        quadratureMagnitudes.extend(quadratureMagnitude)
        quadraturePhases.extend(quadraturePhase)
        
        phaseDerivatives = []
        lastDerivativeSmall = False
        twoSuccessiveSmallDerivatives = False
        doublePhaseDerivatives = []
        for i in range(len(quadraturePhase)-1):
            phaseDerivative = (quadraturePhase[i+1]-quadraturePhase[i]) / timestep
            if abs(phaseDerivative) < 1e8:
                phaseDerivatives.append(phaseDerivative)
                if lastDerivativeSmall == True:
                    doublePhaseDerivative = (phaseDerivatives[-1]-phaseDerivatives[-2]) / timestep
                    if abs(doublePhaseDerivative) < 1e8:
                        print("Double derivative")
                        doublePhaseDerivatives.append(doublePhaseDerivative)
                lastDerivativeSmall = True
            else:
                print("Dismissing a derivative - too large")
                lastDerivativeSmall = False
                
                
        averagePhaseDerivative = numpy.mean(phaseDerivatives)
        print("Phase derivative:", averagePhaseDerivative)
        
        if len(doublePhaseDerivatives) > 0:
            averageDoublePhaseDerivative = numpy.mean(doublePhaseDerivatives)
            print("Double Phase Derivative:", averageDoublePhaseDerivative)

        chirpConstants.append(chirpConstant)
        
        
        # Take phase information and adjust reference
        # reference = numpy.sin(2*numpy.pi * chirpFrequencyCoefficient * times) # - quadraturePhase[-1])
        # referencePhaseShifted = numpy.cos(2*numpy.pi * chirpFrequencyCoefficient * times) # - quadraturePhase[-1])
        # print("Threshold:", 0.5e-8*timestep*chunkSize)
        # print("Integrated Sum:", numpy.sum(numpy.array(quadratureMagnitude*timestep)))
        # if numpy.sum(numpy.array(quadratureMagnitude*timestep)) > 1.5e-8*timestep*chunkSize:
        #     print("Activating phase matching")
        #     chirpConstant = chirpConstant - averageDoublePhaseDerivative
        #     complexReference = numpy.exp(rootMinusOne * (phaseOffset + downmixedCyclotronAngFreq * times + chirpConstant*times**2/2)) 
        #     reference = numpy.real(complexReference)
        #     referencePhaseShifted = numpy.imag(complexReference)

    
    figRef, axRef = pyplot.subplots(1,1, figsize=[18,8])
    axRef.plot(times[:100], reference[:100])
    axRef.set_xlabel("Time (s)")
    axRef.set_ylabel("Reference Amplitude")
    axRef.set_title("Reference Start")
    
    figRefEnd, axRefEnd = pyplot.subplots(1,1, figsize=[18,8])
    axRefEnd.plot(times[-100:], reference[-100:])
    axRefEnd.set_xlabel("Time (s)")
    axRefEnd.set_ylabel("Reference Amplitude")
    axRefEnd.set_title("Reference End")
    
    figFrequencyGradient, axFrequencyGradient = pyplot.subplots(1, 1, figsize=[18,8])
    axFrequencyGradient.plot(times[::chunkSize], numpy.array(chirpConstants[:-1])/(2*numpy.pi))
    axFrequencyGradient.set_xlabel("Time (s)")
    axFrequencyGradient.set_ylabel("Frequency Gradient (Hz/s)")
    
    spectrumAnalyserOutput = FFTAnalyser(usedReference,
                                         times,
                                         lowFreqLimit=20e6,
                                         upperFreqLimit=21e6,
                                         nTimeSplits=10,
                                         nFreqBins=40)
    zFFT = spectrumAnalyserOutput[2]
    
    figFFTTimeDependent,axFFTTimeDependent = pyplot.subplots(nrows=1,ncols=1,figsize = [12,8])
    im = axFFTTimeDependent.pcolormesh(spectrumAnalyserOutput[0],spectrumAnalyserOutput[1], numpy.transpose(zFFT))
    axFFTTimeDependent.set_xlabel("Time (s)")
    axFFTTimeDependent.set_ylabel("Frequency (Hz)")
    figFFTTimeDependent.colorbar(im, ax=axFFTTimeDependent)
    figFFTTimeDependent.tight_layout()
    axFFTTimeDependent.set_title("FFT of Reference Signal")
    
    return quadratureMagnitudes, quadraturePhases
    
    
    
    
    
