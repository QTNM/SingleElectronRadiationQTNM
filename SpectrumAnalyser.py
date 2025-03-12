import SignalProcessingFunctions as SPF
import numpy
import matplotlib.pyplot as pyplot
from scipy import signal as ScipySignal
import LockInAmplifierObjectOriented as LIA
# from memory_profiler import profile

from scipy.signal import find_peaks

eMass = 9.1093837015e-31 # kg

def FFTAnalyser(Powers, Times, lowFreqLimit, upperFreqLimit, nTimeSplits, nFreqBins, showGraphs=False):
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
        FFTFreqsSplit[i],FFTPowersSplit[i] = SPF.FourierTransformWaveform(PowersSplit[i],TimesSplit[i])
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
    if showGraphs==True:
        zFFT = totalAmplitudesBounded
        figFFTTimeDependent,axFFTTimeDependent = pyplot.subplots(nrows=1,ncols=1,figsize = [12,8])
        im = axFFTTimeDependent.pcolormesh(xAxisTimes,yFrequencyBounds, numpy.transpose(zFFT))
        axFFTTimeDependent.set_xlabel("Time (s)")
        axFFTTimeDependent.set_ylabel("Frequency (Hz)")
        colorbar = figFFTTimeDependent.colorbar(im, ax=axFFTTimeDependent)
        colorbar.ax.set_ylabel("Fourier Transform Magnitude (V)")
        figFFTTimeDependent.tight_layout()
        axFFTTimeDependent.set_ylim(350e3,465e3)
    
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
    #axFFTTimeDependent.set_ylim(0,1e6)
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
    
    
    
def SweepingLockInAmplifier(signal, times, startFrequency, frequencyGradient, lowPassCutoff, sampleRate, showGraphs=False,  filterType = "Scipy Single Pole IIR"):
    """A normal lock in amplifier but with a sweeping reference signal."""
    # print("\n    Lock-In Amplifier (Chirp Reference) Analyser")   
    timeLength = times[-1]-times[0]
    lockInAmpOutputs = []
    lowPassOrder = 1
    if filterType == "Single Pole IIR":
        lowPassFilter = SPF.SinglePoleIIR(timeConstantInSamples = 1e-4*sampleRate)
    elif filterType == "Chebyshev":
        lowPassFilter = SPF.ChebyLowPass(lowPassCutoff, lowPassOrder, sampleRate)   
    elif filterType == "Butterworth":
        lowPassFilter = SPF.ButterLowPass(lowPassCutoff, lowPassOrder, sampleRate)
    elif filterType == "Scipy Single Pole IIR":
        lowPassFilter = SPF.ScipySinglePoleIIR(lowPassCutoff, lowPassOrder, sampleRate)
    else:
        print("Filter type unknown, default to Butterworth")
        lowPassFilter = SPF.ButterLowPass(lowPassCutoff, lowPassOrder, sampleRate)
    
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
        axLIATest.set_xlim(0,0.0001)
        
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
        spectrumAnalyserOutput = FFTAnalyser(reference, times, 300e3, 900e3, 10, 40)
        zFFT = spectrumAnalyserOutput[2]
        figFFTTimeDependent,axFFTTimeDependent = pyplot.subplots(nrows=1,ncols=1,figsize = [12,8])
        im = axFFTTimeDependent.pcolormesh(spectrumAnalyserOutput[0],spectrumAnalyserOutput[1], numpy.transpose(zFFT))
        axFFTTimeDependent.set_xlabel("Time (s)")
        axFFTTimeDependent.set_ylabel("Frequency (Hz)")
        figFFTTimeDependent.colorbar(im, ax=axFFTTimeDependent)
        figFFTTimeDependent.tight_layout()
    return quadratureMagnitude




def SweepingLockInAmplifierTimeShifted(signal, times, startFrequency, frequencyGradient, lowPassCutoff, sampleRate, showGraphs=False,  filterType = "Scipy Single Pole IIR", startTime=0, LIAReferenceLength=0.001):
    """A sweeping lock in amplifier with custom start time. Also has repetition after some amount of time."""
    # print("\n    Lock-In Amplifier (Chirp Reference) Analyser")   
    # timeLength = times[-1]-times[0]
    lockInAmpOutputs = []
    
    # Set low pass filter properties
    lowPassOrder = 1
    if filterType == "Single Pole IIR":
        lowPassFilter = SPF.SinglePoleIIR(timeConstantInSamples = 1e-4*sampleRate)
    elif filterType == "Chebyshev":
        lowPassFilter = SPF.ChebyLowPass(lowPassCutoff, lowPassOrder, sampleRate)   
    elif filterType == "Butterworth":
        lowPassFilter = SPF.ButterLowPass(lowPassCutoff, lowPassOrder, sampleRate)
    elif filterType == "Scipy Single Pole IIR":
        lowPassFilter = SPF.ScipySinglePoleIIR(lowPassCutoff, lowPassOrder, sampleRate)
    else:
        print("Filter type unknown, default to Butterworth")
        lowPassFilter = SPF.ButterLowPass(lowPassCutoff, lowPassOrder, sampleRate)
    
    
    # finalFrequency = startFrequency + frequencyGradient * timeLength
    # chirpFrequencyCoefficient = (finalFrequency - startFrequency) * times/(2*timeLength) + startFrequency # ((f1-f0)t/2T+f0)
    # (finalFrequency - startFrequency) / timelength = frequencyGradient so can simplify this
    # Also add in reset in the form of a modulo on the incoming time, so it repeats each x ms
    chirpFrequencyCoefficient = frequencyGradient * numpy.remainder(times-startTime, LIAReferenceLength) / 2 + startFrequency
    reference = numpy.sin(2*numpy.pi * chirpFrequencyCoefficient * numpy.remainder(times-startTime, LIAReferenceLength))
    referencePhaseShifted = numpy.cos(2*numpy.pi * chirpFrequencyCoefficient * numpy.remainder(times-startTime, LIAReferenceLength))
    
    lockInAmplifier = LIA.LockInAmp(signal, reference, referencePhaseShifted, lowPassFilter)
    quadratureMagnitude, quadraturePhase = lockInAmplifier.ProcessQuadrature()
    lockInAmpOutputs.append(quadratureMagnitude)
    
    if showGraphs==True:
        figLIATest, axLIATest = pyplot.subplots(1,1, figsize=[18,8])
        axLIATest.plot(times,quadratureMagnitude)
        axLIATest.set_xlabel("Time (s)")
        axLIATest.set_ylabel("Voltage (V)")
        axLIATest.set_xlim(0,0.0001)
        
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
        spectrumAnalyserOutput = FFTAnalyser(reference, times, 300e3, 900e3, 10, 40)
        zFFT = spectrumAnalyserOutput[2]
        figFFTTimeDependent,axFFTTimeDependent = pyplot.subplots(nrows=1,ncols=1,figsize = [12,8])
        im = axFFTTimeDependent.pcolormesh(spectrumAnalyserOutput[0],spectrumAnalyserOutput[1], numpy.transpose(zFFT))
        axFFTTimeDependent.set_xlabel("Time (s)")
        axFFTTimeDependent.set_ylabel("Frequency (Hz)")
        figFFTTimeDependent.colorbar(im, ax=axFFTTimeDependent)
        figFFTTimeDependent.tight_layout()
    return quadratureMagnitude
    







def TestFFTAnalyser():
    timeLength = 1e-3
    times = numpy.arange(0,timeLength,step=1e-9)
    signal = 1.055 * 11e-3*numpy.sqrt(2)*  SPF.GenerateChirpSignal(times, startFrequency=350e3, frequencyGradient=1.11e8)
    noise = SPF.GenerateAdderSpectrumNoise(timeLength, sampleRate=1e9, noiseRMS=290e-3)
    # noise = numpy.random.normal(0, SPF.GetRMS(signal)/0.00764917, len(signal))
        
    signal *= 1

    print("RMS of noise:", SPF.GetRMS(noise))
    print("SNR: ", SPF.GetRMS(signal)/SPF.GetRMS(noise))
    # For 500MHz bandwidth, want 0.00764917 SNR (RMS)

    
    signal +=noise
    nTimeSplits = int(timeLength// 95e-6) # number of time segments, optimal is 95us for 111kHz/ms freq grad
    # print(nTimeSplits)
    nFreqBins = int(0.5e9 // 10.6e3)
    FFTAnalyser(signal, times, lowFreqLimit=0, upperFreqLimit=0.5e9, nTimeSplits=nTimeSplits, nFreqBins=nFreqBins, showGraphs=True)
    return 0



def TestLIAResponse():
    timeLength = 1e-3
    times = numpy.arange(0,timeLength,step=1e-9)
    signal = 3* 11e-3*numpy.sqrt(2)*  SPF.GenerateChirpSignal(times, startFrequency=350e3, frequencyGradient=1.11e8)
    signal[len(signal)//2:] = 0
    noise = SPF.GenerateAdderSpectrumNoise(timeLength, sampleRate=1e9, noiseRMS=290e-3)
    signal +=noise
    
    
    
    startFrequency = 350e3
    frequencyGradient = 1.11e8
    lowPassCutoff=15e3
    sampleRate = 1/(times[1]-times[0])
    
    LIAMagnitudeOutput = SweepingLockInAmplifier(signal, times, startFrequency, frequencyGradient, lowPassCutoff, sampleRate, showGraphs=False,  filterType = "Scipy Single Pole IIR")
    lowPassCutoffNarrow=1.5e3
    LIAMagnitudeOutputNarrowFilter = SweepingLockInAmplifier(signal, times, startFrequency, frequencyGradient, lowPassCutoffNarrow, sampleRate, showGraphs=False,  filterType = "Scipy Single Pole IIR")

    fig, ax = pyplot.subplots(1, 1, figsize=[16,8])
    ax.plot(times, LIAMagnitudeOutput, label="15 kHz")
    ax.plot(times, LIAMagnitudeOutputNarrowFilter, label="1.5 kHz")
    ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    return 0











def CalculatePeakMean(data, maxIndex, peakWidth):
        # peakWidth = 50 # samples, gives half the width of the peak to be investigated
        if maxIndex-peakWidth < 0:
            lowerPeakIndex = 0
        else:
            lowerPeakIndex = maxIndex-peakWidth
        if maxIndex+peakWidth > len(data)-1:
            upperPeakIndex = -2
        else:
            upperPeakIndex = maxIndex+peakWidth
            
        dataMeanAroundMax = numpy.average(data[lowerPeakIndex : upperPeakIndex+1])
        
        return dataMeanAroundMax

def CalculateBiggestMaxMeanOfNPeaks(data, nPeaks, peakWidth, distance=25, printGraphs = False, broadnessArray = None, label=None):
    peakIndices, peakDict = find_peaks(data, height=(None, None), distance=distance)
    peakHeights = peakDict["peak_heights"]
    # highest_peak_index = peak_indices[numpy.argmax(peak_heights)]
    # second_highest_peak_index = peak_indices[numpy.argpartition(peak_heights,-2)[-2]]
    locationsOfHighestNPeaks = peakIndices[ numpy.argpartition(peakHeights,-nPeaks)[-nPeaks:] ]
    biggestMaxMean = 0
    biggestPeakIndex = 0
    for location in locationsOfHighestNPeaks:
        currentMaxMean = CalculatePeakMean(data, location, peakWidth)
        if currentMaxMean > biggestMaxMean:
            biggestMaxMean = currentMaxMean
            biggestPeakIndex = location
    
    if printGraphs == True:
        print("Location of highest N peaks:", locationsOfHighestNPeaks)
        figTestPeaks,axTestPeaks = pyplot.subplots(1, 1, figsize=[16,8])
        axTestPeaks.plot(data)
        axTestPeaks.scatter(locationsOfHighestNPeaks, numpy.zeros(len(locationsOfHighestNPeaks)))
        axTestPeaks.set_title(label)
    
    
    #if broadnessArray is not None:
        #biggestPeakBroadness = FindBroadnessOfPeak(data, biggestPeakIndex, relativeHeight = 0.75)
        #broadnessArray.append(biggestPeakBroadness)
        
    
    
    return biggestMaxMean

def TestLIAFrequencyResponse():
    timeLength = 1e-3
    times = numpy.arange(0,timeLength,step=1e-9)
    signal = 2 * 11e-3*numpy.sqrt(2)*  SPF.GenerateChirpSignal(times, startFrequency=350e3, frequencyGradient=1.11e8)
    signal[len(signal)//2:] = 0
    noise = SPF.GenerateAdderSpectrumNoise(timeLength, sampleRate=1e9, noiseRMS=290e-3)
    signal +=noise 
    
    startFrequencies = numpy.arange(200e3,500e3, step=1e3)
    frequencyGradient=1.11e8
    lowPassCutoff=15e3
    sampleRate = 1/(times[1]-times[0])
    
    
    peakWidth = int(sampleRate*0.1*timeLength)
    
    
    
    LIAMeanMaxes = []
    
    for startFrequency in startFrequencies:
        LIAMagnitudeOutput = SweepingLockInAmplifier(signal, times, startFrequency, frequencyGradient, lowPassCutoff, sampleRate, showGraphs=False,  filterType = "Scipy Single Pole IIR")
        LIAMeanMaxes.append( CalculateBiggestMaxMeanOfNPeaks(LIAMagnitudeOutput, nPeaks=5, peakWidth=peakWidth, distance=peakWidth/2, printGraphs = False, broadnessArray = None, label=None) )
    
    fig,ax = pyplot.subplots(1, 1, figsize=[16,8])
    ax.plot(startFrequencies, LIAMeanMaxes)
    ax.set_xlabel("Start Frequency (Hz)")
    ax.set_ylabel("LIA Response")
    
    return 0



if __name__ == "__main__":
    #times = numpy.arange(0,0.002,step=0.0000001)
    #signal = SPF.GenerateChirpSignal(times, startFrequency=350000, frequencyGradient=3.45e8)
    #signal[len(signal)//2:] = 0
    #figSig,axSig = pyplot.subplots(1, 1, figsize=[16,8])
    #axSig.plot(times, signal)
    #SweepingLockInAmplifier(signal,times, startFrequency=350000,frequencyGradient=3.45e8,lowPassCutoff=15000,sampleRate=1/(times[1]-times[0]),showGraphs=True)
    #sweepingLockInOutput = SweepingLockInAmplifierTimeShifted(signal, times, startFrequency=350000, frequencyGradient=3.45e8, lowPassCutoff=15000, sampleRate=1/(times[1]-times[0]), showGraphs=True,  filterType = "Scipy Single Pole IIR", startTime=-0.0005, LIAReferenceLength=0.001)

    TestFFTAnalyser()
    #TestLIAResponse()
    #TestLIAFrequencyResponse()





    
    
    
    
    
