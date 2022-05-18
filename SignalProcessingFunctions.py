import numpy
import matplotlib.pyplot as pyplot
from scipy import fft
from scipy import signal


class ButterLowPass:
    
    def __init__(self, cutoff, order, sampleRate):
        self.cutoff = cutoff
        self.order = order
        normal_cutoff = cutoff
        self.secondOrderSections = signal.butter(order, normal_cutoff, btype='low', analog=False, output='sos', fs=sampleRate)
    
    def ApplyFilter(self, waveform):
        filteredData = signal.sosfilt(self.secondOrderSections, waveform)
        return filteredData
    


class ButterBandPass:
    def __init__(self, cutoffLow, cutoffHigh, order, sampleRate):
        self.cutoffLow = cutoffLow
        self.cutoffHigh = cutoffHigh
        self.order = order
        normal_cutoffLow = cutoffLow
        normal_cutoffHigh = cutoffHigh
        self.secondOrderSections = signal.butter(order, [normal_cutoffLow, normal_cutoffHigh], btype='bandpass', analog=False, output='sos', fs=sampleRate)
 
    def ApplyFilter(self, waveform):
        filteredData = signal.sosfilt(self.secondOrderSections, waveform)
        return filteredData
    
    
    
    
    
class ChebyLowPass:
    def __init__(self, cutoff, order, sampleRate):
        self.cutoff = cutoff
        self.order = order
        self.secondOrderSections = signal.cheby1(order, rp=1, Wn=cutoff, btype="lowpass", analog=False, output="sos", fs=sampleRate)
        
    def ApplyFilter(self, waveform):
        filteredData = signal.sosfilt(self.secondOrderSections, waveform)
        return filteredData
    
    
    def ApplyFilterSequential(self, waveformSection, filterState):
        filteredData, newFilterState = signal.sosfilt(self.secondOrderSections, waveformSection, zi=filterState)
        return filteredData, newFilterState
    
    def GetInitialFilterState(self):
        initialFilterState = signal.sosfilt_zi(self.secondOrderSections)
        return initialFilterState
    




def Mixer(signal, sampleRate, mixingFrequency, mixingAmplitude, mixingPhase):
    # Mixes a signal with a reference oscillator (sine wave) of a given frequency, amplitude and phase
    maxSignalTime = len(signal)/sampleRate
    timeStep = 1/sampleRate
    times = numpy.arange(0, maxSignalTime, step=timeStep)
    mixerSin = mixingAmplitude * numpy.sin(times*2*numpy.pi*mixingFrequency + mixingPhase)
    return signal * mixerSin

def FourierTransformPower(Powers, Times):
    # Do FFT (real) on the input powers
    # Returns the frequency values and the corresponding amplitude
    FFTPowers = fft.rfft(Powers)
    nPoints = len(Times)
    SampleSpacing = (Times.max()-Times.min())/nPoints
    FFTFreqs = fft.rfftfreq(nPoints, d=SampleSpacing)
    FFTPowers = numpy.abs(FFTPowers)
    return FFTFreqs, FFTPowers

def InverseFourierTransformPower(Freqs,complexPhasors, lengthOfTimes):
    Powers = fft.irfft(complexPhasors, lengthOfTimes)
    return Powers



def GenerateThermalNoise(temperature, bandwidth, lengthInSamples, lowFreq, sampleRate):
    k_B = 1.38064852e-23
    kTdB = k_B * temperature * bandwidth
    noise = numpy.random.normal(size = lengthInSamples) * kTdB
    
    lowPassFilter = ButterBandPass(order=5, cutoffLow=lowFreq, cutoffHigh=lowFreq+bandwidth, sampleRate=sampleRate)
    noise = lowPassFilter.ApplyFilter(noise)
    noiseRMS = numpy.sqrt(numpy.mean(noise**2))
    print("Noise RMS:", noiseRMS)
    correctionFactor = kTdB/noiseRMS
    noise *= correctionFactor
    noiseRMS = numpy.sqrt(numpy.mean(noise**2))
    print("Corrected noise RMS:", noiseRMS)
    print("Expected RMS:", kTdB)
    return noise


def GenerateThermalNoiseVoltageFromPSD(sampleRate, lengthInSamples, antennaBandwidth, frequencyLow, resistance, temp):
    # Generates voltage from a flat power spectral density in a given bandwidth.
    print("    Voltage Noise Generation:    ")
    k_B = 1.380649e-23
    
    frequencyStep = sampleRate/2 / (lengthInSamples/2)
    
    print("Frequency Step:", frequencyStep)
    maxFreq = sampleRate/2
    frequencies = numpy.arange(0, maxFreq, step = frequencyStep)
    frequencyComplexValues = []
    
    numBinsTotal = len(frequencies)
    print("Number of bins:",numBinsTotal)
    complexPSDMagnitude = 4 * k_B * temp * resistance
    
    numBinsFilled = 0
    frequencyHigh = frequencyLow+antennaBandwidth

    # Create the PSD in the specified frequencies
    for i in range(len(frequencies)):
        if frequencies[i] >= frequencyLow and frequencies[i] < frequencyHigh:
            complexPhase = (numpy.random.random()-0.5)*numpy.pi*2
            complexNumber = complex(complexPSDMagnitude*numpy.cos(complexPhase), complexPSDMagnitude*numpy.sin(complexPhase)) # (x, y) for x+iy
            frequencyComplexValues.append(complexNumber)
            numBinsFilled+=1
        else:
            frequencyComplexValues.append(complex(0,0))
    
    # Plot the PSD
    figThermalFreqs, axThermalFreqs = pyplot.subplots(1,1, figsize=[12,8])
    axThermalFreqs.plot(frequencies, numpy.abs(frequencyComplexValues))
    print("Number of bins filled:", numBinsFilled)
    
    fftComplexValues = []
    for i in range(len(frequencyComplexValues)):
        fftComplexValues.append( numpy.sqrt( frequencyStep/2 * frequencyComplexValues[i]) )
    
    powers = InverseFourierTransformPower(frequencies, fftComplexValues, 2*(len(frequencies)-1))
    powers*=len(powers)
    newMaxTime = len(powers)/sampleRate
    times = numpy.arange(0, newMaxTime, step = 1/sampleRate)
    if len(times)%2 == 1:
        times = times[:-1]
    figNoiseTime, axNoiseTime = pyplot.subplots(1,1,figsize=[12,8])
    axNoiseTime.plot(times,powers)
    axNoiseTime.set_xlabel("Voltage / V")
    print("Length of powers array:", len(powers))
    print("Length of frequencies array:", len(frequencies))
    
    print("RMS of generated noise:", numpy.sqrt(numpy.mean(powers**2)))
    print("Total power per bin:", frequencyStep * complexPSDMagnitude)
    print("Total power over all bins:", frequencyStep * numBinsFilled * complexPSDMagnitude)
    
    expectedVRMS = numpy.sqrt( 4 * k_B * temp * resistance * antennaBandwidth)
    print("Expected voltage RMS:", expectedVRMS)
    
    VRMSfromPSD = numpy.sqrt(frequencyStep * numBinsFilled * complexPSDMagnitude)
    print("VRMS calculated from PSD:", VRMSfromPSD)
    print(VRMSfromPSD/expectedVRMS)
    print("exp/gen:", expectedVRMS / numpy.sqrt(numpy.mean(powers**2)))
    return powers


def GenerateVoltageNoise(sampleRate, numSamples, temp, resistance, antennaLowCutoff, antennaBandwidth, hidePrints=False):
    k_B = 1.380649e-23
    settlingTime = 1/antennaBandwidth*6
    settlingTimeInSamples = int(settlingTime*sampleRate)    
    
    expectedVRMS = numpy.sqrt( 4 * k_B * temp * resistance * antennaBandwidth)
    randomVoltage = numpy.random.normal(loc=0, scale=1.0, size=numSamples+settlingTimeInSamples)
    
    bandwidthFilter = ButterBandPass(cutoffLow = antennaLowCutoff,
                                     cutoffHigh = antennaLowCutoff+antennaBandwidth,
                                     order = 17,
                                     sampleRate = sampleRate)
    
    randomVoltage = bandwidthFilter.ApplyFilter(randomVoltage)
    randomVoltage = randomVoltage[settlingTimeInSamples:]
    
    randomVoltageRMS = numpy.sqrt(numpy.mean(randomVoltage**2))
    scalingFactor = expectedVRMS / randomVoltageRMS
    randomVoltage *= scalingFactor
    if hidePrints != True:
        print("Expected VRMS:", expectedVRMS)
        print("Output VRMS:", numpy.sqrt(numpy.mean(randomVoltage**2)))
        print("Power:", numpy.mean(randomVoltage**2)/(4*resistance))
        print("Expected Power:", k_B * temp * antennaBandwidth)
    return randomVoltage



def RunVoltageNoiseTest():
    sampleRate = 200e6
    times = numpy.arange(0,1e-3,step=1/sampleRate)
    voltageNoise = GenerateVoltageNoise(sampleRate = sampleRate,
                                        numSamples = len(times),
                                        temp = 293,
                                        resistance = 1e3,
                                        antennaLowCutoff = 9e6,
                                        antennaBandwidth = 1e6)
    
    figVoltageNoise, axVoltageNoise = pyplot.subplots(1, 1, figsize=[18,8])
    axVoltageNoise.plot(times, voltageNoise)


def RunThermalNoiseTest():
    maxTime = 1e-3
    sampleRate = 209.357876539817e8

    times = numpy.arange(0, maxTime, step=1/sampleRate)
    print(len(times))
    noise = GenerateThermalNoiseVoltageFromPSD(sampleRate,
                                        lengthInSamples=len(times),
                                        antennaBandwidth=1.00e6,
                                        frequencyLow=9e6,
                                        resistance=1e3,
                                        temp=293)
    
    # Graph the noise in frequency space to check the response of the filter
    FFTFreqs, FFTPowers = FourierTransformPower(noise, times)

    lengthDifference = len(noise)-len(times)
    if len(times) > len(noise):
        times = times[:len(noise)]
    elif len(times) < len(noise):
        noise = noise[:len(times)]
    
    lengthDifference = len(FFTPowers)-len(FFTFreqs)
    FFTFreqs = FFTFreqs[0:lengthDifference]

    figNoiseTest, axNoiseTest = pyplot.subplots(nrows=2, ncols=1, figsize = [12, 16])
    axNoiseTest[0].plot(times, noise)
    axNoiseTest[1].plot(FFTFreqs, FFTPowers)
    print("RMS Noise:", numpy.sqrt(numpy.mean(noise**2)))
    return 0



def RunFilterTest():
    sampleRate = 1e12
    timeStep = 1/sampleRate
    maxTime = 1e-4   # seconds
    times = numpy.arange(0, maxTime, step=timeStep)
    print("Data length:", len(times))
    
    k_B = 1.38064852e-23
    temperature = 4 # K
    noiseBandwidth = sampleRate/2 # Hz
    noiseAmplitude = k_B * temperature * noiseBandwidth 
    print("Noise amplitude:", noiseAmplitude)
    noise = numpy.random.normal(size=len(times)) * noiseAmplitude
    
    print("Noise RMS pre-filter:", numpy.sqrt(numpy.mean(noise**2)))
    cutoff = 1e4
    order = 20
    lowPassFilter = ButterLowPass(cutoff, order, sampleRate)
    
    mixSin = numpy.sin(2*numpy.pi*times*27e9)
    mixCos = numpy.sin(2*numpy.pi*times*27e9)
    mixedNoise = mixSin*noise
    mixedNoisePS = mixCos*noise
    
    
    noiseFiltered = lowPassFilter.ApplyFilter(noise)
    mixedNoiseFiltered = lowPassFilter.ApplyFilter(mixedNoise)
    mixednoiseFilteredPS = lowPassFilter.ApplyFilter(mixedNoisePS)
    noiseQuadratureMagnitude = numpy.sqrt(mixedNoiseFiltered**2+mixednoiseFilteredPS**2)
    print("Noise RMS post filter:", numpy.sqrt(numpy.mean(noiseFiltered**2)))
    print("Mixed Noise RMS post filter:", numpy.sqrt(numpy.mean(mixedNoiseFiltered**2)))  
    print("Noise Quadrature:", numpy.mean(noiseQuadratureMagnitude))
    
    print("Filter Cutoff:", lowPassFilter.cutoff)
    print("Filter Order:", lowPassFilter.order)
    print("Filter SOS:", lowPassFilter.secondOrderSections)
    
    figNoiseFilter, axNoiseFilter = pyplot.subplots(nrows=2, ncols=1, figsize=[12, 8])
    axNoiseFilter[0].plot(times, noise)
    axNoiseFilter[1].plot(times, noiseFiltered)
    
    noiseFFTFreqs, noiseFFTPowers = FourierTransformPower(noise, times)
    figNoiseFFT, axNoiseFFT = pyplot.subplots(1, 1, figsize=[12, 8])
    axNoiseFFT.plot(noiseFFTFreqs, numpy.abs(noiseFFTPowers))
    
    mixedNoiseFFTFreqs, mixedNoiseFFTPowers = FourierTransformPower(mixedNoise, times)
    figMixedNoiseFFT, axMixedNoiseFFT = pyplot.subplots(1, 1, figsize=[12, 8])
    axMixedNoiseFFT.plot(mixedNoiseFFTFreqs, numpy.abs(mixedNoiseFFTPowers)) 
    
    noiseFilteredFFTFreqs, noiseFilteredFFTPowers = FourierTransformPower(noiseFiltered, times)
    fignoiseFilteredFFT, axnoiseFilteredFFT = pyplot.subplots(1, 1, figsize=[12, 8])
    axnoiseFilteredFFT.plot(noiseFilteredFFTFreqs, numpy.abs(noiseFilteredFFTPowers))     

def runMixerTest():
    sampleRate = 100 # samples per second
    mixingFrequency = 2 # Hz
    mixingAmplitude = 1
    mixingPhase = 0.5*numpy.pi
    
    times = numpy.arange(0, 6, step=1/sampleRate)
    signalAmplitude = 1
    signalFrequency = 1.3 # Hz
    signalPhase = 0*numpy.pi
    testSignal = signalAmplitude*numpy.sin(times*2*numpy.pi*signalFrequency + signalPhase)
    
    mixingSignal = mixingAmplitude*numpy.sin(times*2*numpy.pi*mixingFrequency + mixingPhase)
    
    mixedSignal = Mixer(testSignal, sampleRate, mixingFrequency, mixingAmplitude, mixingPhase)

    figMixerTest,axMixerTest = pyplot.subplots(nrows = 3, ncols = 1, figsize = [10, 10])
    axMixerTest[0].plot(times, testSignal)
    axMixerTest[1].plot(times, mixingSignal)
    axMixerTest[2].plot(times, mixedSignal)
    figMixerTest.tight_layout()





if __name__ == "__main__":
    # runMixerTest()
    # RunFilterTest()
    # RunThermalNoiseTest()
    RunVoltageNoiseTest()