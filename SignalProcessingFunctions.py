import numpy
import matplotlib.pyplot as pyplot
from scipy import fft
from scipy import signal
from Constants import k_B
import pandas as pd


def GetRMS(inputArray):
    """Obtains the root mean square of a given input array."""
    sumOfSquares = 0
    for i in range(len(inputArray)):
        sumOfSquares+=inputArray[i]**2
    return numpy.sqrt( sumOfSquares/len(inputArray) )


class ButterLowPass:
    """A Butterworth low pass filter, takes cutoff, order and sample rate as inputs."""
    
    def __init__(self, cutoff, order, sampleRate):
        self.cutoff = cutoff
        self.order = order
        normal_cutoff = cutoff
        self.secondOrderSections = signal.butter(order, normal_cutoff, btype='low', analog=False, output='sos', fs=sampleRate)
    
    def ApplyFilter(self, waveform):
        filteredData = signal.sosfilt(self.secondOrderSections, waveform)
        return filteredData
    
class ScipySinglePoleIIR:
    """Scipy implementation of a single pole IIR filter, seemingly based on Butterworth or other filters already in scipy."""
    
    def __init__(self, cutoff, order, sampleRate):
        self.cutoff = cutoff
        self.order = order
        normal_cutoff = cutoff
        self.secondOrderSections = signal.iirfilter(order, normal_cutoff, btype='lowpass', analog=False, output='sos', fs=sampleRate)
    
    def ApplyFilter(self, waveform):
        filteredData = signal.sosfilt(self.secondOrderSections, waveform)
        return filteredData
    
    

class ButterBandPass:
    """Butterworth band pass filter, uses low cutoff and high cutoff to set the passband."""
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
    """Chebyshev low pass filter, has steeper roll-off than Butterworth but ripple in the passband."""
    def __init__(self, cutoff, order, sampleRate):
        self.cutoff = cutoff
        self.order = order
        self.secondOrderSections = signal.cheby1(order, rp=1, Wn=cutoff, btype="lowpass", analog=False, output="sos", fs=sampleRate)
        
    def ApplyFilter(self, waveform):
        filteredData = signal.sosfilt(self.secondOrderSections, waveform)
        return filteredData
    
    
class SinglePoleIIR:
    """Single pole infinite impulse response filter."""
    
    def __init__(self, timeConstantInSamples):
        self.decayFactor = numpy.exp(-1/timeConstantInSamples)
        # d = e^-1/tau where tau is the time constant of the filter
        # also tau = 1/(2 pi f_c)
        
    def ApplyFilter(self, waveform):
        output = []
        a0 = 1 - self.decayFactor
        b1 = self.decayFactor
        for i in range(len(waveform)):
            if i == 0:
                output.append(waveform[0])
                continue
            output.append( a0*waveform[i] + b1*output[i-1] )
        return numpy.array(output)
    
    
def GenerateChirpSignal(times, startFrequency, frequencyGradient, phaseOffset=0):
    """Generates a chirp signal with amplitude 1, using starting frequency and frequency gradient to set the 
    correct chirp rate."""
    rootMinusOne = complex(0,1)
    startAngularFrequency = startFrequency * 2 * numpy.pi
    chirpConstant = frequencyGradient * 2 * numpy.pi
    complexChirp = numpy.exp(rootMinusOne * (phaseOffset + startAngularFrequency * times + chirpConstant*times**2/2)) 
    return numpy.real(complexChirp)

def Mixer(signal, sampleRate, mixingFrequency, mixingAmplitude, mixingPhase):
    """Mixes a signal with a reference oscillator (sine wave) of a given frequency, amplitude and phase."""
    maxSignalTime = len(signal)/sampleRate
    timeStep = 1/sampleRate
    times = numpy.linspace(0, maxSignalTime, num=len(signal))
    mixerSin = mixingAmplitude * numpy.sin(times*2*numpy.pi*mixingFrequency + mixingPhase)
    return signal * mixerSin

def FourierTransformWaveform(Waveform, Times):
    """Performs a (real) FFT on the input.
    Returns the frequency values and the corresponding amplitude."""
    FFTValues = fft.rfft(Waveform)
    nPoints = len(Times)
    SampleSpacing = (Times.max()-Times.min())/nPoints
    FFTFreqs = fft.rfftfreq(nPoints, d=SampleSpacing)
    FFTValues = numpy.abs(FFTValues)
    return FFTFreqs, FFTValues

def InverseFourierTransform(Freqs, complexPhasors, lengthOfTimes):
    """Performs an inverse (complex) Fourier transform on the input."""
    Powers = fft.irfft(complexPhasors, lengthOfTimes)
    return Powers


def GenerateVoltageNoise(sampleRate, numSamples, temp, resistance, antennaLowCutoff, antennaBandwidth, hidePrints=False):
    """Generates thermal (Johnson-Nyquist) noise as a voltage signal."""
    # To apply the bandwidth filter, add extra time to allow the filter to settle,
    # then just use the end portion of noise from after the filter settled.
    settlingTime = (1/antennaBandwidth)*6
    settlingTimeInSamples = int(settlingTime*sampleRate)
    
    expectedVRMS = numpy.sqrt( 4 * k_B * temp * resistance * antennaBandwidth)
    randomVoltage = numpy.random.normal(loc=0, scale=1.0, size=numSamples+settlingTimeInSamples)
    
    bandwidthFilter = ButterBandPass(cutoffLow = antennaLowCutoff,
                                     cutoffHigh = antennaLowCutoff+antennaBandwidth,
                                     order = 5,
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


def GenerateWhiteVoltageNoise(sampleRate, numSamples, noiseRMS, antennaLowCutoff, antennaBandwidth, hidePrints=False):
    """Generates white noise as a voltage signal."""
    # To apply the bandwidth filter, add extra time to allow the filter to settle,
    # then just use the end portion of noise from after the filter settled.
    settlingTime = (1/antennaBandwidth)*6
    settlingTimeInSamples = int(settlingTime*sampleRate)
    
    expectedVRMS = noiseRMS
    randomVoltage = numpy.random.normal(loc=0, scale=1.0, size=numSamples+settlingTimeInSamples)
    
    bandwidthFilter = ButterBandPass(cutoffLow = antennaLowCutoff,
                                     cutoffHigh = antennaLowCutoff+antennaBandwidth,
                                     order = 5,
                                     sampleRate = sampleRate)
    
    randomVoltage = bandwidthFilter.ApplyFilter(randomVoltage)
    randomVoltage = randomVoltage[settlingTimeInSamples:]
    
    randomVoltageRMS = numpy.sqrt(numpy.mean(randomVoltage**2))
    scalingFactor = expectedVRMS / randomVoltageRMS
    randomVoltage *= scalingFactor
    if hidePrints != True:
        print("Expected VRMS:", expectedVRMS)
        print("Output VRMS:", numpy.sqrt(numpy.mean(randomVoltage**2)))
        #print("Power:", numpy.mean(randomVoltage**2)/(4*resistance))
        #print("Expected Power:", k_B * temp * antennaBandwidth)
    return randomVoltage




def ConvertdBmToVolts(data):
    resistance = 1e6 # mokulab input resistance
    V = numpy.sqrt(resistance/1000) * 10**(data/20)
    return V


def ConvertVoltsTodBm(data):
    resistance = 1e6
    dBm = 10 * numpy.log10( (data**2*1000/resistance) )
    return dBm


def InverseFourierTransformPower(Freqs,fftValues):
    Powers = fft.irfft(fftValues)
    # irfft assumes an even output length, last frequency is at the nyquist frequency (maybe don't need to worry about this, it's just if you want a different output length)
    return Powers



def GenerateAdderSpectrumNoise(timeToGenerate, sampleRate, noiseRMS, checkNoiseSpectrum=False):
    
    averagedFileName = "D:/Project Stuff/Noise Characterisation Data/NoiseSpectrum 50MHz 50pt Averaged.csv"
    averagedNoiseSpectrumData = pd.read_csv(averagedFileName, skiprows=range(7), header=None).to_numpy()
    
    
    ### Try with interpolated powers

    # maxTimeNoiseGenerated = timeToGenerate# 1e-3
    # sample rate of 125e6 Hz
    # numTimePoints = int(maxTimeNoiseGenerated/(1/sampleRate)) # /(1/125e6))
    # print("Num time points:",numTimePoints)
    # numFreqPoints = numTimePoints//2+1

    #interpolatedPowerSpectrumFreqs = numpy.interp(averagedNoiseSpectrumData[:,0])
    #### interpolatedFrequencies = numpy.linspace(0,max(averagedNoiseSpectrumData[:,0]), num = numFreqPoints)
    # The above is incorrect, it doesn't account for a change in sample rate.
    numberOfTimeSamples = int(timeToGenerate*sampleRate)
    if numberOfTimeSamples%2 == 1: # if number of time samples is odd:
        numberOfFrequencySamples = int( (numberOfTimeSamples+1)/2 )
        #print("Odd length:",numberOfFrequencySamples)
    else:
        numberOfFrequencySamples = int ( numberOfTimeSamples/2 + 1 ) # This is from the effect of rfft
        #print("Even length:",numberOfFrequencySamples)
        
    interpolatedFrequencies = numpy.linspace(0,sampleRate/2,num = numberOfFrequencySamples)
    # print("Max freq",max(interpolatedFrequencies))
    interpolatedPowerSpectrumValues = numpy.interp(interpolatedFrequencies, xp=averagedNoiseSpectrumData[:,0], fp=averagedNoiseSpectrumData[:,1])

    if checkNoiseSpectrum == True:    
        figSpectrumInterpolated, axSpectrumInterpolated = pyplot.subplots(1, 1, figsize=[16,8])
        axSpectrumInterpolated.plot(interpolatedFrequencies,interpolatedPowerSpectrumValues, color='b')
        axSpectrumInterpolated.set_title("Interpolated Spectrum Check")
        axSpectrumInterpolated.plot(averagedNoiseSpectrumData[:,0], averagedNoiseSpectrumData[:,1], linestyle='--', color='orange')
        
    ### Now want to try making noise with this power spectrum
    # First: (Optional?) Add 1.26 RMS white noise to the spectrum to simulate the randomness of real spectra on top of the average
    interpolatedPowerSpectrumValues = numpy.random.normal(interpolatedPowerSpectrumValues, scale=1.26)
    
    if checkNoiseSpectrum == True:    
        figSpectrumInterpolatedPlusNoise, axSpectrumInterpolatedPlusNoise = pyplot.subplots(1, 1, figsize=[16,8])
        axSpectrumInterpolatedPlusNoise.plot(interpolatedFrequencies,interpolatedPowerSpectrumValues, color='b')
        axSpectrumInterpolatedPlusNoise.set_title("Interpolated Spectrum Check")
        axSpectrumInterpolatedPlusNoise.plot(averagedNoiseSpectrumData[:,0], averagedNoiseSpectrumData[:,1], linestyle='--', color='orange')
    
    
    # Second: Convert from dBm to Volts before inverse fourier transform with random complex phases
    interpolatedVoltageSpectrum = ConvertdBmToVolts(interpolatedPowerSpectrumValues)
    # complexSpectrum = voltageSpectrum*numpy.exp(phase*1j)
    interpolatedComplexSpectrum = numpy.empty(len(interpolatedVoltageSpectrum), dtype=complex)
    rng = numpy.random.default_rng()
    for i in range(len(interpolatedVoltageSpectrum)):
        phase = rng.random()*2*numpy.pi
        interpolatedComplexSpectrum[i] = interpolatedVoltageSpectrum[i]*numpy.exp(phase*1j)

    #print(len(voltageSpectrum))
    interpolatedPowers = InverseFourierTransformPower(interpolatedFrequencies, interpolatedComplexSpectrum)
    interpolatedPowerRMS = GetRMS(interpolatedPowers)
    desiredRMS = noiseRMS # 0.290
    interpolatedPowers *= desiredRMS / interpolatedPowerRMS
    # print("Noise RMS:", GetRMS(interpolatedPowers))
    #timeLength = 1/(averagedNoiseSpectrumData[2,0]-averagedNoiseSpectrumData[1,0])
    #print("Time length:",timeLength)
    timePerSample = timeToGenerate/len(interpolatedPowers)
    # print("Check sample rate",1/timePerSample)
    
    if checkNoiseSpectrum == True:
        interpolatedTimes = numpy.arange(0,timeToGenerate, step = timePerSample)
        figCheckInterpolatedPowers, axCheckInterpolatedPowers = pyplot.subplots(3, 1, figsize=[16,24])
        axCheckInterpolatedPowers[0].plot(interpolatedTimes, interpolatedPowers)
        FFTCheckInterpolatedFreqs, FFTCheckInterpolatedValues = FourierTransformWaveform(interpolatedPowers, interpolatedTimes)
        axCheckInterpolatedPowers[1].plot(FFTCheckInterpolatedFreqs, FFTCheckInterpolatedValues)
        axCheckInterpolatedPowers[2].plot(FFTCheckInterpolatedFreqs, ConvertVoltsTodBm(FFTCheckInterpolatedValues))
        axCheckInterpolatedPowers[2].plot(interpolatedFrequencies, interpolatedPowerSpectrumValues,color='orange', linestyle='--')
        axCheckInterpolatedPowers[2].plot(averagedNoiseSpectrumData[:,0], averagedNoiseSpectrumData[:,1], linestyle='--', color='green')
        axCheckInterpolatedPowers[2].set_xlim(interpolatedFrequencies[0],interpolatedFrequencies[-1])
        axCheckInterpolatedPowers[0].set_title("Noise Time Series")
        axCheckInterpolatedPowers[1].set_title("Noise Voltage FFT")
        axCheckInterpolatedPowers[2].set_title("Noise Power Spectrum Comparison before/after")
        
    return interpolatedPowers











def RunVoltageNoiseTest():
    sampleRate = 800e6
    times = numpy.arange(0,1e-3,step=1/sampleRate)
    voltageNoise = GenerateVoltageNoise(sampleRate = sampleRate,
                                        numSamples = len(times),
                                        temp = 8,
                                        resistance = 72.2,
                                        antennaLowCutoff = 9e6,
                                        antennaBandwidth = 121812251.502986)# 1e6)
    
    figVoltageNoise, axVoltageNoise = pyplot.subplots(1, 1, figsize=[18,8])
    axVoltageNoise.plot(times, voltageNoise)


def RunThermalNoiseTest():
    maxTime = 1e-3
    sampleRate = 209.357876539817e8
    antennaLowCutoff=9e6
    antennaBandwidth=1.00e6

    times = numpy.arange(0, maxTime, step=1/sampleRate)
    print(len(times))
    noise = GenerateVoltageNoise(sampleRate,
                                 numSamples=len(times),
                                 temp=293,
                                 resistance=1e3,
                                 antennaLowCutoff=antennaLowCutoff,
                                 antennaBandwidth=antennaBandwidth
                                 )
    
    # Graph the noise in frequency space to check the response of the filter
    FFTFreqs, FFTPowers = FourierTransformWaveform(noise, times)

    if len(times) > len(noise):
        times = times[:len(noise)]
    elif len(times) < len(noise):
        noise = noise[:len(times)]
    
    if len(FFTFreqs) > len(FFTPowers):
        FFTFreqs = FFTFreqs[:len(FFTPowers)]
    elif len(FFTFreqs) < len(FFTPowers):
        FFTPowers = FFTPowers[:len(FFTFreqs)]

    figNoiseTest, axNoiseTest = pyplot.subplots(nrows=2, ncols=1, figsize = [12, 16])
    axNoiseTest[0].plot(times, noise)
    axNoiseTest[1].plot(FFTFreqs, FFTPowers)
    axNoiseTest[1].set_xscale("log")
    axNoiseTest[1].set_xlim(antennaLowCutoff/2,(antennaLowCutoff+antennaBandwidth)*2)
    print("RMS Noise:", numpy.sqrt(numpy.mean(noise**2)))
    return 0



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


def runFilterTest():
    print("Cutoff:", 8000)
    print("TC:", 1/(2*numpy.pi*8000))
    print("Sampling Freq:", 1000/0.001, "Hz")
    timeToDecayInSamples = 2e-5*1e6
    # this is just time constant in units of sample number
    print("2e-5s in samples:", timeToDecayInSamples)
    alpha = numpy.exp(-1/timeToDecayInSamples)
    print("alpha from time to decay:", alpha)
    
    lowPassFilter = SinglePoleIIR(timeConstantInSamples=timeToDecayInSamples)
    xs = numpy.arange(0,1e-3,step=1e-6)
    ys = numpy.sin(xs*2*numpy.pi*8000)
    noisyYs = ys#numpy.random.normal(ys, scale = 0.2)
    fig,ax = pyplot.subplots(1, 1, figsize=[16,8])
    ax.plot(xs, noisyYs, label="Before")
    filteredYs = lowPassFilter.ApplyFilter(noisyYs)
    ax.plot(xs, filteredYs, label="After")
    ax.legend()



def ChirpTest():
    times = numpy.arange(0,1e-3,step=1/6.25e6)
    print("Len(times):",len(times))
    signal = GenerateChirpSignal(times, startFrequency=280e3, frequencyGradient=3.45e8, phaseOffset = numpy.random.random(1)*2*numpy.pi)
    print("Amplitude:",numpy.amax(signal), numpy.amin(signal))
    firstBlock = signal[0:1000]
    lastBlock = signal[-1000:]
    FirstFFTFreqs, FirstFFTValues = FourierTransformWaveform(firstBlock, times[0:1000])
    LastFFTFreqs, LastFFTValues = FourierTransformWaveform(lastBlock, times[-1000:])
    figChirpCheck, axChirpCheck = pyplot.subplots(2, 1, figsize=[16,16])
    axChirpCheck[0].plot(FirstFFTFreqs, FirstFFTValues)
    axChirpCheck[1].plot(LastFFTFreqs, LastFFTValues)
    axChirpCheck[0].axvline(280e3)
    axChirpCheck[1].axvline(625e3)
    return 0

def VoltageConversionTest():
    Z0 = 1e6 # Ohms
    V1 = numpy.array([10.6,1]) # Volts
    expectedPower = numpy.array([-9.493882694704595,-30]) # dBm
    print("Expected power:",expectedPower,"dBm")
    print("Calculated power:",ConvertVoltsTodBm(V1), "dBm")
    print("Expected voltage from dBm:",V1,"V")
    print("Calculated voltage from dBm:", ConvertdBmToVolts(expectedPower),"V")

def InverseFourierTransformTest():
    # spectrumFreqs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    spectrumFreqs = numpy.linspace(0,19,num=20)
    spectrumValues = numpy.zeros(20)
    spectrumValues[2]+=1
    # spectrumValues = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
    timeSeriesSampleTime = 1/(spectrumFreqs[-1]*2) # 1/(2*nyquist frequency)
    print("Time series sample rate:", timeSeriesSampleTime, "s")
    ifftData = InverseFourierTransformPower(spectrumFreqs, spectrumValues)
    ifftTimes = numpy.arange(0, (len(ifftData))*timeSeriesSampleTime, step=timeSeriesSampleTime)
    figIFFTCheck, axIFFTCheck = pyplot.subplots(2,1,figsize=[16,16])
    axIFFTCheck[0].plot(spectrumFreqs, spectrumValues)
    axIFFTCheck[1].plot(ifftTimes, ifftData)
    axIFFTCheck[1].axvline(1/spectrumFreqs[2])
    FFTFreqs, FFTValues = FourierTransformWaveform(ifftData, ifftTimes)
    axIFFTCheck[0].plot(FFTFreqs, FFTValues, color='orange')
    #axIFFTCheck[0].set_xlim(0,10)
    return 0

def AdderNoiseTest():
    noise = GenerateAdderSpectrumNoise(timeToGenerate=1e-3, sampleRate=6.25e6, noiseRMS=0.290,checkNoiseSpectrum=True)
    


if __name__ == "__main__":
    # runMixerTest()
    # RunThermalNoiseTest()
    # RunVoltageNoiseTest()
    # runFilterTest()
    # ChirpTest()
    # VoltageConversionTest()
    # InverseFourierTransformTest()
    AdderNoiseTest()