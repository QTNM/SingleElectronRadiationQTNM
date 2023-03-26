import numpy
import matplotlib.pyplot as pyplot
from scipy import fft
from scipy import signal
from Constants import k_B



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
    times = numpy.arange(0, maxSignalTime, step=timeStep)
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
    # this is just time constant in units of sample number, right?
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



if __name__ == "__main__":
    runMixerTest()
    # RunThermalNoiseTest()
    # RunVoltageNoiseTest()
    # runFilterTest()