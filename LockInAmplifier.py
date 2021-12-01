import numpy
import matplotlib.pyplot as pyplot
from scipy import signal
pyplot.rcParams.update({'font.size': 18})

class LockInAmp:
  
    def __init__(self, signal, reference, referencePhaseShifted, lowPassFilter):
        # Accepts inputs of signal, reference, 90 degree phase shifted eference and low pass filter
        self.signal = signal
        self.reference = reference
        self.referencePhaseShifted = referencePhaseShifted
        self.filter = lowPassFilter
        
    def ProcessQuadrature(self):
        # Mixes the signal and reference signals and applies a low pass filter
        # Returns the quadrature magnitude and phase components.
        mixedSignalAndReference = self.signal*self.reference
        mixedSignalAndReferencePhaseShifted = self.signal*self.referencePhaseShifted
        filteredMixedSignal = self.filter.ApplyFilter(mixedSignalAndReference)
        filteredMixedSignalPhaseShifted = self.filter.ApplyFilter(mixedSignalAndReferencePhaseShifted)
        quadratureMagnitude = numpy.sqrt(filteredMixedSignal**2 + filteredMixedSignalPhaseShifted**2)
        quadraturePhase = numpy.arctan2(filteredMixedSignalPhaseShifted, filteredMixedSignal)
        return quadratureMagnitude, quadraturePhase
    
    def GraphQuadrature(self, times, magnitude, phase):
        # Makes a plot of magnitude and phase over time
        fig, ax = pyplot.subplots(nrows=2, ncols=1, figsize=[12,12])
        ax[0].plot(times, magnitude)
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Magnitude")
        ax[1].plot(times, phase)
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Phase")
        fig.tight_layout()
    

    
class ButterLowPass:
    
    def __init__(self, cutoff, order, samplingRate):
        self.cutoff = cutoff
        self.order = order
        nyquistFrequency = 0.5 * samplingRate
        normal_cutoff = cutoff / nyquistFrequency
        self.secondOrderSections = signal.butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    
    def ApplyFilter(self, waveform):
        filteredData = signal.sosfilt(self.secondOrderSections, waveform)
        return filteredData
        

# Set up signal
signalFrequency = 27e9
signalAmplitude = 1
signalTimePeriod = 1/signalFrequency
numberOfPeriods = 5000000
samplingRate = 1e11
times = numpy.arange(0, numberOfPeriods*signalTimePeriod, step=1/samplingRate)
signalSin = numpy.sin(2*numpy.pi*signalFrequency*times) * signalAmplitude

# Add noise to signal
noiseAmplitude = 166
noise = noiseAmplitude*numpy.random.normal(size=len(times))
noisySignal = signalSin + noise

# Set up reference
referenceFrequency = 27.0e9
referencePhaseShift = 0.0*numpy.pi
referenceAmplitude = 1

# Produce reference and 90 degree phase shifted reference
referenceSin = numpy.sin(2*numpy.pi*referenceFrequency*times + referencePhaseShift) * referenceAmplitude
referenceCos = numpy.cos(2*numpy.pi*referenceFrequency*times + referencePhaseShift) * referenceAmplitude

# Plot signal
figNoiseSignal, axNoiseSignal = pyplot.subplots(nrows=1, ncols=1, figsize=[10,7])
axNoiseSignal.plot(times, noisySignal)
axNoiseSignal.set_xlabel("Time (s)")
axNoiseSignal.set_ylabel("Amplitude")
figNoiseSignal.tight_layout()

# Actual Signal



# Set up filter parameters
cutoffs = [1e5, 1e4, 1e3]
cutoffLabels = ["100 kHz", "10 kHz", "1 kHz"]

# Apply lock-in amplifier for range of cutoffs
magnitudesAndPhases = []
for i in range(len(cutoffs)):
    cutoff = cutoffs[i]
    lowPassFilter = ButterLowPass(cutoff,1,samplingRate)
    lockInAmpInstance = LockInAmp(noisySignal, referenceSin, referenceCos, lowPassFilter)
    magnitudesAndPhases.append(lockInAmpInstance.ProcessQuadrature())
    
# Plot multiple cutoffs on same graph
figMultiCutoff, axMultiCutoff = pyplot.subplots(nrows=2, ncols=1, figsize=[12,12])
for i in range(len(cutoffs)):
    axMultiCutoff[0].plot(times, magnitudesAndPhases[i][0], label=cutoffLabels[i])
    axMultiCutoff[1].plot(times, magnitudesAndPhases[i][1])
axMultiCutoff[0].set_ylabel("Magnitude")
axMultiCutoff[1].set_xlabel("Time (s)")
axMultiCutoff[1].set_ylabel("Phase")
figMultiCutoff.legend()
figMultiCutoff.tight_layout()
    