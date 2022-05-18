import numpy
import matplotlib.pyplot as pyplot
import SignalProcessingFunctions as SPF
pyplot.rcParams.update({'font.size': 18})

# from memory_profiler import profile


 
 
class LockInAmp:
  
    def __init__(self, signal, reference, referencePhaseShifted, lowPassFilter):
        # Accepts inputs of signal, reference, 90 degree phase shifted eference and low pass filter
        self.signal = signal
        self.reference = reference
        self.referencePhaseShifted = referencePhaseShifted
        self.filter = lowPassFilter
    
    # @profile
    def ProcessQuadrature(self):
        # Mixes the signal and reference signals and applies a low pass filter
        # Returns the quadrature magnitude and phase components.
        
        filteredMixedSignal = self.filter.ApplyFilter(self.signal*self.reference)
        filteredMixedSignalPhaseShifted = self.filter.ApplyFilter(self.signal*self.referencePhaseShifted)
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
        
    def ProcessQuadratureSequential(self, filterStateInPhase, filterStateOutOfPhase):
        # Mixes signal and reference signals and applies low pass filter
        # When working with sequential loop data, need to output the filter state and add it to the next element in the loop
        # Which requires different ApplyFilter functions 
        filteredMixedSignal, filterStateInPhase = self.filter.ApplyFilterSequential(self.signal*self.reference, filterState=filterStateInPhase)
        filteredMixedSignalPhaseShifted, filterStateOutOfPhase = self.filter.ApplyFilterSequential(self.signal*self.referencePhaseShifted, filterState=filterStateOutOfPhase)
        quadratureMagnitude = numpy.sqrt(filteredMixedSignal**2 + filteredMixedSignalPhaseShifted**2)
        quadraturePhase = numpy.arctan2(filteredMixedSignalPhaseShifted, filteredMixedSignal)
        return quadratureMagnitude, quadraturePhase, filterStateInPhase, filterStateOutOfPhase       
    



def RunLockInAmplifierTest():
    # Set up signal
    signalFrequency = 27.0e9
    meanSignalPower = 3e-8
    signalAmplitude = meanSignalPower * numpy.sqrt(2)
    signalTimePeriod = 1/signalFrequency
    maxTime = 1e-3
    numberOfPeriods = maxTime*signalFrequency

    sampleRate = 1e11
    times = numpy.arange(0, numberOfPeriods*signalTimePeriod, step=1/sampleRate)
    
    signalSin = numpy.sin(2*numpy.pi*signalFrequency*times) * signalAmplitude
    
    # Add noise to signal
    temperature = 1 # K
    antennaBandwidth = 51.096e6
    lengthInSamples = len(signalSin)
    lowAntennaFreq = 26.975e9
    noise = SPF.GenerateVoltageNoise(sampleRate = sampleRate,
                                     temp = temperature,
                                     resistance = 73.2,
                                     antennaBandwidth = antennaBandwidth,
                                     numSamples = lengthInSamples,
                                     antennaLowCutoff = lowAntennaFreq
                                     )
    print("RMS Noise Pre Filter:", numpy.sqrt(numpy.mean(noise**2)))
    noisySignal = signalSin + noise
    
    # Set up reference
    referenceFrequency = 27.0e9
    referencePhaseShift = 0.0*numpy.pi
    referenceAmplitude = 1
    
    # Produce reference and 90 degree phase shifted reference
    referenceSin = numpy.sin(2*numpy.pi*referenceFrequency*times + referencePhaseShift) * referenceAmplitude
    referenceCos = numpy.cos(2*numpy.pi*referenceFrequency*times + referencePhaseShift) * referenceAmplitude
    
    # Set up filter parameters
    cutoffs = [1e4] 
    cutoffLabels = ["10 kHz"]

    # Apply lock-in amplifier for range of cutoffs
    magnitudesAndPhases = []
    for i in range(len(cutoffs)):
        cutoff = cutoffs[i]
        order = 5
        lowPassFilter = SPF.ChebyLowPass(cutoff, order, sampleRate)
        lockInAmpInstance = LockInAmp(noisySignal, referenceSin, referenceCos, lowPassFilter)
        magnitudesAndPhases.append(lockInAmpInstance.ProcessQuadrature())
        
        
    # Plot multiple cutoffs on same graph
    figMultiCutoff, axMultiCutoff = pyplot.subplots(nrows=2, ncols=1, figsize=[12,12])
    for i in range(len(cutoffs)):
        axMultiCutoff[0].plot(times, magnitudesAndPhases[i][0], label=cutoffLabels[i])
        axMultiCutoff[1].plot(times, magnitudesAndPhases[i][1])
    axMultiCutoff[0].set_ylabel("Voltage (V)")
    axMultiCutoff[1].set_xlabel("Time (s)")
    axMultiCutoff[1].set_ylabel("Phase")
    figMultiCutoff.suptitle("Sine wave signal")
    figMultiCutoff.legend()
    figMultiCutoff.tight_layout()
    return 0



if __name__ == "__main__":
    RunLockInAmplifierTest()
    