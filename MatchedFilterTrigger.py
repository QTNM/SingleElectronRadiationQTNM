import numpy
import matplotlib.pyplot as pyplot
import SpectrumAnalyser
from scipy.signal import correlate
import SignalProcessingFunctions as SPF


def GenerateChirpSignal(times, startFrequency, frequencyGradient, phaseOffset=0):
    rootMinusOne = complex(0,1)
    startAngularFrequency = startFrequency * 2 * numpy.pi
    chirpConstant = frequencyGradient * 2 * numpy.pi
    complexChirp = numpy.exp(rootMinusOne * (phaseOffset + startAngularFrequency * times + chirpConstant*times**2/2)) 
    return numpy.real(complexChirp)


def TestMatchedFilterTrigger():
    # Test chirp correlation for two different chirps
    sampleRate = 200e6
    times = numpy.arange(0, 1e-3, step = 1/sampleRate)
    signalAmplitude = 3.075179595089654e-08 * 2**0.5
    signal = signalAmplitude * GenerateChirpSignal(times, 20.2e6, 3e8, 0)
    template = GenerateChirpSignal(times, 20.1e6, 3e8, 0)
    
    
    # Add some noise to the data 
    k_B = 1.38064852e-23
    temperature = 4 # K
    antennaBandwidth = 1e6
    antennaLowCutoff = 20e6
    kTdB = k_B * temperature * antennaBandwidth 
    print("kTdB:", kTdB)
    resistance = 73.2
    noise = SPF.GenerateVoltageNoise(sampleRate = sampleRate,
                                     numSamples = len(times),
                                     temp = temperature,
                                     resistance = resistance,
                                     antennaLowCutoff = antennaLowCutoff,
                                     antennaBandwidth = antennaBandwidth)
    signal += noise
    
    fullTimes = numpy.arange(-1e-3+1/sampleRate,1e-3, step = 1/sampleRate)
    correlation = correlate(template, signal, mode='full') # Seemingly need to put template first to get correct time order
    print(len(times))
    print(len(correlation))
        
    frequencyGradients = numpy.arange(2e8, 4e8, step=1e5)
    peakCorrelationsGradient = []
    for i in range(len(frequencyGradients)):
        template = GenerateChirpSignal(times, 20.2e6, frequencyGradients[i], 0)
        correlation = correlate(template, signal, mode='full')
        peakCorrelationsGradient.append(numpy.amax(correlation))
    
        
    startFrequencies = numpy.arange(20e6,21e6,step=0.02e6)
    peakCorrelationsStart = []
    for i in range(len(startFrequencies)):
        template = GenerateChirpSignal(times, startFrequencies[i], 3e8, 0)
        correlation = correlate(signal, template, mode='full')
        peakCorrelationsStart.append(numpy.amax(correlation))
    
    
    
    figTestCorrelation, axTestCorrelation = pyplot.subplots(1, 1, figsize = [18,8])
    axTestCorrelation.plot(fullTimes, correlation)
    axTestCorrelation.set_xlabel("Time (s)")
    axTestCorrelation.set_ylabel("Correlation")
    
    
    SpectrumAnalyser.ApplyFFTAnalyser(times,
                                      signal, # Add together to plot both on same axis kind of
                                      lowSearchFreq=20e6,
                                      highSearchFreq=21e6,
                                      nTimeSplits=10, 
                                      nFreqBins=40)
    
    
    figPeakCorrelations, axPeakCorrelations = pyplot.subplots(1, 1, figsize = [18,8])
    axPeakCorrelations.plot(frequencyGradients, peakCorrelationsGradient) #/numpy.amax(peakCorrelationsGradient)*100)
    axPeakCorrelations.set_xlabel("Frequency Gradient (Hz/s)")
    axPeakCorrelations.set_ylabel("Peak Correlation")
    
    
    figPeakCorrelationsStartFreq, axPeakCorrelationsStartFreq = pyplot.subplots(1, 1, figsize = [18,8])
    axPeakCorrelationsStartFreq.plot(startFrequencies, peakCorrelationsStart) #/numpy.amax(peakCorrelationsStart)*100)
    axPeakCorrelationsStartFreq.set_xlabel("Start Frequency (Hz)")
    axPeakCorrelationsStartFreq.set_ylabel("Peak Correlation")
    return 0


def TriggerStatisticsTest():
    # Generate a bunch of signals within expected frequency gradient/start frequency ranges (maybe antenna bandwidth limited for the start frequencies)
    numberOfSignals = 300
    numberOfNoises = numberOfSignals
    startFrequencyBounds = [19e6, 21e6]
    frequencyGradientBounds = [3.29e8, 3.61e8]
    
    sampleRate = 200e6
    times = numpy.arange(0, 1e-3, step = 1/sampleRate)
    signalAmplitude = 3.075179595089654e-08 * 2**0.5
    k_B = 1.38064852e-23
    temperature = 4 # K
    antennaBandwidth = 1e6
    antennaLowCutoff = 20e6
    resistance = 73.2
   
    template = GenerateChirpSignal(times, 20.1e6, 3.45e8, 0)
    
    
    triggerThresholds = numpy.linspace(0.0005, 0.005, num=20)
    correctSignalDetectionRates = []
    falseAlarmRates = []
    for t in range(len(triggerThresholds)):
        triggerThreshold = triggerThresholds[t]
        triggerDecisions = []
        for i in range(numberOfSignals):
            randomStartFrequency = numpy.random.uniform(startFrequencyBounds[0], startFrequencyBounds[1])
            randomFrequencyGradient = numpy.random.uniform(frequencyGradientBounds[0], frequencyGradientBounds[1])
            randomPhase = numpy.random.uniform(0,2*numpy.pi)
            signal = signalAmplitude * GenerateChirpSignal(times, randomStartFrequency, randomFrequencyGradient, randomPhase)
            noise = SPF.GenerateVoltageNoise(sampleRate = sampleRate,
                                             numSamples = len(times),
                                             temp = temperature,
                                             resistance = resistance,
                                             antennaLowCutoff = antennaLowCutoff,
                                             antennaBandwidth = antennaBandwidth,
                                             hidePrints = True)
            signal += noise
            peakCorrelation = numpy.amax( correlate(signal, template) )
            if peakCorrelation > triggerThreshold:
                triggerDecisions.append(True)
            else:
                triggerDecisions.append(False)
           
        noiseTriggerDecisions = []
        for i in range(numberOfNoises):
            noise = SPF.GenerateVoltageNoise(sampleRate = sampleRate,
                                             numSamples = len(times),
                                             temp = temperature,
                                             resistance = resistance,
                                             antennaLowCutoff = antennaLowCutoff,
                                             antennaBandwidth = antennaBandwidth,
                                             hidePrints = True)
            peakCorrelation = numpy.amax( correlate(noise, template) )
            if peakCorrelation > triggerThreshold:
                noiseTriggerDecisions.append(True)
            else:
                noiseTriggerDecisions.append(False)
        
                
        correctSignalDetectionRate = numpy.count_nonzero(triggerDecisions) / numberOfSignals
        print("Correct signal detection rate:", correctSignalDetectionRate)
        correctSignalDetectionRates.append(correctSignalDetectionRate)
        
        falseAlarmRate = numpy.count_nonzero(noiseTriggerDecisions) / numberOfNoises
        falseAlarmRates.append(falseAlarmRate)
    
    figDetectionRates, axDetectionRates = pyplot.subplots(1, 1, figsize=[18,8])
    axDetectionRates.plot(triggerThresholds, correctSignalDetectionRates)
    axDetectionRates.set_xlabel("Trigger Threshold")
    axDetectionRates.set_ylabel("Correct Signal Detection Rate")
    
    figFAR, axFAR = pyplot.subplots(1, 1, figsize=[18,8])
    axFAR.plot(triggerThresholds, falseAlarmRates)
    axFAR.set_xlabel("Trigger Threshold")
    axFAR.set_ylabel("False Alarm Rate")    
    
    figPCDvsFAR, axPCDvsFAR = pyplot.subplots(1, 1, figsize = [18,8])
    axPCDvsFAR.plot(falseAlarmRates, correctSignalDetectionRates)
    axPCDvsFAR.set_xlabel("False Alarm Rate")
    axPCDvsFAR.set_ylabel("Probability of correct detection")
    return 0


if __name__ == "__main__":
    # TestMatchedFilterTrigger()
    TriggerStatisticsTest()


















