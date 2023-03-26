import numpy
import matplotlib.pyplot as pyplot
import SpectrumAnalyser
from scipy.signal import correlate
import SignalProcessingFunctions as SPF
import random
import Constants


def TestMatchedFilterTrigger():
    # Test chirp correlation for two different chirps
    sampleRate = 200e6
    times = numpy.arange(0, 1e-3, step = 1/sampleRate)
    signalAmplitude = 3.075179595089654e-08 * 2**0.5
    signal = signalAmplitude * SPF.GenerateChirpSignal(times, 20.2e6, 3.45e8, 0)
    template = SPF.GenerateChirpSignal(times, 20.1e6, 3e8, 0)
    
    
    # Add some noise to the data 
    temperature = 4 # K
    antennaBandwidth = 1e6
    antennaLowCutoff = 20e6
    kTdB = Constants.k_B * temperature * antennaBandwidth 
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
    correlation = correlate(template, signal, mode='full') # Need to put template first to get correct time order
    print(len(times))
    print(len(correlation))
        
    frequencyGradients = numpy.arange(3.29e8, 3.61e8, step=1e6)
    peakCorrelationsGradient = []
    for i in range(len(frequencyGradients)):
        template = SPF.GenerateChirpSignal(times, 20.2e6, frequencyGradients[i], 0)
        correlation = correlate(template, signal, mode='full')
        peakCorrelationsGradient.append(numpy.amax(correlation))
    
        
    startFrequencies = numpy.arange(20e6,21e6,step=0.02e6)
    peakCorrelationsStart = []
    for i in range(len(startFrequencies)):
        template = SPF.GenerateChirpSignal(times, startFrequencies[i], 3e8, 0)
        correlation = correlate(signal, template, mode='full')
        peakCorrelationsStart.append(numpy.amax(correlation))
    
    
    
    figTestCorrelation, axTestCorrelation = pyplot.subplots(1, 1, figsize = [18,8])
    axTestCorrelation.plot(fullTimes, correlation)
    axTestCorrelation.set_xlabel("Time (s)")
    axTestCorrelation.set_ylabel("Correlation")
    
    
    SpectrumAnalyser.ApplyFFTAnalyser(times,
                                      signal, 
                                      lowSearchFreq=20e6,
                                      highSearchFreq=21e6,
                                      nTimeSplits=10, 
                                      nFreqBins=40)
    
    
    figPeakCorrelations, axPeakCorrelations = pyplot.subplots(1, 1, figsize = [18,8])
    axPeakCorrelations.plot(frequencyGradients, peakCorrelationsGradient/numpy.amax(peakCorrelationsGradient)*100)
    axPeakCorrelations.set_xlabel("Frequency Gradient (Hz/s)")
    axPeakCorrelations.set_ylabel("Peak Correlation")
    
    
    figPeakCorrelationsStartFreq, axPeakCorrelationsStartFreq = pyplot.subplots(1, 1, figsize = [18,8])
    axPeakCorrelationsStartFreq.plot(startFrequencies, peakCorrelationsStart/numpy.amax(peakCorrelationsStart)*100)
    axPeakCorrelationsStartFreq.set_xlabel("Start Frequency (Hz)")
    axPeakCorrelationsStartFreq.set_ylabel("Peak Correlation")
    return 0



def GetCorrectSignalDetectionRate(numberOfTests, triggerThreshold, templates, sampleRate, times, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth):
    triggerDecisions = []
    for i in range(numberOfTests):
        randomStartFrequency = numpy.random.uniform(startFrequencyBounds[0], startFrequencyBounds[1])
        randomFrequencyGradient = numpy.random.uniform(frequencyGradientBounds[0], frequencyGradientBounds[1])
        randomPhase = numpy.random.uniform(0,2*numpy.pi)
        signal = signalAmplitude * SPF.GenerateChirpSignal(times, randomStartFrequency, randomFrequencyGradient, randomPhase)
        noise = SPF.GenerateVoltageNoise(sampleRate = sampleRate,
                                         numSamples = len(times),
                                         temp = temperature,
                                         resistance = resistance,
                                         antennaLowCutoff = antennaLowCutoff,
                                         antennaBandwidth = antennaBandwidth,
                                         hidePrints = True)
        signal += noise
        
        peakCorrelations = []
        for template in templates:
            peakCorrelations.append( numpy.amax( correlate(signal, template) ) )
        peakCorrelation = numpy.amax(peakCorrelations)
        
        if peakCorrelation > triggerThreshold:
            triggerDecisions.append(True)
        else:
            triggerDecisions.append(False)
            
    correctSignalDetectionRate = numpy.count_nonzero(triggerDecisions) / numberOfTests
    print("Correct signal detection rate:", correctSignalDetectionRate)
    return correctSignalDetectionRate
    
    
    
def GetFalseAlarmRate(numberOfTests, triggerThreshold, templates, sampleRate, times, temperature, resistance, antennaLowCutoff, antennaBandwidth):
    noiseTriggerDecisions = []
    for i in range(numberOfTests):
        noise = SPF.GenerateVoltageNoise(sampleRate = sampleRate,
                                         numSamples = len(times),
                                         temp = temperature,
                                         resistance = resistance,
                                         antennaLowCutoff = antennaLowCutoff,
                                         antennaBandwidth = antennaBandwidth,
                                         hidePrints = True)

        peakCorrelations = []
        for template in templates:
            peakCorrelations.append( numpy.amax( correlate(noise, template) ) )
        peakCorrelation = numpy.amax(peakCorrelations)


        if peakCorrelation > triggerThreshold:
            noiseTriggerDecisions.append(True)
        else:
            noiseTriggerDecisions.append(False)
    
    falseAlarmRate = numpy.count_nonzero(noiseTriggerDecisions) / numberOfTests
    print(falseAlarmRate)
    return falseAlarmRate


def GetSignalMFScoreForTrigger(numberOfTests, templates, sampleRate, times, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth):
    matchedFilterScores = []
    for i in range(numberOfTests):
        if i==0:
            print("0%")
        if numberOfTests/(i+1) == 2:
            print("50%")
        if numberOfTests/(i+1) == 1:
            print("100%")
            
            
        randomStartFrequency = numpy.random.uniform(startFrequencyBounds[0], startFrequencyBounds[1])
        randomFrequencyGradient = numpy.random.uniform(frequencyGradientBounds[0], frequencyGradientBounds[1])
        randomPhase = numpy.random.uniform(0,2*numpy.pi)
        signal = signalAmplitude * SPF.GenerateChirpSignal(times, randomStartFrequency, randomFrequencyGradient, randomPhase)
        
        # # Apply bandpass filter to signal
        # settlingTime = 1/antennaBandwidth*6
        # settlingTimeInSamples = int(settlingTime*sampleRate) 
        # settlingTimes = numpy.arange(-settlingTime, 0, step=1/sampleRate)
        # settlingTimeSignal = signalAmplitude * SPF.GenerateChirpSignal(settlingTimes, randomStartFrequency, randomFrequencyGradient, randomPhase)
        # signal = numpy.insert(signal, 0, settlingTimeSignal)
        # bandwidthFilter = SPF.ButterBandPass(cutoffLow = antennaLowCutoff,
        #                          cutoffHigh = antennaLowCutoff+antennaBandwidth,
        #                          order = 1,
        #                          sampleRate = sampleRate)
        # signal = bandwidthFilter.ApplyFilter(signal)
        # # Restore just the signal after the filter has settled
        # signal = signal[settlingTimeInSamples:]
        
        # # Simulate time shifting
        # halfwayIndex = len(signal)//2
        # cutBeginning = random.random() < 0.5
        # indecesToCut = random.randint(0, halfwayIndex)
        # if cutBeginning == True:
        #     signal[:indecesToCut] *= 0 # remove the first chunk of the signal to simulate a signal that starts after the lock-ins reset
        # else:
        #     signal[(len(signal)-indecesToCut):] *= 0 # remove the last chunk of the signal to simulate a signal that started before the lock-ins reset.
        
        noise = SPF.GenerateVoltageNoise(sampleRate = sampleRate,
                                         numSamples = len(times),
                                         temp = temperature,
                                         resistance = resistance,
                                         antennaLowCutoff = antennaLowCutoff,
                                         antennaBandwidth = antennaBandwidth,
                                         hidePrints = True)
        # print("sig:",len(signal))
        # print(len(noise))
        # signal = signal[:-1]

        if len(signal) > len(noise):
            signal = signal[:len(noise)]
        else:
            noise = noise[:len(signal)]
            
        # print("SNR:", GetRMS(signal) / GetRMS(noise))   
        signal += noise 
        

        
        peakCorrelations = []
        for template in templates:
            peakCorrelations.append( numpy.amax( correlate(signal, template) ) )
        matchedFilterScores.append( numpy.amax(peakCorrelations) )
    return matchedFilterScores


def GetNoiseMFScoreForTrigger(numberOfTests, templates, sampleRate, times, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth):
    matchedFilterScores = []
    for i in range(numberOfTests):
        if i==0:
            print("0%")
        if numberOfTests/(i+1) == 2:
            print("50%")
        if numberOfTests/(i+1) == 1:
            print("100%")
        noise = SPF.GenerateVoltageNoise(sampleRate = sampleRate,
                                         numSamples = len(times),
                                         temp = temperature,
                                         resistance = resistance,
                                         antennaLowCutoff = antennaLowCutoff,
                                         antennaBandwidth = antennaBandwidth,
                                         hidePrints = True) 

        peakCorrelations = []
        for template in templates:
            peakCorrelations.append( numpy.amax( correlate(noise, template) ) )
        matchedFilterScores.append( numpy.amax(peakCorrelations) )
    return matchedFilterScores



def TriggerStatisticsTest():
    # Generate a bunch of signals within expected frequency gradient/start frequency ranges (maybe antenna bandwidth limited for the start frequencies)
    numberOfTests = 300
    # startFrequencyBounds = [20e6, 21e6-345000]
    startFrequencyBounds = [20e6, 20e6]
    frequencyGradientBounds = [3.45e8, 3.4500001e8]
    
    sampleRate = 125e6
    times = numpy.arange(0, 1e-3, step = 1/sampleRate)
    signalAmplitude = 3.075179595089654e-08 * 2**0.5
    signalAmplitude *= 1.0 # Moderate increase from phased array
    temperature = 8 # K
    antennaBandwidth = 15e6
    antennaLowCutoff = 1e0
    resistance = 73.2
    
    # startFrequencies = numpy.arange(19.5e6,20.5e6, step = 0.2e6)

    triggerThresholds = numpy.linspace(0.00005, 0.03, num=3000)
    correctSignalDetectionRates = []
    falseAlarmRates = []
    
    # templateStartFrequencies = numpy.linspace(20.1e6, 20.9e6, num=5)
    templateStartFrequencies = numpy.linspace(20e6, 20e6, num = 1)
    templateFrequencyGradients = numpy.linspace(3.45e8,3.45e8, num=1)
    templates = []
    for i in range(len(templateStartFrequencies)):
        for j in range(len(templateFrequencyGradients)):
            templates.append( SPF.GenerateChirpSignal(times, templateStartFrequencies[i], templateFrequencyGradients[j], 0) )




    # Do multiple tests for different lengths of signal
    maxTimes = numpy.linspace(1e-3,1e-3,num=1)
    maxTimeLabels = []
    for i in range(len(maxTimes)):
        maxTimeLabels.append(maxTimes[i])
        maxTimeLabels[i] = "{:.1E}".format(maxTimeLabels[i])
    variedSignalLengthFARs = numpy.empty( (len(maxTimes)) , dtype = 'object')
    variedSignalLengthPCDs = numpy.empty( (len(maxTimes)) , dtype = 'object')

    for t in range(len(maxTimes)):
        times = numpy.arange(0, maxTimes[t], step = 1/sampleRate)
        print(len(times))

        correctSignalDetectionRates = []
        falseAlarmRates = []
        SignalMFScore = GetSignalMFScoreForTrigger(numberOfTests,
                                                   templates,
                                                   sampleRate,
                                                   times, 
                                                   startFrequencyBounds, 
                                                   frequencyGradientBounds,
                                                   signalAmplitude,
                                                   temperature,
                                                   resistance,
                                                   antennaLowCutoff,
                                                   antennaBandwidth)
        NoiseMFScore = GetNoiseMFScoreForTrigger(numberOfTests,
                                                   templates,
                                                   sampleRate,
                                                   times, 
                                                   startFrequencyBounds, 
                                                   frequencyGradientBounds,
                                                   signalAmplitude,
                                                   temperature,
                                                   resistance,
                                                   antennaLowCutoff,
                                                   antennaBandwidth)

        for triggerThreshold in triggerThresholds:
            signalTriggerDecisions = []
            noiseTriggerDecisions = []
            
            for MFScore in SignalMFScore:
                # print(mean)
                if MFScore > triggerThreshold:
                    signalTriggerDecisions.append(True)
                else:
                    signalTriggerDecisions.append(False)
            correctSignalDetectionRate = numpy.count_nonzero(signalTriggerDecisions) / numberOfTests
            # print("Correct signal detection rate:", correctSignalDetectionRate)
            correctSignalDetectionRates.append(correctSignalDetectionRate)
            
            for MFScore in NoiseMFScore:
                # print(mean)
                if MFScore > triggerThreshold:
                    noiseTriggerDecisions.append(True)
                else:
                    noiseTriggerDecisions.append(False)
            falseAlarmRate = numpy.count_nonzero(noiseTriggerDecisions) / numberOfTests
            # print("False alarm rate:",falseAlarmRate)
            falseAlarmRates.append(falseAlarmRate)
            
        variedSignalLengthFARs[t] = falseAlarmRates
        variedSignalLengthPCDs[t] = correctSignalDetectionRates
           

    
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
    
    
    figVariedTimeROC, axVariedTimeROC = pyplot.subplots(1, 1, figsize=[16,8])
    for i in range(len(maxTimes)):
        axVariedTimeROC.plot(variedSignalLengthFARs[i],variedSignalLengthPCDs[i], label = maxTimeLabels[i]+"s")
    axVariedTimeROC.legend(loc="lower right")
    axVariedTimeROC.set_xlabel("False Alarm Rate")
    axVariedTimeROC.set_ylabel("Correct Signal Detection Rate")
    
    
    
    return 0



def MatchedFilterFrequencyResolution(hidePrints = False):
    sampleRate = 125e6
    times = numpy.arange(0, 0.001, 1/sampleRate)
    signalAmplitude = 3.075179595089654e-08 * 2**0.5
    startFrequency = 500000
    frequencyGradient = 3.45e8
    signal = signalAmplitude * SPF.GenerateChirpSignal(times, startFrequency, frequencyGradient)
    
    temperature = 8
    resistance = 73.2
    antennaLowCutoff = 0.1
    antennaBandwidth = 15e6
    
    noise = SPF.GenerateVoltageNoise(sampleRate = sampleRate,
                                     numSamples = len(times),
                                     temp = temperature,
                                     resistance = resistance,
                                     antennaLowCutoff = antennaLowCutoff,
                                     antennaBandwidth = antennaBandwidth,
                                     hidePrints = True)
    if len(noise) != len(signal):
        if len(noise) > len(signal):
            noise = noise[:-1]
        else:
            signal = signal[:-1]
    
    if hidePrints == False:
        print("Signal/Noise ratio (RMS):", SPF.GetRMS(signal)/SPF.GetRMS(noise))
    signal += noise
    
    template = 1 * SPF.GenerateChirpSignal(times, startFrequency, frequencyGradient)
    matchedFilterResult = correlate(signal, template)
    
    xShifts = numpy.linspace(-1e-3, 1e-3, num=len(matchedFilterResult))
    if hidePrints == False:
        print("Lengths:")
        print(len(signal))
        print(len(template))
        print(len(matchedFilterResult))
        print(len(xShifts))
        figMF, axMF = pyplot.subplots(1, 1, figsize = [16,8])
        axMF.plot(xShifts, matchedFilterResult)
    
    # Now repeat for many templates of varying frequency
    matchedFilterPeaks = []
    noisePeaks = []
    frequencies = numpy.arange(480000, 520000, step=200)
    for frequency in frequencies:
        template = 1 * SPF.GenerateChirpSignal(times, frequency, frequencyGradient)
        matchedFilterResult = correlate(signal, template)
        matchedFilterPeaks.append(numpy.amax(matchedFilterResult))
        noiseResult = correlate(noise, template)
        noisePeaks.append(numpy.amax(noiseResult))
    
    if hidePrints == False:
        figFreqRes, axFreqRes = pyplot.subplots(1, 1, figsize=[16,8])
        axFreqRes.plot(frequencies, matchedFilterPeaks)
        axFreqRes.set_xlabel("Frequency (Hz)")
        axFreqRes.set_ylabel("Matched Filter Response")
    
    timeConstantInSamples = 1e-7 * sampleRate
    movingAverageFilter = SPF.MovingAverage(timeConstantInSamples)
    
    smoothedMatchedFilterPeaks = movingAverageFilter.ApplyFilter(matchedFilterPeaks)
    smoothedNoisePeaks = movingAverageFilter.ApplyFilter(noisePeaks)
    
    if hidePrints == False:
        print(len(smoothedMatchedFilterPeaks))

        figFreqResSmoothed, axFreqResSmoothed = pyplot.subplots(1, 1, figsize=[16,8])
        axFreqResSmoothed.plot(frequencies-500000, smoothedMatchedFilterPeaks, label="Signal+Noise")
        #axFreqResSmoothed.plot(frequencies, smoothedNoisePeaks, label="Noise Only")
        #axFreqResSmoothed.legend()
        axFreqResSmoothed.set_xlabel(r"$\Delta f$ (Hz)")
        axFreqResSmoothed.set_ylabel("Matched Filter Response")
        
    if numpy.argmax(smoothedNoisePeaks) > 3*len(frequencies)/4:
        figFreqResSmoothed, axFreqResSmoothed = pyplot.subplots(1, 1, figsize=[16,8])
        axFreqResSmoothed.plot(frequencies-500000, smoothedMatchedFilterPeaks, label="Signal+Noise")
        #axFreqResSmoothed.plot(frequencies, smoothedNoisePeaks, label="Noise Only")
        #axFreqResSmoothed.legend()
        axFreqResSmoothed.set_xlabel(r"$\Delta f$ (Hz)")
        axFreqResSmoothed.set_ylabel("Matched Filter Response")
        # figFreqResSmoothed.close()
    # print(frequencies[numpy.argmax(smoothedMatchedFilterPeaks)])
        
    return frequencies[numpy.argmax(smoothedMatchedFilterPeaks)]


if __name__ == "__main__":
    # TestMatchedFilterTrigger()
    # TriggerStatisticsTest()
    frequencies = numpy.arange(480000, 520000, step=200) # make sure this matches frequencyResolution function's frequencies
    frequenciesAtMaximum = []
    for i in range(1000):
        frequenciesAtMaximum.append(MatchedFilterFrequencyResolution(hidePrints=True))
    
    figAltFreqRes, axAltFreqRes = pyplot.subplots(1, 1, figsize=[16,8])
    axAltFreqRes.hist(frequenciesAtMaximum)

















