import numpy
import SpectrumAnalyser
import SignalProcessingFunctions as SPF
from scipy import optimize
import random
import matplotlib.pyplot as pyplot




def GetCorrectSignalDetectionRate(numberOfTests, triggerThreshold, sampleRate, times, LIAStartFrequencies, LIAFrequencyGradients, LIALowPassCutoff, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth):
    triggerDecisions = []
    for i in range(numberOfTests):
        signal = GenerateRandomChirpSignal(sampleRate, times, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth)
        
        peakCorrelations = []
        # get lock-in amplifier values
        for s in range(len(LIAStartFrequencies)):
            print("Start:", LIAStartFrequencies[s])
            for f in range(len(LIAFrequencyGradients)): # For each lock-in amplifier, get the mean response to the signal
                print("Grad:", LIAFrequencyGradients[f])
                lockInResponse = SpectrumAnalyser.SweepingLockInAmplifier(signal,
                                                                          times,
                                                                          LIAStartFrequencies[s],
                                                                          LIAFrequencyGradients[f],
                                                                          LIALowPassCutoff,
                                                                          sampleRate,
                                                                          showGraphs = False) 
                # lock-in amplifier mean 
                peakCorrelations.append( numpy.mean( lockInResponse ) )
        peakCorrelation = numpy.amax(peakCorrelations)
        print(peakCorrelation)
        if peakCorrelation > triggerThreshold:
            triggerDecisions.append(True)
        else:
            triggerDecisions.append(False)
            
    correctSignalDetectionRate = numpy.count_nonzero(triggerDecisions) / numberOfTests
    print("Correct signal detection rate:", correctSignalDetectionRate)
    return correctSignalDetectionRate





def GetSignalLIAMeansForTrigger(numberOfTests, sampleRate, times, LIAStartFrequencies, LIAFrequencyGradients, LIALowPassCutoff, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth):
    LIAMeans = []
    for i in range(numberOfTests):
        # print("Test Number:", i+1)
        signal = GenerateRandomChirpSignal(sampleRate, times, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth)  

        if i == 300:
            print("Signal RMS:", SPF.GetRMS(signal))

        currentTestMeans = []
        for s in range(len(LIAStartFrequencies)):
            # print("Start:", LIAStartFrequencies[s])
            for f in range(len(LIAFrequencyGradients)): # For each lock-in amplifier, get the mean response to the signal
                # print("Grad:", LIAFrequencyGradients[f])
                lockInResponse = SpectrumAnalyser.SweepingLockInAmplifier(signal,
                                                                          times,
                                                                          LIAStartFrequencies[s],
                                                                          LIAFrequencyGradients[f],
                                                                          LIALowPassCutoff,
                                                                          sampleRate,
                                                                          showGraphs = False)  # lock-in amplifier mean      
                currentTestMeans.append( numpy.mean( lockInResponse ) )
        LIAMeans.append( numpy.amax(currentTestMeans) )
    # print(LIAMeans)
    return LIAMeans




def GetNoiseLIAMeansForTrigger(numberOfTests, sampleRate, times, LIAStartFrequencies, LIAFrequencyGradients, LIALowPassCutoff, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth):
    LIAMeans = []
    for i in range(numberOfTests):
        # print("Test Number:", i+1)
        noise = SPF.GenerateVoltageNoise(sampleRate = sampleRate,
                                         numSamples = len(times),
                                         temp = temperature,
                                         resistance = resistance,
                                         antennaLowCutoff = antennaLowCutoff,
                                         antennaBandwidth = antennaBandwidth,
                                         hidePrints = True)
        
        
        ### Apply clipping and boost ###
        noise = numpy.clip(noise, -0.001e-6, 0.001e-6)

        currentTestMeans = []
        for s in range(len(LIAStartFrequencies)):
            # print("Start:", LIAStartFrequencies[s])
            for f in range(len(LIAFrequencyGradients)): # For each lock-in amplifier, get the mean response to the signal
                # print("Grad:", LIAFrequencyGradients[f])
                lockInResponse = SpectrumAnalyser.SweepingLockInAmplifier(noise,
                                                                          times,
                                                                          LIAStartFrequencies[s],
                                                                          LIAFrequencyGradients[f],
                                                                          LIALowPassCutoff,
                                                                          sampleRate,
                                                                          showGraphs = False)  # lock-in amplifier mean 
                currentTestMeans.append( numpy.mean( lockInResponse ) )
        LIAMeans.append( numpy.amax(currentTestMeans) )
    # print(LIAMeans)
    return LIAMeans




def FlatPiecewiseCurve(x, x0, a, b):
    result = numpy.piecewise(x, [x<x0], [a,b])
    return result

def MatchFlatPiecewiseCurve(times, LIAOutput):
    optimisedParameters, estimatedCovariance = optimize.curve_fit(FlatPiecewiseCurve, times, LIAOutput, p0=[0.5,1e-8,1e-8])
    return optimisedParameters                    
    


def GenerateRandomChirpSignal(sampleRate, times, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth):
    randomStartFrequency = numpy.random.uniform(startFrequencyBounds[0], startFrequencyBounds[1])
    randomFrequencyGradient = numpy.random.uniform(frequencyGradientBounds[0], frequencyGradientBounds[1])
    randomPhase = numpy.random.uniform(0,2*numpy.pi)
    signal = signalAmplitude * SPF.GenerateChirpSignal(times, randomStartFrequency, randomFrequencyGradient, randomPhase)

    ### Apply bandpass filter to signal ###
    # settlingTime = 1/antennaBandwidth*6
    # settlingTimeInSamples = int(settlingTime*sampleRate) 
    # settlingTimes = numpy.arange(-settlingTime, 0, step=1/sampleRate)
    # settlingTimeSignal = signalAmplitude * SPF.GenerateChirpSignal(settlingTimes, randomStartFrequency, randomFrequencyGradient, randomPhase)
    # signal = numpy.insert(signal, 0, settlingTimeSignal)
    # bandwidthFilter = SPF.ButterBandPass(cutoffLow = antennaLowCutoff,
    #                          cutoffHigh = antennaLowCutoff+antennaBandwidth,
    #                          order = 5,
    #                          sampleRate = sampleRate)
    # signal = bandwidthFilter.ApplyFilter(signal)
    # # Restore just the signal after the filter has settled
    # signal = signal[settlingTimeInSamples:]

    ### Simulate time shifting ###
    # halfwayIndex = len(signal)//2
    # cutBeginning = random.random() < 0.5
    # indecesToCut = random.randint(0, halfwayIndex)
    # if cutBeginning == True:
    #     signal[:indecesToCut] *= 0 # remove the first chunk of the signal to simulate a signal that starts after the lock-ins reset
    # else:
    #     signal[(len(signal)-indecesToCut):] *= 0 # remove the last chunk of the signal to simulate a signal that started before the lock-ins reset.

    ### Aternative Time Shifting ###
    # emptyArray = numpy.zeros(len(times) + 2*len(times)-2)
    # middleStartPoint = len(times)-1
    # middleEndPoint = -len(times)+1
    # signalStartTime = numpy.random.randint(len(emptyArray)-len(times))
    # emptyArray[signalStartTime:signalStartTime+len(times)] += signal
    # signal = emptyArray[middleStartPoint : middleEndPoint]   
    
    ### Shift signal delay ###
    # zeros = numpy.zeros(6000)
    # signal = numpy.concatenate((zeros,signal))[:len(signal)]
    
    
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
    # print("Signal/Noise ratio (RMS):", GetRMS(signal)/GetRMS(noise))
    signal += noise  
    
    ### Apply clipping ###
    signal = numpy.clip(signal, -0.001e-6, 0.001e-6)
    
    return signal

def AddNoiseBeforeStart(signal, sampleRate, temperature, resistance, antennaLowCutoff, antennaBandwidth):
    extensionTime = 3e-3
    extensionLengthInSamples = int(sampleRate*extensionTime)
    noiseToAdd = SPF.GenerateVoltageNoise(sampleRate = sampleRate,
                                          numSamples = extensionLengthInSamples,
                                          temp = temperature,
                                          resistance = resistance,
                                          antennaLowCutoff = antennaLowCutoff,
                                          antennaBandwidth = antennaBandwidth,
                                          hidePrints = True)
    extendedSignal = numpy.insert(signal, 0, noiseToAdd)
    extendedTimes = numpy.arange(-extensionTime, 0, step = 1/sampleRate)
    extendedTimes = numpy.append(extendedTimes, times)
    return extendedSignal, extendedTimes


def GetSignalLIAFitMaxOutputs(numberOfTests, sampleRate, times, LIAStartFrequencies, LIAFrequencyGradients, LIALowPassCutoff, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth):
    maxParameterABOutputs = []
    for i in range(numberOfTests):
        if i==0:
            print("0%")
        if numberOfTests/(i+1) == 2:
            print("50%")
        if numberOfTests/(i+1) == 1:
            print("100%")
        signal = GenerateRandomChirpSignal(sampleRate, times, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth)
        originalSignalLength = len(times)
        signal, extendedTimes = AddNoiseBeforeStart(signal, sampleRate, temperature, resistance, antennaLowCutoff, antennaBandwidth)

        allOptimisedParameters = []
        for s in range(len(LIAStartFrequencies)):
            for f in range(len(LIAFrequencyGradients)): # For each lock-in amplifier, response to the signal
                lockInResponse = SpectrumAnalyser.SweepingLockInAmplifier(signal,
                                                                          extendedTimes,
                                                                          LIAStartFrequencies[s],
                                                                          LIAFrequencyGradients[f],
                                                                          LIALowPassCutoff,
                                                                          sampleRate,
                                                                          showGraphs = False)
                lockInResponse = lockInResponse[-originalSignalLength:] # Take the signal after the filter settles on noise
                allOptimisedParameters.append ( MatchFlatPiecewiseCurve(times, lockInResponse)[1:] ) # for simple flat piecewise, x0,a,b -> a,b
        
        maxParameterABOutputs.append( numpy.amax(allOptimisedParameters) )
        # print(maxParameterABOutputs)                   
    return maxParameterABOutputs


def GetNoiseLIAFitMaxOutputs(numberOfTests, sampleRate, times, LIAStartFrequencies, LIAFrequencyGradients, LIALowPassCutoff, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth):
    maxParameterABOutputs = []
    for i in range(numberOfTests):
        if i==0:
            print("0%")
        if numberOfTests/(i+1) == 2:
            print("50%")
        if numberOfTests/(i+1) == 1:
            print("100%")
        signal = SPF.GenerateVoltageNoise(sampleRate, len(times), temperature, resistance, antennaLowCutoff, antennaBandwidth, hidePrints = True)
        originalSignalLength = len(times)
        signal, extendedTimes = AddNoiseBeforeStart(signal, sampleRate, temperature, resistance, antennaLowCutoff, antennaBandwidth)
       
        allOptimisedParameters = []
        for s in range(len(LIAStartFrequencies)):
            for f in range(len(LIAFrequencyGradients)): # For each lock-in amplifier, response to the signal
                lockInResponse = SpectrumAnalyser.SweepingLockInAmplifier(signal,
                                                                          extendedTimes,
                                                                          LIAStartFrequencies[s],
                                                                          LIAFrequencyGradients[f],
                                                                          LIALowPassCutoff,
                                                                          sampleRate,
                                                                          showGraphs = False) 
                lockInResponse = lockInResponse[-originalSignalLength:] # Take the signal after the filter settles on noise
                allOptimisedParameters.append ( MatchFlatPiecewiseCurve(times, lockInResponse)[1:] ) # for simple flat piecewise, x0,a,b -> a,b
        maxParameterABOutputs.append( numpy.amax(allOptimisedParameters) )
    return maxParameterABOutputs

    






if __name__ == "__main__":
    
    # Initialise Parameters  
    sampleRate = 125e6
    timeStep = 1/sampleRate
    maxTime = 1e-3   # seconds
    
    k_B = 1.38064852e-23
    temperature = 8 # K   
    resistance = 73.2
    antennaBandwidth = 15e6
    antennaLowCutoff = 1e0
    
    numberOfTests = 1
    times = numpy.arange(0, maxTime, step=timeStep)
    LIALowPassCutoff = 8000# 0.2e6
    noiseRMSForTrigger = numpy.sqrt(4*k_B*temperature*resistance*LIALowPassCutoff)
    print("Noise RMS:", noiseRMSForTrigger)
    triggerThresholds = numpy.linspace(0*noiseRMSForTrigger, 0.01*noiseRMSForTrigger, num=60000)
    # LIAStartFrequencies = numpy.linspace(20.1e6, 20.9e6, num=5)
    LIAStartFrequencies = [500000]
    LIAFrequencyGradients = [0] # [3.45e8]
    startFrequencyBounds = [350000,350001]
    #startFrequencyBounds = [300000,300001]
    frequencyGradientBounds = [3.45e8,3.45000001e8] #[3.29e8, 3.61e8]
    signalAmplitude = 3.075179595089654e-08 * 2**0.5
    signalAmplitude *= 1 # some phased antenna improvements
    

    
    
    
    # Trigger based on means
    # SignalLIAMeans = GetSignalLIAMeansForTrigger(numberOfTests, sampleRate, times, LIAStartFrequencies, LIAFrequencyGradients, LIALowPassCutoff, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth )
    # NoiseLIAMeans = GetNoiseLIAMeansForTrigger(numberOfTests, sampleRate, times, LIAStartFrequencies, LIAFrequencyGradients, LIALowPassCutoff, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth )
    

    # Do multiple tests for different lengths of signal
    maxTimes = numpy.linspace(1e-3,1e-3,num=1)
    maxTimeLabels = []
    for i in range(len(maxTimes)):
        maxTimeLabels.append(maxTimes[i])
        maxTimeLabels[i] = "{:.1E}".format(maxTimeLabels[i])
    variedSignalLengthFARs = numpy.empty( (len(maxTimes)) , dtype = 'object')
    variedSignalLengthPCDs = numpy.empty( (len(maxTimes)) , dtype = 'object')




    for t in range(len(maxTimes)):
        # print("Number:",t+1)
        times = numpy.arange(0, maxTimes[t], step = 1/sampleRate)
    
        # # Trigger based on curve fit
        # signalLIAFitOutputMaxes = GetSignalLIAFitMaxOutputs(numberOfTests, sampleRate, times, LIAStartFrequencies, LIAFrequencyGradients, LIALowPassCutoff, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth)
        # noiseLIAFitOutputMaxes = GetNoiseLIAFitMaxOutputs(numberOfTests, sampleRate, times, LIAStartFrequencies, LIAFrequencyGradients, LIALowPassCutoff, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth)
       
        # Trigger based on means
        signalLIAFitOutputMaxes = GetSignalLIAMeansForTrigger(numberOfTests, sampleRate, times, LIAStartFrequencies, LIAFrequencyGradients, LIALowPassCutoff, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth )
        noiseLIAFitOutputMaxes = GetNoiseLIAMeansForTrigger(numberOfTests, sampleRate, times, LIAStartFrequencies, LIAFrequencyGradients, LIALowPassCutoff, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth )
        
        
        correctSignalDetectionRates = []
        falseAlarmRates = []
        for triggerThreshold in triggerThresholds:
            signalTriggerDecisions = []
            noiseTriggerDecisions = []
            
            for triggerValue in signalLIAFitOutputMaxes:
                if triggerValue > triggerThreshold:
                    signalTriggerDecisions.append(True)
                else:
                    signalTriggerDecisions.append(False)   
            
            for triggerValue in noiseLIAFitOutputMaxes:
                if triggerValue > triggerThreshold:
                    noiseTriggerDecisions.append(True)
                else:
                    noiseTriggerDecisions.append(False)   
            # for mean in SignalLIAMeans:
            #     # print(mean)
            #     if mean > triggerThreshold:
            #         signalTriggerDecisions.append(True)
            #     else:
            #         signalTriggerDecisions.append(False)
            
            # for mean in NoiseLIAMeans:
            #     # print(mean)
            #     if mean > triggerThreshold:
            #         noiseTriggerDecisions.append(True)
            #     else:
            #         noiseTriggerDecisions.append(False)
            
            correctSignalDetectionRate = numpy.count_nonzero(signalTriggerDecisions) / numberOfTests
            # print("Correct signal detection rate:", correctSignalDetectionRate)
            correctSignalDetectionRates.append(correctSignalDetectionRate)
            
            falseAlarmRate = numpy.count_nonzero(noiseTriggerDecisions) / numberOfTests
            # print("False alarm rate:",falseAlarmRate)
            falseAlarmRates.append(falseAlarmRate)
        variedSignalLengthFARs[t] = falseAlarmRates
        variedSignalLengthPCDs[t] = correctSignalDetectionRates
        
        


    figPCD, axPCD = pyplot.subplots(1, 1, figsize=[16,8])
    axPCD.plot(triggerThresholds / noiseRMSForTrigger, correctSignalDetectionRates)
    axPCD.set_xlabel("Trigger Threshold (/noiseRMS)")
    axPCD.set_ylabel("Correct Signal Detection Rate")

    figFAR, axFAR = pyplot.subplots(1, 1, figsize=[16,8])
    axFAR.plot(triggerThresholds / noiseRMSForTrigger, falseAlarmRates)
    axFAR.set_xlabel("Trigger Threshold (/noiseRMS)")
    axFAR.set_ylabel("False Alarm Rate")
    
    figROC, axROC = pyplot.subplots(1, 1, figsize=[16,8])
    axROC.plot(falseAlarmRates, correctSignalDetectionRates, '--')
    axROC.set_xlabel("False Alarm Rate")
    axROC.set_ylabel("Correct Signal Detection Rate")
    
    figVariedTimeROC, axVariedTimeROC = pyplot.subplots(1, 1, figsize=[16,8])
    for i in range(len(maxTimes)):
        axVariedTimeROC.plot(variedSignalLengthFARs[i],variedSignalLengthPCDs[i], label = maxTimeLabels[i]+"s")
    axVariedTimeROC.legend(loc="lower right")
    axVariedTimeROC.set_xlabel("False Alarm Rate")
    axVariedTimeROC.set_ylabel("Correct Signal Detection Rate")


    print("CHECK MISSED SIGNAL:")
    numberOfTests = 300
    # ViewMissedTriggersFor90PercentDetection(numberOfTests, sampleRate, times, LIAStartFrequencies, LIAFrequencyGradients, LIALowPassCutoff, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth)


