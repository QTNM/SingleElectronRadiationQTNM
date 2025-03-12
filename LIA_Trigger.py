import numpy
import SpectrumAnalyser
import SignalProcessingFunctions as SPF
from scipy import optimize
import random
import matplotlib.pyplot as pyplot

from scipy.signal import find_peaks
import pandas as pd



    
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
    
    
def FindBroadnessOfPeak(data, peakIndex, relativeHeight = 0.5):
    # start from peak, head forwards and backwards, when you find the first value below peak height*relativeHeight, stop and say the one before it is the width
    # if the peak hits the edge of the data before it finds anything, use just the other half and double, if it hits both sides, use the full width of the data?
    foundWidth = False
    foundForwardsWidth = False
    foundBackwardsWidth =False 
    upperSideCutOff = False
    lowerSideCutOff = False
    i=1
    upperWidth = 0
    lowerWidth = 0
    
    while foundWidth == False:
        # Look forwards first
        if foundForwardsWidth == False:
            if peakIndex+i > len(data)-1:
                upperSideCutOff = True
                foundForwardsWidth = True
            else:
                if data[peakIndex+i] < data[peakIndex]*relativeHeight:
                    upperWidth = i # samples
                    foundForwardsWidth = True
        # Then look backwards
        if foundBackwardsWidth == False:
            if peakIndex-i < 0:
                lowerSideCutOff = True
                foundBackwardsWidth = True
            else:
                if data[peakIndex-i] < data[peakIndex]*relativeHeight:
                    lowerWidth = i # samples
                    foundBackwardsWidth = True
        
        # if both forwards and backwards have been found, stop iterating, otherwise, iterate up i by 1
        if foundForwardsWidth and foundBackwardsWidth:
            foundWidth = True
        i+=1
    # Then process the sides and check whether the sides have been cut off, change width calculation accordingly.
    if upperSideCutOff and lowerSideCutOff:
        #print("Both sides cut off, returning full length of data")
        return len(data)
    elif upperSideCutOff:
        #print("Upper side cut off, returning twice lower width")
        return lowerWidth*2
    elif lowerSideCutOff:
        #print("Lower side cut off, returning twice upper width")
        return upperWidth*2
    else:
        return upperWidth+lowerWidth
        
    
    
    
    
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
    
    
    if broadnessArray is not None:
        biggestPeakBroadness = FindBroadnessOfPeak(data, biggestPeakIndex, relativeHeight = 0.75)
        broadnessArray.append(biggestPeakBroadness)
        
    
    
    return biggestMaxMean
    
    









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





def GetSignalLIAMeansForTrigger(numberOfTests, sampleRate, times, LIAStartFrequencies, LIAFrequencyGradients, LIALowPassCutoff, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth, LIAStartTimes = []):
    LIAMeans = []
    for i in range(numberOfTests):
        if i == numberOfTests//2:
            print("50%")
        elif i == numberOfTests*3//4:
            print("75%")
        elif i == numberOfTests//4:
            print("25%")
        elif i == numberOfTests:
            print("100%")
        signal = GenerateRandomChirpSignal(sampleRate, times, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth)  
        
        if i == 1:
            print("Signal RMS:", SPF.GetRMS(signal))


        # Hack to fix the times being wrong now the signal is being generated in a slightly different way that sometimes means the length is shorter than it was before.
        if len(signal) != len(times):
            if len(signal) > len(times):
                signal = signal[:len(times)]
            else:
                times = times[:len(signal)]

        currentTestMeans = []
        for s in range(len(LIAStartFrequencies)):
            # print("Start:", LIAStartFrequencies[s])
            
            # Select start time for this start frequency
            if LIAStartTimes != []:
                currentStartTime = LIAStartTimes[s]
            
            for f in range(len(LIAFrequencyGradients)): # For each lock-in amplifier, get the mean response to the signal
                # print("Grad:", LIAFrequencyGradients[f])
                
                # If there are no start times specified, just assume all start at t=0 and use the old method
                if LIAStartTimes == []:
                    lockInResponse = SpectrumAnalyser.SweepingLockInAmplifier(signal,
                                                                              times,
                                                                              LIAStartFrequencies[s],
                                                                              LIAFrequencyGradients[f],
                                                                              LIALowPassCutoff,
                                                                              sampleRate,
                                                                              showGraphs = False,
                                                                              filterType="Scipy Single Pole IIR")  # lock-in amplifier mean      
                    
                # Otherwise, use the new one and for each start frequency have a time offset in the same order
                else:
                    lockInResponse = SpectrumAnalyser.SweepingLockInAmplifierTimeShifted(
                                                                              signal,
                                                                              times,
                                                                              LIAStartFrequencies[s],
                                                                              LIAFrequencyGradients[f],
                                                                              LIALowPassCutoff,
                                                                              sampleRate,
                                                                              showGraphs = False,
                                                                              filterType="Scipy Single Pole IIR",
                                                                              startTime=currentStartTime,
                                                                              LIAReferenceLength=0.001
                                                                              )  # lock-in amplifier mean      
                    
                
                # currentTestMeans.append( numpy.mean( lockInResponse ) )
                ### Swapped it out for the maxmean approach to match what I do with my hardware data
                roughSampleRateConversion = int(125e6/500e3) # convert from internal clock speed to sampled data clock speed
                currentTestMeans.append( CalculateBiggestMaxMeanOfNPeaks(lockInResponse[::roughSampleRateConversion], nPeaks=5, peakWidth=50, distance=25, printGraphs = False, broadnessArray = None, label=None) )
                #
        LIAMeans.append( numpy.amax(currentTestMeans) )
    # print(LIAMeans)
    
    figSignalLIACheck, axSignalLIACheck = pyplot.subplots(1, 1, figsize=[16,8])
    axSignalLIACheck.plot(lockInResponse[0:10000])
    axSignalLIACheck.set_title("Signal LIA Response Check")
    
    return LIAMeans




def GetNoiseLIAMeansForTrigger(numberOfTests, sampleRate, times, LIAStartFrequencies, LIAFrequencyGradients, LIALowPassCutoff, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth, LIAStartTimes = []):
    LIAMeans = []
    for i in range(numberOfTests):
        if i == numberOfTests//2:
            print("Noise 50%")
        elif i == numberOfTests*3//4:
            print("Noise 75%")
        elif i == numberOfTests//4:
            print("Noise 25%")
        elif i == numberOfTests:
            print("Noise 100%")
        
        
        
        
        # noise = SPF.GenerateVoltageNoise(sampleRate = sampleRate,
        #                                  numSamples = len(times),
        #                                  temp = temperature,
        #                                  resistance = resistance,
        #                                  antennaLowCutoff = antennaLowCutoff,
        #                                  antennaBandwidth = antennaBandwidth,
        #                                  hidePrints = True)
        
        # noise = SPF.GenerateWhiteVoltageNoise(sampleRate, len(times), 290e-3, antennaLowCutoff, antennaBandwidth, True)
        # noise = upsampledNoise[int(noiseIndexes[100]*125e6/1000) : int((noiseIndexes[100]+1)*125e6/1000) ]
        noise = SPF.GenerateAdderSpectrumNoise(times[-1], sampleRate, noiseRMS=0.290)
        
        
        if len(noise) != len(times):
            if len(noise) > len(times):
                noise = noise[:len(times)]
            else:
                times = times[:len(noise)]
        ### Apply clipping and boost ###
        # noise = numpy.clip(noise, -0.291*0.5, 0.291*0.5)

        currentTestMeans = []
        for s in range(len(LIAStartFrequencies)):
            # print("Start:", LIAStartFrequencies[s])
            
            # Select start time for this start frequency
            if LIAStartTimes != []:
                currentStartTime = LIAStartTimes[s]
            
            for f in range(len(LIAFrequencyGradients)): # For each lock-in amplifier, get the mean response to the signal
                # print("Grad:", LIAFrequencyGradients[f])
                
                # If there are no start times specified, just assume all start at t=0 and use the old method
                if LIAStartTimes == []:
                    lockInResponse = SpectrumAnalyser.SweepingLockInAmplifier(noise,
                                                                              times,
                                                                              LIAStartFrequencies[s],
                                                                              LIAFrequencyGradients[f],
                                                                              LIALowPassCutoff,
                                                                              sampleRate,
                                                                              showGraphs = False,
                                                                              filterType="Scipy Single Pole IIR")  # lock-in amplifier mean 
                    
                # Otherwise, use the new one and for each start frequency have a time offset in the same order
                else:
                    lockInResponse = SpectrumAnalyser.SweepingLockInAmplifierTimeShifted(
                                                                              noise,
                                                                              times,
                                                                              LIAStartFrequencies[s],
                                                                              LIAFrequencyGradients[f],
                                                                              LIALowPassCutoff,
                                                                              sampleRate,
                                                                              showGraphs = False,
                                                                              filterType="Scipy Single Pole IIR",
                                                                              startTime=currentStartTime,
                                                                              LIAReferenceLength=0.001
                                                                              )  # lock-in amplifier mean   
                
                
                # currentTestMeans.append( numpy.mean( lockInResponse ) )
                ### Swapped it out for the maxmean approach to match what I do with my hardware data
                roughSampleRateConversion = int(125e6/500e3) # convert from internal clock speed to sampled data clock speed
                currentTestMeans.append( CalculateBiggestMaxMeanOfNPeaks(lockInResponse[::roughSampleRateConversion], nPeaks=5, peakWidth=50, distance=25, printGraphs = False, broadnessArray = None, label=None) )
                # 
        LIAMeans.append( numpy.amax(currentTestMeans) )
    # print(LIAMeans)
    
    figNoiseLIACheck, axNoiseLIACheck = pyplot.subplots(1, 1, figsize=[16,8])
    axNoiseLIACheck.plot(lockInResponse[0:10000])
    axNoiseLIACheck.set_title("Noise LIA Response Check")
    
    
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
    #print(SPF.GetRMS(signal))
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




    # Simulate Axial Ocillation
    axialFrequency = 6872216 # 89 degree electron in 20cm trap
    randomAxialPhase = numpy.random.rand()*numpy.pi*2
    #for i in range(len(signal)):
        
    # signal[i] *= 0.5 * numpy.sin(2 * numpy.pi * times[i] * axialFrequency + randomAxialPhase) + 1

    
    # assume position moves sinusoidally, with max original voltage occurring in the middle.
    trapsize = 0.2
    
    distanceSquared =  0.02**2 + (trapsize * numpy.sin(2*numpy.pi * axialFrequency * times + randomAxialPhase))**2 # pythagoras on 2cm away from antenna radially and axial distance oscillation
    
    distanceSquaredEffect = distanceSquared / 0.004 # Normalise such that the drop off is nothing at the centre (1) and maintains the r^2 relationship
    
    signal *= 1/distanceSquaredEffect

    # figTestSig, axTestSig = pyplot.subplots(1, 1, figsize=[16,8])
    # axTestSig.plot(times, signal)
    # axTestSig.set_xlabel("Time (s)")
    # axTestSig.set_ylabel("Voltage (V)")
    # axTestSig.set_xlim(0,1e-6)
    


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
    
    ### Third time shifting method (random full signal within interval) ###
    # indexStart = 0
    # indexEnd = len(times)
    # randomIndex = numpy.random.randint(indexStart, indexEnd//2) # shift within the first half so the whole signal is still present 
    # if randomIndex != 0:
    #     zerosForShift = numpy.zeros(randomIndex)
    #     signal = numpy.insert(signal,0,zerosForShift)
    #     signal = signal[:-randomIndex]
        # print("Signal check:", signal[randomIndex-2:randomIndex+3])
    
    
    
    ##### CUSTOM SIGNAL LIMITING TO SIMULATE THE SHORTER SIGNALS WITH LONGER NOISE #####
    signal[int(len(signal)/10):] = 0
    
    
    # noise = SPF.GenerateVoltageNoise(sampleRate = sampleRate,
    #                                  numSamples = len(times),
    #                                  temp = temperature,
    #                                  resistance = resistance,
    #                                  antennaLowCutoff = antennaLowCutoff,
    #                                  antennaBandwidth = antennaBandwidth,
    #                                  hidePrints = True)
    
    # noise = SPF.GenerateWhiteVoltageNoise(sampleRate, len(times), 290e-3, antennaLowCutoff, antennaBandwidth, True)
    
    # noise = upsampledNoise[int(noiseIndexes[100]*125e6/1000) : int((noiseIndexes[100]+1)*125e6/1000) ]
    noise = SPF.GenerateAdderSpectrumNoise(times[-1], sampleRate, noiseRMS=0.290)
    
    #print("Noise shape, signal shape:", noise.shape, signal.shape)
    
    #print("SignalRMS:",SPF.GetRMS(signal))
    #print("NoiseRMS:",SPF.GetRMS(noise))
    if len(noise) != len(signal):
        if len(noise) > len(signal):
            noise = noise[:len(signal)]
        else:
            signal = signal[:len(noise)]
    # print("Signal/Noise ratio (RMS):", GetRMS(signal)/GetRMS(noise))
    signal += noise  
    
    #print("Signal+Noise RMS",SPF.GetRMS(signal))
    ### Apply clipping ###
    # signal = numpy.clip(signal, -0.291*0.5, 0.291*0.5)
    
    #figTest, axTest = pyplot.subplots(1, 1, figsize=[16,8])
    #axTest.plot(signal)
    #axTest.set_title("Test signal plot")
    
    return signal

def AddNoiseBeforeStart(signal, sampleRate, temperature, resistance, antennaLowCutoff, antennaBandwidth):
    extensionTime = 1e-3
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
    # maxTime = 1e-3   # seconds
    
    k_B = 1.38064852e-23
    temperature = 8 # K   
    resistance = 73.2
    antennaBandwidth = 125e6/2-2 # 125e6/2-2
    # actually for the noise through the voltage adder the bandwidth is like 15MHz ish
    antennaLowCutoff = 1e0
    
    numberOfTests = 500 # 1000
    # times = numpy.arange(0, maxTime, step=timeStep)
    LIALowPassCutoff = 1.5e4# 15915.5 # 0.2e6
    print("LIA Time constant:",1/(2*numpy.pi*LIALowPassCutoff))
    
    #noiseRMSForTrigger = numpy.sqrt(4*k_B*temperature*resistance*LIALowPassCutoff)
    noiseRMSForTrigger = 290e-3
    print("Noise RMS:", noiseRMSForTrigger)
    # triggerThresholds = numpy.linspace(0*noiseRMSForTrigger, 8*noiseRMSForTrigger, num=60000)
    triggerThresholds = numpy.logspace(-10,numpy.log(8*noiseRMSForTrigger), num=60000)
    # LIAStartFrequencies = numpy.linspace(20.1e6, 20.9e6, num=5)
    numberOfLIAs = 1
    LIAStartFrequencies = numpy.repeat([3.5e5],numberOfLIAs)# [3.5e5]
    LIAStartTimes = numpy.arange(0,0.001, step = 0.001/numberOfLIAs)
    LIAStartTimes = LIAStartTimes.tolist()
    print(LIAStartTimes)
    #print(LIAStartFrequencies)
    LIAFrequencyGradients = [3.45e8]# [3.45e8]
    print("LIA Time Coverage:",2*LIALowPassCutoff / LIAFrequencyGradients[0]*1000,"ms")
    startFrequencyBounds = [3.5e5, 3.5e5] # These are the signal start frequency bounds, 1MHz for our frequency range
    #startFrequencyBounds = [300000,300001]
    frequencyGradientBounds = [3.45e8, 3.45e8]# [3.45e8, 3.45e8] # from magnetic trap calculations
    #frequencyGradientBounds = [3.45e8,3.45000001e8]
    # signalAmplitude = 3.075179595089654e-08 * 2**0.5
    signalAmplitude = 11e-3*numpy.sqrt(2)
    signalAmplitude *= 1.08*1 # some phased antenna improvements
    signalAmplitude *= 1.10
    
    print("SNR Pre LIA:",signalAmplitude/numpy.sqrt(2) / noiseRMSForTrigger )

    # ### For loading noise from data, keep an index of all the possible noise indexes ###
    # noiseIndexes = numpy.arange(0,3000,1)
    # noiseFile = pd.read_csv("D:/Project Stuff/Noise Characterisation Data/PureNoiseFromAdderForCharacterisation.csv", skiprows=range(7), header=None).to_numpy()[:,1]
    # maxNumberOfNoiseSignals = numpy.floor(len(noiseFile)/sampleRate)
    # print("Max noise signals =", maxNumberOfNoiseSignals)
    # noiseSampleRate = 500000
    # ### Need to upsample noise by zero stuffing then low pass filtering ###
    # ### I know my sample rate was 125e6 for the lock-in and 500e3 for the noise data, so need to make noise sample 250x longer
    # ### this means inserting 249 zeros between each value, then low pass filtering at (250)kHz
    # print(noiseFile[0:250])
    # upsampledNoise = numpy.zeros(len(noiseFile)*250)
    # upsampledNoise[::250] = noiseFile
    # print(upsampledNoise[0:501])
    # lowPassForNoiseUpsampling = SPF.ButterLowPass(cutoff=250000, order=1, sampleRate=125e6)
    # upsampledNoise = lowPassForNoiseUpsampling.ApplyFilter(upsampledNoise)
    # print(upsampledNoise[0:501])
    # FFTBeforeUpsampling = SPF.FourierTransformWaveform(noiseFile[:1000000], numpy.arange(0, 1000000 / 500000, step = 1/500000))
    # FFTAfterUpsampling = SPF.FourierTransformWaveform(upsampledNoise[:250000000], numpy.arange(0, 250000000 / 125e6, step = 1/125e6))
    
    
    # figNoiseCheck, axNoiseCheck = pyplot.subplots(1, 1, figsize=[16,8])
    # axNoiseCheck.plot(FFTBeforeUpsampling[0][1:], FFTBeforeUpsampling[1][1:], color="b", label="Before", alpha=0.5)
    # axNoiseCheck.plot(FFTAfterUpsampling[0][1:], FFTAfterUpsampling[1][1:], color="r", label="After",alpha=0.5)
    # axNoiseCheck.set_xlim(-1000,300000)
    # axNoiseCheck.set_ylim(-1,100)
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
        signalLIAFitOutputMaxes = GetSignalLIAMeansForTrigger(numberOfTests, sampleRate, times, LIAStartFrequencies, LIAFrequencyGradients, LIALowPassCutoff, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth, LIAStartTimes = LIAStartTimes )
        noiseLIAFitOutputMaxes = GetNoiseLIAMeansForTrigger(numberOfTests, sampleRate, times, LIAStartFrequencies, LIAFrequencyGradients, LIALowPassCutoff, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth, LIAStartTimes = LIAStartTimes )
        
        
        correctSignalDetectionRates = []
        falseAlarmRates = []
        for triggerThreshold in triggerThresholds:
            signalTriggerDecisions = []
            noiseTriggerDecisions = []
            
            # Need to adjust this for varying numbers of LIAs
            
            # if numberOfLIAs > 1:
            #     for i in range(len(signalLIAFitOutputMaxes)):
            #         # if at least one of the triggers activates, call it a correct detection
            #         currentTriggers = signalLIAFitOutputMaxes[i*numberOfLIAs:(i+1)*numberOfLIAs]
            #         if any(currentTrigger > triggerThreshold for currentTrigger in currentTriggers): # if any of the responses are above the trigger threshold, count a trigger
            #             signalTriggerDecisions.append(True)
            #         else:
            #             signalTriggerDecisions.append(False)
                
            #         currentNoiseTriggers = noiseLIAFitOutputMaxes[i*numberOfLIAs:(i+1)*numberOfLIAs]
            #         if any(currentNoiseTrigger > triggerThreshold for currentNoiseTrigger in currentNoiseTriggers):
            #             noiseTriggerDecisions.append(True)
            #         else:
            #             noiseTriggerDecisions.append(False)
                    
                    

            # print("Using old method for triggers")
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
                    
                    
            # print("Length of signalLIAFitOutputMaxes:", len(signalLIAFitOutputMaxes))
            # print("Len trigger decisions:", len(signalTriggerDecisions))
            # print("Number of tests:",numberOfTests)
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
        axVariedTimeROC.plot(variedSignalLengthFARs[i],variedSignalLengthPCDs[i], label = maxTimeLabels[i]+"s", linestyle="-")
    axVariedTimeROC.legend(loc="lower right")
    axVariedTimeROC.set_xlabel("False Alarm Rate")
    axVariedTimeROC.set_ylabel("Correct Signal Detection Rate")


    print("CHECK MISSED SIGNAL:")
    numberOfTests = 300
    # ViewMissedTriggersFor90PercentDetection(numberOfTests, sampleRate, times, LIAStartFrequencies, LIAFrequencyGradients, LIALowPassCutoff, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth)


