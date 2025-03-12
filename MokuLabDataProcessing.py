import pandas as pd
import matplotlib.pyplot as pyplot
import numpy
from scipy import optimize
#pyplot.rcdefaults()
pyplot.rcParams.update({'font.size': 18})
from scipy.signal import find_peaks


# Some functions from LIA trigger code for fitting two flat regions to see if this beats mean 
def FlatPiecewiseCurve(x, x0, a, b):
    result = numpy.piecewise(x, [x<x0], [a,b])
    return result

def MatchFlatPiecewiseCurve(times, LIAOutput):
    optimisedParameters, estimatedCovariance = optimize.curve_fit(FlatPiecewiseCurve, times, LIAOutput, p0=[0.5,0.001,0.001])
    return optimisedParameters  

def FlatPeakCurve(x, x0, x1, a, b, c):
    # print(type(x))
    result = numpy.piecewise(x, [ x<=x0 , numpy.logical_and(x0<x,x<x1) , x>=x1 ], [a,b,c])
    return result

def MatchFlatPeakCurve(times, LIAOutput):
    optimisedParameters, estimatedCovariance = optimize.curve_fit(FlatPeakCurve, times, LIAOutput, p0=[0.35,0.65,0.001,0.002,0.001])
    return optimisedParameters


def GetRMS(inputArray):
    sumOfSquares = 0
    for i in range(len(inputArray)):
        sumOfSquares+=inputArray[i]**2

    return numpy.sqrt( sumOfSquares/len(inputArray) )



# def InvestigatePerformance():

#     timesSignalHigherInvestigate = []
    
#     bestPerformingShift = 0
#     previousBestNumberBeatingNoise = 0
#     bestPerformingMeans = []
#     indexShifts = numpy.arange(0,stepLength*4,step=1012//92)
#     #indexShifts = [0]
    
#     print("Start time:", mokulabData2[0][0])
#     print("End time:", mokulabData2[0][stepLength])
    
#     for j in range(len(indexShifts)):
#         #
#         numberOfSignalBeatingNoise = 0
#         fitSignalBeatNoise = 0
#         fitPeakSignalBeatNoise = 0
        
#         signalMeans = []
#         noiseMeans = []
        
    
#         # For the matched sweeps, use every other chunk as the noise
#         for i in range(1000):
#             # Loop over 2000ms (1000 for signal chunks, 1000 for noise chunks)
#             startIndexSignal = 2*i*stepLength + indexShifts[j] # *2 for skipping every other chunk
#             startIndexNoise = 2*i*stepLength + indexShifts[j] + stepLength # the next chunk along?
#             stopIndexSignal = startIndexSignal + stepLength
#             stopIndexNoise = startIndexNoise + stepLength
            
#             times = ( mokulabData2[0][startIndexSignal : stopIndexSignal] - mokulabData2[0][startIndexSignal] ) *1000
#             times = times.to_numpy()
            
#             LIAOutputNoise = mokulabData2[1][startIndexNoise : stopIndexNoise]
#             LIAOutputSignalAndNoise = mokulabData2[1][startIndexSignal : stopIndexSignal]
            
#             currentMeanNoise = numpy.mean(LIAOutputNoise)
#             currentMeanSignalAndNoise = numpy.mean(LIAOutputSignalAndNoise)
#             if currentMeanSignalAndNoise > currentMeanNoise:
#                 numberOfSignalBeatingNoise+=1
#             signalMeans.append(currentMeanSignalAndNoise)
#             noiseMeans.append(currentMeanNoise)
         
          
#         # for i in range(1000):
#         #     # Loop over the first 1000ms for now.
#         #     startIndex = i*stepLength + indexShifts[j]
#         #     stopIndex = (i+1)*stepLength + indexShifts[j]
            
#         #     # times = mokulabData[0][i*stepLength : (i+1)*stepLength] * 1000 - mokulabData[0][i*stepLength] # Get times and shift to between 0 and ~1ms, converted into ms to match expected input for optimised curve
#         #     times = ( mokulabData[0][startIndex : stopIndex] - mokulabData[0][startIndex] ) * 1000
            
#         #     times = times.to_numpy()
#         #     LIAOutputNoise = mokulabData[1][startIndex : stopIndex]
#         #     LIAOutputSignalAndNoise = mokulabData2[1][startIndex : stopIndex]
            
#         #     currentMeanNoise = numpy.mean(LIAOutputNoise)
#         #     currentMeanSignalAndNoise = numpy.mean(LIAOutputSignalAndNoise)
#         #     if currentMeanSignalAndNoise > currentMeanNoise:
#         #         numberOfSignalBeatingNoise+=1
#         #     signalMeans.append(currentMeanSignalAndNoise)
#         #     noiseMeans.append(currentMeanNoise)
                
#         #     # # Try fitting with flat regions
#         #     # optimisedParametersNoise = MatchFlatPiecewiseCurve(times, LIAOutputNoise)
#         #     # optimisedParametersSignal = MatchFlatPiecewiseCurve(times, LIAOutputSignalAndNoise)
#         #     # noiseHeight = max(optimisedParametersNoise[1:])
#         #     # signalHeight = max(optimisedParametersSignal[1:])
            
#         #     # if noiseHeight < signalHeight:
#         #     #     fitSignalBeatNoise+=1
                
#         #     # # Fit with 3 flat regions for peak potential
#         #     # optimisedParametersNoise = MatchFlatPeakCurve(times, LIAOutputNoise)
#         #     # optimisedParametersSignal = MatchFlatPiecewiseCurve(times, LIAOutputSignalAndNoise)
#         #     # noiseHeight = max(optimisedParametersNoise[2:])
#         #     # signalHeight = max(optimisedParametersSignal[2:])
            
#         #     # if noiseHeight < signalHeight:
#         #     #     fitPeakSignalBeatNoise+=1
        
        
#         print("Number beating:", numberOfSignalBeatingNoise)
#         print("Previous best:", previousBestNumberBeatingNoise)
#         if numberOfSignalBeatingNoise > previousBestNumberBeatingNoise:
#             print("UPDATING BEST SHIFT")
#             bestPerformingShift = indexShifts[j]
#             previousBestNumberBeatingNoise = numberOfSignalBeatingNoise
            
#         print("Times signal mean higher than noise mean:", numberOfSignalBeatingNoise, "in 1000 attempts.")
#         signalMeans = numpy.array(signalMeans)
#         noiseMeans = numpy.array(noiseMeans)
#         thresholds = numpy.linspace(0.0,0.06, num=300)
#         signalsPassingThreshold = []
#         noisePassingThreshold = []
#         for threshold in thresholds:
#             signalsPassingThreshold.append( (signalMeans > threshold).sum() )
#             noisePassingThreshold.append( (noiseMeans > threshold).sum() )
    
    
#         timesSignalHigherInvestigate.append(numberOfSignalBeatingNoise)
#         print(len(timesSignalHigherInvestigate))
    
    
#     # There are like 5s worth of data, I take 1s each time, so I can go up to like 3999ms large scale shifts
#     # I only want even numbers though once I have the small scale shift
#     # so I want like 0, 2, 4, 6, 8, ...,3998 ms which is then *stepLength and +bestSmallShift
#     largeScaleShifts = numpy.arange(0, 2956, step=10) * stepLength + bestPerformingShift
#     timesSignalHigherInvestigateLong = []
#     bestPerformingLongShift = 0
#     previousBestNumberBeatingNoise = 0
#     bestPerformingMeans = []
    
#     for j in range(len(largeScaleShifts)):
#         # Need to change the times thing to loop properly and skip every other 1ms.
#         numberOfSignalBeatingNoise = 0 
#         signalMeans = []
#         noiseMeans = []
        
#         # For the matched sweeps, use every other chunk as the noise
#         for i in range(1000):
#             # Loop over 2000ms (1000 for signal chunks, 1000 for noise chunks)
#             startIndexSignal = 2*i*stepLength + largeScaleShifts[j] # *2 for skipping every other chunk
#             startIndexNoise = 2*i*stepLength + stepLength + largeScaleShifts[j] # the next chunk along
#             stopIndexSignal = startIndexSignal + stepLength
#             stopIndexNoise = startIndexNoise + stepLength
            
#             times = ( mokulabData2[0][startIndexSignal : stopIndexSignal] - mokulabData2[0][startIndexSignal] ) *1000
#             times = times.to_numpy()
            
#             LIAOutputNoise = mokulabData2[1][startIndexNoise : stopIndexNoise]
#             LIAOutputSignalAndNoise = mokulabData2[1][startIndexSignal : stopIndexSignal]
            
#             currentMeanNoise = numpy.mean(LIAOutputNoise)
#             currentMeanSignalAndNoise = numpy.mean(LIAOutputSignalAndNoise)
#             if currentMeanSignalAndNoise > currentMeanNoise:
#                 numberOfSignalBeatingNoise+=1
#             signalMeans.append(currentMeanSignalAndNoise)
#             noiseMeans.append(currentMeanNoise)
    
#         # print("Number beating:", numberOfSignalBeatingNoise)
#         # print("Previous best:", previousBestNumberBeatingNoise)
#         if numberOfSignalBeatingNoise > previousBestNumberBeatingNoise:
#             # print("UPDATING BEST SHIFT")
#             bestPerformingLongShift = largeScaleShifts[j]
#             previousBestNumberBeatingNoise = numberOfSignalBeatingNoise
            
#         # print("Times signal mean higher than noise mean:", numberOfSignalBeatingNoise, "in 1000 attempts.")
#         signalMeans = numpy.array(signalMeans)
#         noiseMeans = numpy.array(noiseMeans)
#         thresholds = numpy.linspace(0.0,0.06, num=300)
#         signalsPassingThreshold = []
#         noisePassingThreshold = []
#         for threshold in thresholds:
#             signalsPassingThreshold.append( (signalMeans > threshold).sum() )
#             noisePassingThreshold.append( (noiseMeans > threshold).sum() )
    
    
#         timesSignalHigherInvestigateLong.append(numberOfSignalBeatingNoise)
#         # print(len(timesSignalHigherInvestigateLong))
        
  
#     # Now that the best performing shift has been found, run again with that index shift
#     # For the matched sweeps, use every other chunk as the noise
#     numberOfSignalBeatingNoise = 0
#     fitSignalBeatNoise = 0
#     fitPeakSignalBeatNoise = 0
#     signalMeans = []
#     noiseMeans = []
#     for i in range(1000):
#         # Loop over 2000ms (1000 for signal chunks, 1000 for noise chunks)
#         startIndexSignal = 2*i*stepLength + bestPerformingLongShift # *2 for skipping every other chunk
#         startIndexNoise = 2*i*stepLength + bestPerformingLongShift + stepLength # the next chunk along
#         stopIndexSignal = startIndexSignal + stepLength
#         stopIndexNoise = startIndexNoise + stepLength
        
#         times = ( mokulabData2[0][startIndexSignal : stopIndexSignal] - mokulabData2[0][startIndexSignal] ) *1000
#         times = times.to_numpy()
        
#         LIAOutputNoise = mokulabData2[1][startIndexNoise : stopIndexNoise]
#         LIAOutputSignalAndNoise = mokulabData2[1][startIndexSignal : stopIndexSignal]
        
#         currentMeanNoise = numpy.mean(LIAOutputNoise)
#         currentMeanSignalAndNoise = numpy.mean(LIAOutputSignalAndNoise)
#         if currentMeanSignalAndNoise > currentMeanNoise:
#             numberOfSignalBeatingNoise+=1
#         signalMeans.append(currentMeanSignalAndNoise)
#         noiseMeans.append(currentMeanNoise)
     
    
#     print("Times signal mean higher than noise mean:", numberOfSignalBeatingNoise, "in 1000 attempts.")
#     signalMeans = numpy.array(signalMeans)
#     noiseMeans = numpy.array(noiseMeans)
#     thresholds = numpy.linspace(0.0,0.06, num=300)
#     signalsPassingThreshold = []
#     noisePassingThreshold = []
#     for threshold in thresholds:
#         signalsPassingThreshold.append( (signalMeans > threshold).sum() )
#         noisePassingThreshold.append( (noiseMeans > threshold).sum() )
    
#     print("Best Performing Shift:", bestPerformingShift)
#     print("Best Performing Long Shift:", bestPerformingLongShift)
    
    
#     figInv, axInv = pyplot.subplots(1, 1, figsize=[16,8])
#     axInv.plot(numpy.arange(1,len(indexShifts)+1), timesSignalHigherInvestigate)
#     axInv.set_xlabel("Index shift number")
    
#     figInv2, axInv2 = pyplot.subplots(1, 1, figsize=[16,8])
#     axInv2.plot(numpy.arange(1,len(largeScaleShifts)+1), timesSignalHigherInvestigateLong)
#     axInv2.set_xlabel("Long Index shift number")
    
    
    
    
#     signalsPassingThreshold = numpy.array(signalsPassingThreshold)
#     noisePassingThreshold = numpy.array(noisePassingThreshold)
#     figROC, axROC = pyplot.subplots(1, 1, figsize=[16,8])
#     axROC.plot(noisePassingThreshold / len(signalMeans), signalsPassingThreshold / len(signalMeans))
#     axROC.set_xlabel("False Alarm Rate")
#     axROC.set_ylabel("Correct Detection Rate")
#     #axROC.plot(numpy.arange(0,1,step=0.001), numpy.arange(0,1,step=0.001), "--")
    
#     # print("Times signal fit higher than noise fit:", fitSignalBeatNoise, "in 1000 attempts.")
#     # print("Times signal peak fit higher than noise fit:", fitPeakSignalBeatNoise, "in 1000 attempts.")
    

    
    
    
    
        
def FindIndexOfClosestToValue(array, value):
    if not isinstance(array, numpy.ndarray):
        array = numpy.array(array)
    differenceArray = array-value
    absoluteDifferenceArray = numpy.abs(differenceArray)
    closestValueIndex = numpy.argmin(absoluteDifferenceArray)
    return closestValueIndex
    

    
def GetMatchedFilterTriggerPerformance(fileName): # , triggerThresholds):

    signalData = pd.read_csv(fileName + "_Sig.csv", skiprows=range(7), header=None).to_numpy()
    pureNoiseData = pd.read_csv(fileName + "_Noise.csv", skiprows=range(7), header=None).to_numpy()
    
    # figDataPlot, axDataPlot = pyplot.subplots(3, 1, figsize=[16,24])
    # axDataPlot[0].set_title("Signal + Noise Repsonse")
    # axDataPlot[0].plot(signalData[:5001,0], signalData[:5001,1], color="blue")
    # axDataPlot[0].set_xlabel("Time (s)")
    # axDataPlot[0].set_ylabel("Voltage (V)")
    # axDataPlot[1].set_title("Pure Noise Response")
    # axDataPlot[1].plot(pureNoiseData[:5001,0], pureNoiseData[:5001,1],color="orange")
    # axDataPlot[1].set_xlabel("Time (s)")
    # axDataPlot[1].set_ylabel("Voltage (V)")
    # axDataPlot[2].set_title("Comparison Response")
    # axDataPlot[2].plot(signalData[:5001,0], signalData[:5001,1], color="blue")
    # axDataPlot[2].plot(pureNoiseData[:5001,0], pureNoiseData[:5001,1], color="orange")
    # axDataPlot[2].set_xlabel("Time (s)")
    # axDataPlot[2].set_ylabel("Voltage (V)")
    # figDataPlot.tight_layout()
    

    
    print("Number of signals:",len(find_peaks(signalData[:,2], height=0)[0]))
    
    sampleRate = 1/(signalData[1,0] - signalData[0,0])
    print("Sample Rate:",sampleRate)
    samplesIn2ms = int(0.002*sampleRate)
    print("Samples in 2ms:", samplesIn2ms)
    
    # position of first trigger position
    firstTriggerPositionSig = numpy.argmax(signalData[:samplesIn2ms,2])
    firstTriggerPositionNoise = numpy.argmax(pureNoiseData[:samplesIn2ms,2])
    startTriggerOffsetSig = firstTriggerPositionSig
    startTriggerOffsetNoise = firstTriggerPositionNoise
    # if firstTriggerPosition > 150:
    #     startTriggerOffset = firstTriggerPosition-150# is 150 less than the first trigger position if the first trigger position is greater than 150, just 0 otherwise.
    # else:
    #     startTriggerOffset = 0
    
    signalStartLocations = find_peaks(signalData[:,2], height=0)[0]
    noiseStartLocations = find_peaks(pureNoiseData[:,2], height=0)[0]
    
    numberOfSignals = len(signalStartLocations) # len(signalData)//samplesIn2ms # For sampling of 1MSps, 2ms = 2000 samples. For 0.5Msps, 2ms = 1000 samples
    numberOfNoise = len(noiseStartLocations)  # len(pureNoiseData)//samplesIn2ms
    print("Number of Signals:", numberOfSignals)
    print("Number of Noise:", numberOfNoise)
    numberOfSignals = min([numberOfSignals, numberOfNoise]) # limited by whichever has the lowest number of data points since the mokulab's data collection seems to be a little bit inconsistent on this front.
    

    baselineVoltageForThreshold = GetRMS(pureNoiseData[:,1])
    print("Baseline voltage for threshold:", baselineVoltageForThreshold)
    thresholds = numpy.arange(baselineVoltageForThreshold/30, baselineVoltageForThreshold*3, step = baselineVoltageForThreshold/1000)
    # thresholds = triggerThresholds
    
    triggerDecisionsSignal_ThresholdAverage = numpy.zeros( (len(thresholds), numberOfSignals) )
    triggerDecisionsNoise_ThresholdAverage = numpy.zeros( (len(thresholds), numberOfSignals) )
    
    triggerDecisionsSignal_ThresholdMax = numpy.zeros( (len(thresholds), numberOfSignals) )
    triggerDecisionsNoise_ThresholdMax = numpy.zeros( (len(thresholds), numberOfSignals) )
    
    for i in range(numberOfSignals): # -2 because some strange things were happening with the last 2, one was empty (makes more sense) and one was short (Presumably just the remainder of the data) - actually, it was just that not all the data had the same length, changed that so it should work now

        #timeChunk = signalData[ (startTriggerOffsetSig + i*samplesIn2ms) : (startTriggerOffsetSig + ((i+1)*samplesIn2ms)), 0]
        #signalDataChunk = signalData[ (startTriggerOffsetSig + i*samplesIn2ms) : (startTriggerOffsetSig + ((i+1)*samplesIn2ms)), 1 ]
        #testTriggerChunk = signalData[(startTriggerOffsetSig + i*samplesIn2ms) : (startTriggerOffsetSig + ((i+1)*samplesIn2ms)), 2 ]
        #noiseDataChunk = pureNoiseData[ (startTriggerOffsetNoise + i*samplesIn2ms) : (startTriggerOffsetNoise + ((i+1)*samplesIn2ms)), 1 ]
        timeChunk = signalData[ signalStartLocations[i] : signalStartLocations[i]+samplesIn2ms//2 , 0]
        signalDataChunk = signalData[ signalStartLocations[i] : signalStartLocations[i]+samplesIn2ms//2 , 1]
        testTriggerChunk = signalData[ signalStartLocations[i] : signalStartLocations[i]+samplesIn2ms//2 , 2]
        noiseDataChunk = pureNoiseData[ noiseStartLocations[i] : noiseStartLocations[i]+samplesIn2ms//2 , 1]
        
        testExample = True
        if testExample == True and i==1000:
            figTestExample, axTestExample = pyplot.subplots(2, 1, figsize=[16,16])
            axTestExample[0].plot(timeChunk, signalDataChunk)
            axTestExample[1].plot(timeChunk, noiseDataChunk)
            axTestExample[0].plot(timeChunk, testTriggerChunk)
        
        
        signalDataChunkMean = numpy.average(signalDataChunk)
        noiseDataChunkMean = numpy.average(noiseDataChunk)
        
        # signalDataChunkMax = numpy.max(signalDataChunk) # use [:(len(signalDataChunk)//2)] to get only the first ms for the new triggered data
        # noiseDataChunkMax = numpy.max(noiseDataChunk)
        signalDataChunkMax = numpy.max(signalDataChunk[:(len(signalDataChunk))])
        noiseDataChunkMax = numpy.max(noiseDataChunk[:(len(noiseDataChunk))])
        
        
        for t in range(len(thresholds)):
            # Do mean based threshold triggers
            if signalDataChunkMean > thresholds[t]:
                triggerDecisionsSignal_ThresholdAverage[t,i] = 1
            if noiseDataChunkMean > thresholds[t]:
                triggerDecisionsNoise_ThresholdAverage[t,i] = 1
            
            # Do max based threshold triggers
            if signalDataChunkMax > thresholds[t]:
                triggerDecisionsSignal_ThresholdMax[t,i] = 1
            if noiseDataChunkMax > thresholds[t]:
                triggerDecisionsNoise_ThresholdMax[t,i] = 1
        
    

    numCorrectSignal_ThresholdAverage = []
    numMissedSignal_ThresholdAverage = []
    numFalsePositiveNoise_ThresholdAverage = []
    numCorrectNoise_ThresholdAverage = []
    
    numCorrectSignal_ThresholdMax = []
    numMissedSignal_ThresholdMax = []
    numFalsePositiveNoise_ThresholdMax = []
    numCorrectNoise_ThresholdMax = []
    
    for t in range(len(thresholds)):
        numCorrectSignal_ThresholdAverage.append( numpy.count_nonzero(triggerDecisionsSignal_ThresholdAverage[t,:]) )
        numMissedSignal_ThresholdAverage.append( numberOfSignals-numCorrectSignal_ThresholdAverage[t] )
        numFalsePositiveNoise_ThresholdAverage.append( numpy.count_nonzero(triggerDecisionsNoise_ThresholdAverage[t,:]) )
        numCorrectNoise_ThresholdAverage.append( numberOfSignals-numFalsePositiveNoise_ThresholdAverage[t] )
        
        numCorrectSignal_ThresholdMax.append( numpy.count_nonzero(triggerDecisionsSignal_ThresholdMax[t,:]) )
        numMissedSignal_ThresholdMax.append( numberOfSignals-numCorrectSignal_ThresholdMax[t] )
        numFalsePositiveNoise_ThresholdMax.append( numpy.count_nonzero(triggerDecisionsNoise_ThresholdMax[t,:]) )
        numCorrectNoise_ThresholdMax.append( numberOfSignals-numFalsePositiveNoise_ThresholdMax[t] )
        
    
    
    AverageStats = {
        "Correct Signal": numCorrectSignal_ThresholdAverage,
        "Incorrect Signal": numMissedSignal_ThresholdAverage,
        "False Positive Noise": numFalsePositiveNoise_ThresholdAverage,
        "Correct Noise": numCorrectNoise_ThresholdAverage,
        "Number of Signals": numberOfSignals
        }
    
    MaxStats = {
        "Correct Signal": numCorrectSignal_ThresholdMax,
        "Incorrect Signal": numMissedSignal_ThresholdMax,
        "False Positive Noise": numFalsePositiveNoise_ThresholdMax,
        "Correct Noise": numCorrectNoise_ThresholdMax,
        "Number of Signals": numberOfSignals
        }
    
    return AverageStats, MaxStats 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
### Matched filter trigger processing
def MatchedFilterTriggerAnalysis():
    
    

    # dataFolder = "D:/Project Stuff/Red Pitaya Matched Filter Data/Chirping Data Collection/1024pt/"
    dataFolder = "D:/Project Stuff/Red Pitaya Matched Filter Data/DataWithTrigger/2048_pt_Data/"
    # dataFile = "Chirp_300kHzTemplate_0p08SNR_280Start"
    # fileName = dataFolder + dataFile     

    
    
    
    FileNameSNRs =  ["0p04","0p08","0p12","0p16","0p32"] # 
    resultsDictionary = {}
    for SNR in FileNameSNRs:
        fileName = dataFolder + "33Cycle_2048pt_" + SNR + "SNR" + "_280Start"
        resultsDictionary.update( { SNR : GetMatchedFilterTriggerPerformance(fileName) } )
    
    
    
    figROCSNR, axROCSNR = pyplot.subplots(1, 1, figsize=[16,8])
    for SNR in FileNameSNRs:
        workingDictionary = resultsDictionary[SNR][1] # for max have [1], average is [0], maxMean is [2]
        # axROCThreshold.plot(numpy.array(resultsDictionary["0p32"][0]["False Positive Noise"])/resultsDictionary["0p32"][0]["Number of Signals"]*100, numpy.array(resultsDictionary["0p32"][0]["Correct Signal"])/resultsDictionary["0p32"][0]["Number of Signals"]*100, label="Threshold Average")
        # axROCThreshold.plot(numpy.array(resultsDictionary["0p32"][1]["False Positive Noise"])/resultsDictionary["0p32"][1]["Number of Signals"]*100, numpy.array(resultsDictionary["0p32"][1]["Correct Signal"])/resultsDictionary["0p32"][1]["Number of Signals"]*100, label="Threshold Max")
        axROCSNR.plot( numpy.array(workingDictionary["False Positive Noise"]) / workingDictionary["Number of Signals"]*100 , numpy.array(workingDictionary["Correct Signal"]) / workingDictionary["Number of Signals"]*100,  label=SNR )
    axROCSNR.set_xlabel("False Positive Rate")
    axROCSNR.set_ylabel("Correct Signal Detection Rate")
    axROCSNR.set_xscale("log")
    axROCSNR.set_xlim(1/1200*100,1e2)
    axROCSNR.legend()
    

    
    # # Bandwidth study
    # startFrequencies = ["280","295","310","325","340","355","370"]
    # bandwidthResults = {}
    # for startFrequency in startFrequencies:
    #     fileName = dataFolder + "ChirpSig_NoChirpTemp_3p45e8Grad_0p04SNR_" + startFrequency + "p5kHz"
    #     print("Start frequency:",startFrequency)
    #     bandwidthResults.update( { startFrequency : GetMatchedFilterTriggerPerformance(fileName) } )
    
    # figROCBandwidth, axROCBandwidth = pyplot.subplots(1, 1, figsize = [16,8])
    # for startFrequency in startFrequencies:
    #     workingDictionary = bandwidthResults[startFrequency][1] # for max have [1]
    #     axROCBandwidth.plot( numpy.array(workingDictionary["False Positive Noise"]) / workingDictionary["Number of Signals"]*100 , numpy.array(workingDictionary["Correct Signal"]) / workingDictionary["Number of Signals"]*100,  label=startFrequency )
    # axROCBandwidth.set_xlabel("False Positive Rate")
    # axROCBandwidth.set_ylabel("Correct Signal Detection Rate")
    # axROCBandwidth.legend()
        
    
    
    # # Need to get all the bandwidths and all the SNRs, get the results for them and then get the 90% values, then make a graph.
    # startFreqAndSNRResults = {}
    # falsePositiveRatesAt90Detection = {}
    # for SNR in FileNameSNRs:
    #     for startFrequency in startFrequencies:
    #         fileName = dataFolder + "ChirpSig_NoChirpTemp_3p45e8Grad_" + SNR + "SNR_" + startFrequency + "p5kHz"
    #         print("Start Frequency, SNR:", startFrequency, SNR)
    #         resultsAverage, resultsMax = GetMatchedFilterTriggerPerformance(fileName)
    #         startFreqAndSNRResults.update( { (SNR,startFrequency) : resultsMax })
    #         currentFalsePositiveRateAt90Detection = (resultsMax["False Positive Noise"][FindIndexOfClosestToValue(resultsMax["Correct Signal"], 0.9 * resultsMax["Number of Signals"])]) / resultsMax["Number of Signals"] * 100
    #         falsePositiveRatesAt90Detection.update( { (SNR,startFrequency) : currentFalsePositiveRateAt90Detection } )
    
            
    # print(falsePositiveRatesAt90Detection)
    # falsePositiveRatesAt90DetectionArray = numpy.zeros((len(FileNameSNRs),len(startFrequencies)))
    # for i in range(len(FileNameSNRs)):
    #     for j in range(len(startFrequencies)):
    #         falsePositiveRatesAt90DetectionArray[i,j] = falsePositiveRatesAt90Detection[(FileNameSNRs[i],startFrequencies[j])]

    # print(falsePositiveRatesAt90DetectionArray)            
    
    # figFPAt90Detection, axFPAt90Detection = pyplot.subplots(1, 1, figsize=[10,8])
    # im = axFPAt90Detection.imshow(falsePositiveRatesAt90DetectionArray)
    # colorbar = figFPAt90Detection.colorbar(im)
    # colorbar.ax.set_ylabel("False Positives")
    # axFPAt90Detection.set_ylabel("SNR")
    # axFPAt90Detection.set_xlabel("Start Frequency (kHz")
    # axFPAt90Detection.set_yticks( numpy.arange(0, len(FileNameSNRs), dtype='int') )
    # axFPAt90Detection.set_xticks( numpy.arange(0, len(startFrequencies), dtype='int') )
    # axFPAt90Detection.set_yticklabels(FileNameSNRs)
    # axFPAt90Detection.set_xticklabels(startFrequencies, rotation=90)
    # figFPAt90Detection.tight_layout()
    
        
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
    
    
def FindBroadnessOfPeak(data, peakIndex, relativeHeight = 0.5):
    # start from peak, head forwards and backwards, when you find the first value below peak height*relativeHeight, stop and say the one before it is the width
    # if the peak hits the edge of the data before it finds anything, use just the other half and double, if it hits both sides, use the full width of the data
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
    
    
    
    
def GetLIATriggerPerformance(fileName): # , triggerThresholds):

    signalData = pd.read_csv(fileName + "_Sig.csv", skiprows=range(7), header=None).to_numpy()
    signalData = signalData[:len(signalData)]
    pureNoiseData = pd.read_csv(fileName + "_Noise.csv", skiprows=range(7), header=None).to_numpy()
    pureNoiseData = pureNoiseData[:len(pureNoiseData)]
    
    
    sampleRate = 1/(signalData[1,0] - signalData[0,0])
    print("Sample Rate:",sampleRate)
    timeBetweenSignals = 0.003333333333333333
    samplesIn1Signal = int(timeBetweenSignals*sampleRate)
    actualSignalLength = int(0.001*sampleRate)
    print("Samples in 1 signal:", samplesIn1Signal)
    
    
    # position of first trigger position
    # firstTriggerPosition = numpy.argmax(signalData[:samplesIn1Signal,2])
    # startTriggerOffset = firstTriggerPosition
    
    
    numberOfSignals = len(signalData)//samplesIn1Signal # For sampling of 1MSps, 2ms = 2000 samples. For 0.5Msps, 2ms = 1000 samples
    numberOfNoise = len(pureNoiseData)//samplesIn1Signal
    print("Number of Signals:", numberOfSignals)
    print("Number of Noise:", numberOfNoise)
    numberOfSignals = min([numberOfSignals, numberOfNoise]) # limited by whichever has the lowest number of data points since the mokulab's data collection seems to be a little bit inconsistent on this front.
    

    baselineVoltageForThreshold = GetRMS(pureNoiseData[:,1])
    print("Baseline voltage for threshold:", baselineVoltageForThreshold)
    thresholds = numpy.arange(baselineVoltageForThreshold/30, baselineVoltageForThreshold*30, step = baselineVoltageForThreshold/100)
    # thresholds = triggerThresholds
    
    triggerDecisionsSignal_ThresholdAverage = numpy.zeros( (len(thresholds), numberOfSignals) )
    triggerDecisionsNoise_ThresholdAverage = numpy.zeros( (len(thresholds), numberOfSignals) )
    
    triggerDecisionsSignal_ThresholdMax = numpy.zeros( (len(thresholds), numberOfSignals) )
    triggerDecisionsNoise_ThresholdMax = numpy.zeros( (len(thresholds), numberOfSignals) )
    
    triggerDecisionsSignal_ThresholdMeanMax = numpy.zeros( (len(thresholds), numberOfSignals) )
    triggerDecisionsNoise_ThresholdMeanMax = numpy.zeros( (len(thresholds), numberOfSignals) )
    
    triggerDecisionsSignal_BiggestMeanMax = numpy.zeros( (len(thresholds), numberOfSignals) )
    triggerDecisionsNoise_BiggestMeanMax = numpy.zeros( (len(thresholds), numberOfSignals) )
    
    ### Get data locations from trigger signal
    # print(signalData[:300,2])
    signalStartLocations = find_peaks(signalData[:,2], height=0)[0]
    noiseStartLocations = find_peaks(pureNoiseData[:,2], height=0)[0]
    print(len(signalData[:,2]))
    print("Noise peaks:",len(noiseStartLocations))
    print("Found",len(signalStartLocations),"peaks.")
    print("Expected:", len(signalData)/(0.00333333333333*sampleRate))
    
    numberOfSignals = min([len(signalStartLocations),len(noiseStartLocations)])
    
    signalBroadnessArray = []
    noiseBroadnessArray = []
    
    
    for i in range(numberOfSignals-1): 
        #print(i)

        #print(noiseStartLocations[i])
        #print(actualSignalLength)
        timeChunk = signalData[ (signalStartLocations[i]) : (signalStartLocations[i] + actualSignalLength), 0]
        signalDataChunk = signalData[ (signalStartLocations[i]) : (signalStartLocations[i] + actualSignalLength), 1 ]
        testTriggerChunk = signalData[(signalStartLocations[i]) : (signalStartLocations[i] + actualSignalLength), 2 ]
        noiseDataChunk = pureNoiseData[ (noiseStartLocations[i]) : (noiseStartLocations[i] + actualSignalLength), 1 ]
        
        
        testExample = False
        if testExample == True and i==1000:
            figTestExample, axTestExample = pyplot.subplots(2, 1, figsize=[16,16])
            axTestExample[0].plot(timeChunk, signalDataChunk)
            axTestExample[1].plot(timeChunk, noiseDataChunk)
            axTestExample[0].plot(timeChunk, testTriggerChunk)
        
        
        signalDataChunkMean = numpy.average(signalDataChunk)
        noiseDataChunkMean = numpy.average(noiseDataChunk)
        
        # signalDataChunkMax = numpy.max(signalDataChunk) # use [:(len(signalDataChunk)//2)] to get only the first ms for the new triggered data
        # noiseDataChunkMax = numpy.max(noiseDataChunk)
        signalDataChunkMax = numpy.max(signalDataChunk)# [:(len(signalDataChunk)//1)])
        noiseDataChunkMax = numpy.max(noiseDataChunk)# [:(len(noiseDataChunk)//1)])
        
        
        # for the max-mean
        # First want to identify some number of maxima, 
        # Then take the mean (or sum) around these maxima
        # Can use numpy argpartition, which basically makes an array in which you have [lower than selected point, selected point, higher than selected point]
        # This means if you want the top n points, you fix the position of the Nth highest peak (-N) and then all the ones from there and above will be the N highest points in the array.
        # Argpartition gives you their arguments in the original array.  They're not ordered on the low/high sides, but they are all higher or lower than the point you selected to be placed correctly.
        # Argpartition uses elements rather than index though, so watch out for that.
        # Actually, this just gets the top 5 points, which could actually all come from the same peak...
        # topNMaximaLocations = numpy.argpartition(signalDataChunk,-5)[-5:]
        
        # Instead, just going to start with one maximum and take the mean around that, see what changes.
        signalArgMax = numpy.argmax(signalDataChunk)
        #print("Max:", numpy.max(signalDataChunk))
        #print("Max 2:",numpy.max(signalDataChunk[:(len(signalDataChunk)//1)]))
        #print("Max 3:", signalDataChunk[signalArgMax])
        noiseArgMax = numpy.argmax(noiseDataChunk)
        


        peakWidth = 50 # samples, gives half the width of the peak to be investigated
        signalMeanAroundMax = CalculatePeakMean(signalDataChunk, signalArgMax, peakWidth)
        noiseMeanAroundMax = CalculatePeakMean(noiseDataChunk, noiseArgMax, peakWidth)
        
        
        
        #print("SignalDiff:", signalMeanAroundMax - signalDataChunkMax)
        #print("NoiseDiff:", noiseMeanAroundMax - noiseDataChunkMax)
        #print("NoiseMeanMax:", noiseMeanAroundMax)
        #print("NMax:", numpy.max(noiseDataChunk))
        #print("NMax 2:",numpy.max(noiseDataChunk[:(len(noiseDataChunk)//1)]))
        #print("NMax 3:", noiseDataChunk[noiseArgMax])
        
        # Find n biggest peaks and then find broadness of these, use biggest average for discrimination and do a plot for signal and noise,
        # highlighting the difference and justifying this approach to selection.
        nPeaks = 5
        if i==0:
            printGraphs = False
        else:
            printGraphs = False
            
            

        signalBiggestMaxMean = CalculateBiggestMaxMeanOfNPeaks(signalDataChunk, nPeaks, peakWidth, distance=25, printGraphs=printGraphs, broadnessArray = signalBroadnessArray, label="Signal peaks example, "+fileName)
        noiseBiggestMaxMean = CalculateBiggestMaxMeanOfNPeaks(noiseDataChunk, nPeaks, peakWidth, distance=25, printGraphs = printGraphs, broadnessArray = noiseBroadnessArray, label="Noise peaks example, "+fileName)
        
        
        
        
        
        
        for t in range(len(thresholds)):
            # Do mean based threshold triggers
            if signalDataChunkMean > thresholds[t]:
                triggerDecisionsSignal_ThresholdAverage[t,i] = 1
            if noiseDataChunkMean > thresholds[t]:
                triggerDecisionsNoise_ThresholdAverage[t,i] = 1
            
            # Do max based threshold triggers
            if signalDataChunkMax > thresholds[t]:
                triggerDecisionsSignal_ThresholdMax[t,i] = 1
            if noiseDataChunkMax > thresholds[t]:
                triggerDecisionsNoise_ThresholdMax[t,i] = 1
                
            # Do max-mean based threshold triggers
            if signalMeanAroundMax > thresholds[t]:
                triggerDecisionsSignal_ThresholdMeanMax[t,i] = 1
            if noiseMeanAroundMax > thresholds[t]:
                triggerDecisionsNoise_ThresholdMeanMax[t,i] = 1
        
            if signalBiggestMaxMean > thresholds[t]:
                triggerDecisionsSignal_BiggestMeanMax[t,i] = 1
            if noiseBiggestMaxMean > thresholds[t]:
                triggerDecisionsNoise_BiggestMeanMax[t,i] = 1
        

   

    # # Check broadness of peaks:
    # bins = numpy.arange(0,120,3)
    # figBroadness, axBroadness = pyplot.subplots(1, 1, figsize=[16,8])
    # axBroadness.hist(signalBroadnessArray, label="Signal", bins=bins, alpha = 0.7)
    # axBroadness.hist(noiseBroadnessArray, label="Noise", bins=bins, alpha = 0.7)
    # axBroadness.legend()
    # axBroadness.set_xlabel("Width (samples)")
    # axBroadness.set_ylabel("Counts")
    # #axBroadness.set_title(fileName+" Data")

    numCorrectSignal_ThresholdAverage = []
    numMissedSignal_ThresholdAverage = []
    numFalsePositiveNoise_ThresholdAverage = []
    numCorrectNoise_ThresholdAverage = []
    
    numCorrectSignal_ThresholdMax = []
    numMissedSignal_ThresholdMax = []
    numFalsePositiveNoise_ThresholdMax = []
    numCorrectNoise_ThresholdMax = []
    
    
    numCorrectSignal_ThresholdMeanMax = []
    numMissedSignal_ThresholdMeanMax = []
    numFalsePositiveNoise_ThresholdMeanMax = []
    numCorrectNoise_ThresholdMeanMax = []
    
    numCorrectSignal_BiggestMeanMax = []
    numMissedSignal_BiggestMeanMax = []
    numFalsePositiveNoise_BiggestMeanMax = []
    numCorrectNoise_BiggestMeanMax = []
    
    
    for t in range(len(thresholds)):
        numCorrectSignal_ThresholdAverage.append( numpy.count_nonzero(triggerDecisionsSignal_ThresholdAverage[t,:]) )
        numMissedSignal_ThresholdAverage.append( numberOfSignals-numCorrectSignal_ThresholdAverage[t] )
        numFalsePositiveNoise_ThresholdAverage.append( numpy.count_nonzero(triggerDecisionsNoise_ThresholdAverage[t,:]) )
        numCorrectNoise_ThresholdAverage.append( numberOfSignals-numFalsePositiveNoise_ThresholdAverage[t] )
        
        numCorrectSignal_ThresholdMax.append( numpy.count_nonzero(triggerDecisionsSignal_ThresholdMax[t,:]) )
        numMissedSignal_ThresholdMax.append( numberOfSignals-numCorrectSignal_ThresholdMax[t] )
        numFalsePositiveNoise_ThresholdMax.append( numpy.count_nonzero(triggerDecisionsNoise_ThresholdMax[t,:]) )
        numCorrectNoise_ThresholdMax.append( numberOfSignals-numFalsePositiveNoise_ThresholdMax[t] )
        
        
        numCorrectSignal_ThresholdMeanMax.append( numpy.count_nonzero(triggerDecisionsSignal_ThresholdMeanMax[t,:]) )
        numMissedSignal_ThresholdMeanMax.append( numberOfSignals-numCorrectSignal_ThresholdMeanMax[t] )
        numFalsePositiveNoise_ThresholdMeanMax.append( numpy.count_nonzero(triggerDecisionsNoise_ThresholdMeanMax[t,:]) )
        numCorrectNoise_ThresholdMeanMax.append( numberOfSignals-numFalsePositiveNoise_ThresholdMeanMax[t] )
        
        numCorrectSignal_BiggestMeanMax.append( numpy.count_nonzero(triggerDecisionsSignal_BiggestMeanMax[t,:]) )
        numMissedSignal_BiggestMeanMax.append( numberOfSignals-numCorrectSignal_BiggestMeanMax[t] )
        numFalsePositiveNoise_BiggestMeanMax.append( numpy.count_nonzero(triggerDecisionsNoise_BiggestMeanMax[t,:]) )
        numCorrectNoise_BiggestMeanMax.append( numberOfSignals-numFalsePositiveNoise_BiggestMeanMax[t] )
        
    
    
    AverageStats = {
        "Correct Signal": numCorrectSignal_ThresholdAverage,
        "Incorrect Signal": numMissedSignal_ThresholdAverage,
        "False Positive Noise": numFalsePositiveNoise_ThresholdAverage,
        "Correct Noise": numCorrectNoise_ThresholdAverage,
        "Number of Signals": numberOfSignals
        }
    
    MaxStats = {
        "Correct Signal": numCorrectSignal_ThresholdMax,
        "Incorrect Signal": numMissedSignal_ThresholdMax,
        "False Positive Noise": numFalsePositiveNoise_ThresholdMax,
        "Correct Noise": numCorrectNoise_ThresholdMax,
        "Number of Signals": numberOfSignals
        }
    MeanMaxStats = {
        "Correct Signal": numCorrectSignal_ThresholdMeanMax,
        "Incorrect Signal": numMissedSignal_ThresholdMeanMax,
        "False Positive Noise": numFalsePositiveNoise_ThresholdMeanMax,
        "Correct Noise": numCorrectNoise_ThresholdMeanMax,
        "Number of Signals": numberOfSignals
        }
    BiggestMeanMaxStats = {
        "Correct Signal": numCorrectSignal_BiggestMeanMax,
        "Incorrect Signal": numMissedSignal_BiggestMeanMax,
        "False Positive Noise": numFalsePositiveNoise_BiggestMeanMax,
        "Correct Noise": numCorrectNoise_BiggestMeanMax,
        "Number of Signals": numberOfSignals
        }
    
    
   
    
    
    
    
    return AverageStats, MaxStats, MeanMaxStats, BiggestMeanMaxStats
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def LIATriggerAnalysis():
    
    dataFolder = "D:/Project Stuff/Red Pitaya LIA Data/1ms Data/"
    FileNameSNRs =  ["0p04","0p08","0p12","0p16"]# ,"0p32"]
    resultsDictionary = {}
    for SNR in FileNameSNRs:
        fileName = dataFolder + SNR + "SNR_1em5TC" # _0p1msChirp
        resultsDictionary.update( { SNR : GetLIATriggerPerformance(fileName) } )
    
    figROCSNR, axROCSNR = pyplot.subplots(1, 1, figsize=[16,8])
    for SNR in FileNameSNRs:
        workingDictionary = resultsDictionary[SNR][0] # for max have [1] # For Average have 0 # For biggest maxmean have 3
        # axROCThreshold.plot(numpy.array(resultsDictionary["0p32"][0]["False Positive Noise"])/resultsDictionary["0p32"][0]["Number of Signals"]*100, numpy.array(resultsDictionary["0p32"][0]["Correct Signal"])/resultsDictionary["0p32"][0]["Number of Signals"]*100, label="Threshold Average")
        # axROCThreshold.plot(numpy.array(resultsDictionary["0p32"][1]["False Positive Noise"])/resultsDictionary["0p32"][1]["Number of Signals"]*100, numpy.array(resultsDictionary["0p32"][1]["Correct Signal"])/resultsDictionary["0p32"][1]["Number of Signals"]*100, label="Threshold Max")
        axROCSNR.plot( numpy.array(workingDictionary["False Positive Noise"]) / workingDictionary["Number of Signals"]*100 , numpy.array(workingDictionary["Correct Signal"]) / workingDictionary["Number of Signals"]*100,  label=SNR , linestyle="-")
    axROCSNR.set_xlabel("False Positive Rate")
    axROCSNR.set_ylabel("Correct Signal Detection Rate")
    axROCSNR.set_xscale("log")
    axROCSNR.set_xlim(1/1200*100,1e2)
    axROCSNR.legend()
    
    
    
    return 0
    
    
    
    
def GetLIATriggerDataPerformance(fileName):
    # For a given file, read the triggers and get the statistics

    signalData = pd.read_csv(fileName + "_Sig.csv", skiprows=range(7), header=None).to_numpy()
    pureNoiseData = pd.read_csv(fileName + "_Noise.csv", skiprows=range(7), header=None).to_numpy()
    
    
    sampleRate = 1/(signalData[1,0] - signalData[0,0])
    print("Sample Rate:",sampleRate)
    timeBetweenSignals = 0.003333333333333333
    samplesIn1Signal = int(timeBetweenSignals*sampleRate)
    actualSignalLength = int(0.001*sampleRate)
    print("Samples in 1 signal:", samplesIn1Signal)
       
    numberOfSignals = len(signalData)//samplesIn1Signal # For sampling of 1MSps, 2ms = 2000 samples. For 0.5Msps, 2ms = 1000 samples
    numberOfNoise = len(pureNoiseData)//samplesIn1Signal
    print("Number of Signals:", numberOfSignals)
    print("Number of Noise:", numberOfNoise)
    numberOfSignals = min([numberOfSignals, numberOfNoise]) # limited by whichever has the lowest number of data points since the mokulab's data collection seems to be a little bit inconsistent on this front.
    
    
    ### Get data locations from trigger signal
    # print(signalData[:300,2])
    signalStartLocations = find_peaks(signalData[:,2], height=0)[0]
    noiseStartLocations = find_peaks(pureNoiseData[:,2], height=0)[0]
    print(len(signalData[:,2]))
    print("Found",len(signalStartLocations),"peaks.")
    print("Expected:", len(signalData)/(0.00333333333333*sampleRate))
    
    numCorrectSignal = 0
    numFalsePositiveNoise = 0
    numCorrectNoise = 0
    numIncorrectSignal = 0
    
    for i in range(numberOfSignals-1): # -2

        timeChunk = signalData[ (signalStartLocations[i]) : (signalStartLocations[i] + actualSignalLength*2), 0]
        signalDataChunk = signalData[ (signalStartLocations[i]) : (signalStartLocations[i] + actualSignalLength*2), 1 ]
        testTriggerChunk = signalData[(signalStartLocations[i]) : (signalStartLocations[i] + actualSignalLength*2), 2 ]
        noiseDataChunk = pureNoiseData[ (noiseStartLocations[i]) : (noiseStartLocations[i] + actualSignalLength*2), 1 ]
        
        
        testExample = True
        if testExample == True and i==1000:
            figTestExample, axTestExample = pyplot.subplots(2, 1, figsize=[16,16])
            axTestExample[0].plot(timeChunk, signalDataChunk)
            axTestExample[1].plot(timeChunk, noiseDataChunk)
            axTestExample[0].plot(timeChunk, testTriggerChunk)
        
        
        if signalData[int(signalStartLocations[i]+actualSignalLength*1.5), 1] > 0.5:
            # If the point halfway through the trigger signal area is high, count the trigger as a correct detection
            numCorrectSignal+=1
        else:
            numIncorrectSignal+=1
        
        if pureNoiseData[int(noiseStartLocations[i]+actualSignalLength*1.5), 1] > 0.5:
            # If the point halfway through the trigger signal area is high for noise, count the trigger as a false positive
            numFalsePositiveNoise+=1
        else:
            numCorrectNoise+=1
        
    Results = {
        "Correct Signal": numCorrectSignal,
        "Incorrect Signal": numIncorrectSignal,
        "False Positive Noise": numFalsePositiveNoise,
        "Correct Noise": numCorrectNoise,
        "Number of Signals": numberOfSignals
        }
    
    return Results
    
    
def LIATriggersProcessing():
    # For processing the data that has triggers as the output instead of the raw LIA data
    dataFolder = "D:/Project Stuff/Red Pitaya LIA Data/With Trigger/1e-5sTC Trigger Responses/"
    #FileNameSNRs =  ["0p08"]# ,"0p32"] # Need to change these to actual values afterwards
    # FileNameTriggerThresholds = ["0p0025", "0p0035", "0p0045", "0p0055", "0p0065", "0p0075", "0p0085", "0p0095", "0p0105"]
    FileNameTriggerThresholds = ["0p0095", "0p0105", "0p0115", "0p0125", "0p0135", "0p0165", "0p0175", "0p0185", "0p0195", "0p0205", "0p0215", "0p0225", "0p0235", "0p0245"]
    TriggerThresholds = [0.0095,0.0105,0.0115,0.0125,0.0135,0.0165,0.0175,0.0185,0.0195,0.0205,0.0215,0.0225,0.0235,0.0245]
    #TriggerThresholds = numpy.arange(0.0095, 0.0205,step=0.001)
    #TriggerThresholds=TriggerThresholds[:-1]
    print(len(TriggerThresholds))
    print(len(FileNameTriggerThresholds))
    resultsDictionary = {}
    correctSignals = []
    incorrectSignals = []
    correctNoises = []
    incorrectNoises = []
    numSignals = []
    for TriggerThreshold in FileNameTriggerThresholds:
        fileName = dataFolder + "Triggers_0p16SNR_" +  TriggerThreshold + "SweepMin_1e_5TC_0p1ms"
        resultsDictionary.update( { TriggerThreshold : GetLIATriggerDataPerformance(fileName) } )
        correctSignals.append(resultsDictionary[TriggerThreshold]["Correct Signal"])
        incorrectSignals.append(resultsDictionary[TriggerThreshold]["Incorrect Signal"])
        correctNoises.append(resultsDictionary[TriggerThreshold]["Correct Noise"])
        incorrectNoises.append(resultsDictionary[TriggerThreshold]["False Positive Noise"])
        numSignals.append(resultsDictionary[TriggerThreshold]["Number of Signals"])
    
    #print( resultsDictionary[])
    
    figTriggerCurve, axTriggerCurve = pyplot.subplots(2, 1, figsize=[16,16])
    axTriggerCurve[0].plot(numpy.array(TriggerThresholds)/2.4, numpy.array(correctSignals)/numSignals[0])
    axTriggerCurve[1].plot(numpy.array(TriggerThresholds)/2.4, numpy.array(correctNoises)/numSignals[0])
    axTriggerCurve[0].set_xlabel("Trigger Threshold (V)")
    axTriggerCurve[0].set_ylabel("Correct Detection Rate")
    axTriggerCurve[0].set_title("Signal")
    axTriggerCurve[1].set_xlabel("Trigger Threshold (V)")
    axTriggerCurve[1].set_ylabel("Correct Detection Rate")
    axTriggerCurve[1].set_title("Noise")
    
    figCombinedCurve, axCombinedCurve = pyplot.subplots(1, 1, figsize=[16,8])
    axCombinedCurve.plot(numpy.array(TriggerThresholds)/2.4, numpy.array(correctSignals)/numSignals[0], color="b", label="Signal")
    axCombinedCurve.plot(numpy.array(TriggerThresholds)/2.4, numpy.array(correctNoises)/numSignals[0], color="b", linestyle="--", label="Noise")
    axCombinedCurve.set_xlabel("Trigger Threshold (V)")
    axCombinedCurve.set_ylabel("Correct Detection Rate")
    axCombinedCurve.set_ylim(-0.1,1.1)
    axCombinedCurve.set_xlim(0.003, 0.011)
    axCombinedCurve.legend()

    return 0




def GetMFTriggerDataPerformance(fileName):
    # For a given file, read the triggers and get the statistics

    signalData = pd.read_csv(fileName + "_Sig.csv", skiprows=range(7), header=None).to_numpy()
    pureNoiseData = pd.read_csv(fileName + "_Noise.csv", skiprows=range(7), header=None).to_numpy()
    
    sampleRate = 1/(signalData[1,0] - signalData[0,0])
    print("Sample Rate:",sampleRate)
    timeBetweenSignals = 0.003333333333333333
    samplesIn1Signal = int(timeBetweenSignals*sampleRate)
    actualSignalLength = int(0.001*sampleRate)
    print("Samples in 1 signal:", samplesIn1Signal)
    
    numberOfSignals = len(signalData)//samplesIn1Signal # For sampling of 1MSps, 2ms = 2000 samples. For 0.5Msps, 2ms = 1000 samples
    numberOfNoise = len(pureNoiseData)//samplesIn1Signal
    print("Number of Signals:", numberOfSignals)
    print("Number of Noise:", numberOfNoise)
    numberOfSignals = min([numberOfSignals, numberOfNoise]) # limited by whichever has the lowest number of data points since the mokulab's data collection seems to be a little bit inconsistent on this front.

    
    ### Get data locations from trigger signal
    # print(signalData[:300,2])
    signalStartLocations = find_peaks(signalData[:,2], height=-0.1)[0]
    noiseStartLocations = find_peaks(pureNoiseData[:,2], height=-0.1)[0]
    print(len(signalData[:,2]))
    print("Found",len(signalStartLocations),"peaks.")
    print("Found",len(noiseStartLocations),"noise peaks.")
    print("Expected:", len(signalData)/(0.00333333333333*sampleRate))
    
    # maxPointOffset = int(0.00022181101449275362318840579710145*sampleRate) # The amount of time to chirp from 280kHz to 356524.8kHz (The last point in the template, or the point that the template fully matches the signal)
    maxPointOffset = int(0.00038565101449275362318840579710145*sampleRate)
    
    ######### ^ THIS PROBABLY NEEDS CHECKING AND CHANGING FOR THE 2048 PT TEMPLATE! #########
    
    
    
    
    numCorrectSignal = 0
    numFalsePositiveNoise = 0
    numCorrectNoise = 0
    numIncorrectSignal = 0
    
    for i in range(numberOfSignals-1): # -2



        timeChunk = signalData[ (signalStartLocations[i]) : (signalStartLocations[i] + actualSignalLength*2), 0]
        signalDataChunk = signalData[ (signalStartLocations[i]) : (signalStartLocations[i] + actualSignalLength*2), 1 ]
        testTriggerChunk = signalData[(signalStartLocations[i]) : (signalStartLocations[i] + actualSignalLength*2), 2 ]
        noiseDataChunk = pureNoiseData[ (noiseStartLocations[i]) : (noiseStartLocations[i] + actualSignalLength*2), 1 ]
        
        
        testExample = False
        if testExample == True and i==1000:
            figTestExample, axTestExample = pyplot.subplots(2, 1, figsize=[16,16])
            axTestExample[0].plot(timeChunk, signalDataChunk)
            axTestExample[1].plot(timeChunk, noiseDataChunk)
            axTestExample[0].plot(timeChunk, testTriggerChunk)
        
        
        #if signalData[int(signalStartLocations[i]+actualSignalLength*1.5), 1] > 0.5:
        #    # If the point halfway through the trigger signal area is high, count the trigger as a correct detection
        #     numCorrectSignal+=1
        # else:
        #     numIncorrectSignal+=1
        
        # if pureNoiseData[int(noiseStartLocations[i]+actualSignalLength*1.5), 1] > 0.5:
        #     # If the point halfway through the trigger signal area is high for noise, count the trigger as a false positive
        #     numFalsePositiveNoise+=1
        # else:
        #     numCorrectNoise+=1
        
        
        ### Make trigger decisions:
        # Since the trigger occurs randomly in a 1ms time after the main point that is like 0.22ms after the trigger signal
        # Check the last point and maybe a few before to guarantee the trigger result
        
        if signalData[signalStartLocations[i] + actualSignalLength + maxPointOffset, 1] > 0.5:
            # Checks the point 1ms after the point where the template fully matches the signal
            numCorrectSignal+=1
        else:
            # If it's zero, check slightly before just in case my check is a few points off:
            if signalData[signalStartLocations[i] + actualSignalLength + maxPointOffset - 10, 1] > 0.5:
                numCorrectSignal+=1
            else:
                numIncorrectSignal+=1
        
        if pureNoiseData[noiseStartLocations[i] + actualSignalLength + maxPointOffset, 1] > 0.5:
            # Checks the point 1ms after the point where the template fully matches the signal
            numFalsePositiveNoise+=1
        else:
            # If it's zero, check slightly before just in case my check is a few points off:
            if pureNoiseData[noiseStartLocations[i] + actualSignalLength + maxPointOffset - 10, 1] > 0.5:
                numFalsePositiveNoise+=1
            else:
                numCorrectNoise+=1
        
        
        
    Results = {
        "Correct Signal": numCorrectSignal,
        "Incorrect Signal": numIncorrectSignal,
        "False Positive Noise": numFalsePositiveNoise,
        "Correct Noise": numCorrectNoise,
        "Number of Signals": numberOfSignals
        }
    
    return Results


def MFTriggersProcessing():
    # For processing the data that has triggers as the output instead of the raw MF data
    dataFolder = "D:/Project Stuff/Red Pitaya Matched Filter Data/WithTriggerOutput/NoSqrt 1024pt 17cycle/"
    #FileNameSNRs =  ["0p08"]# ,"0p32"] # Need to change these to actual values afterwards
    # FileNameTriggerThresholds = ["0p0025", "0p0035", "0p0045", "0p0055", "0p0065", "0p0075", "0p0085", "0p0095", "0p0105"]
    # FileNameTriggerThresholds = ["0.014", "0p410", "0p420", "0p430", "0p440", "0p450", "0p460", "0p470", "0p480", "0p490", "0p500", "0p510"]
    TriggerThresholds = [0.020,0.025,0.030,0.035,0.040,0.045,0.050,0.055,0.060]
    FileNameTriggerThresholds = ["0.020","0.025","0.030","0.035","0.040","0.045","0.050","0.055","0.060"]
    # TriggerThresholds = numpy.arange(0.400, 0.540,step=0.01)
    #TriggerThresholds=TriggerThresholds[:-1]
    print(len(TriggerThresholds))
    print(len(FileNameTriggerThresholds))
    resultsDictionary = {}
    correctSignals = []
    incorrectSignals = []
    correctNoises = []
    incorrectNoises = []
    numSignals = []
    for TriggerThreshold in FileNameTriggerThresholds:
        fileName = dataFolder + "NoSqrt1024_Triggers_0.04SNR_" +  TriggerThreshold + "VThreshold"
        resultsDictionary.update( { TriggerThreshold : GetMFTriggerDataPerformance(fileName) } )
        correctSignals.append(resultsDictionary[TriggerThreshold]["Correct Signal"])
        incorrectSignals.append(resultsDictionary[TriggerThreshold]["Incorrect Signal"])
        correctNoises.append(resultsDictionary[TriggerThreshold]["Correct Noise"])
        incorrectNoises.append(resultsDictionary[TriggerThreshold]["False Positive Noise"])
        numSignals.append(resultsDictionary[TriggerThreshold]["Number of Signals"])
    
    
    figTriggerCurve, axTriggerCurve = pyplot.subplots(2, 1, figsize=[16,16])
    axTriggerCurve[0].plot(numpy.array(TriggerThresholds)/2.4, numpy.array(correctSignals)/numSignals[0])
    axTriggerCurve[1].plot(numpy.array(TriggerThresholds)/2.4, numpy.array(correctNoises)/numSignals[0])
    axTriggerCurve[0].set_xlabel("Trigger Threshold (V)")
    axTriggerCurve[0].set_ylabel("Correct Detection Rate")
    axTriggerCurve[0].set_title("Signal")
    axTriggerCurve[1].set_xlabel("Trigger Threshold (V)")
    axTriggerCurve[1].set_ylabel("Correct Detection Rate")
    axTriggerCurve[1].set_title("Noise")
    
    figCombinedCurve, axCombinedCurve = pyplot.subplots(1, 1, figsize=[16,8])
    # The conversion is /2.4 for the sqrted version of the trigger, /1.3 for the non-sqrt version
    axCombinedCurve.plot(numpy.array(TriggerThresholds)/1.3, numpy.array(correctSignals)/numSignals[0], color="b", label="Signal")
    axCombinedCurve.plot(numpy.array(TriggerThresholds)/1.3, numpy.array(correctNoises)/numSignals[0], color="b", linestyle="--", label="Noise")
    axCombinedCurve.set_xlabel("Trigger Threshold (V)")
    axCombinedCurve.set_ylabel("Correct Detection Rate")
    axCombinedCurve.set_ylim(-0.1,1.1)
    axCombinedCurve.set_xlim(0.0058,0.092)
    axCombinedCurve.legend()

    # Need to re-work out the thresholds for the full 56-bit version
    

    return 0
    
    
if __name__ == "__main__":
    # print("Piecewise test:", FlatPiecewiseCurve(2, 0.1, 1, 2))

    # # First file is noise only
    # folderName = "Sweeping Reference Data/Aligned Sweeps/"
    # fileName = "AXI Master Test/PlainNoise_20230307_134302.csv"
    # mokulabData = pd.read_csv("C:/Users/nemes/Desktop/Project Stuff/Red Pitaya Stuff/%s" % fileName, skiprows=range(8), header=None)
    # mokulabData = mokulabData[:-1] # Last value had some nands in it for some reason
    # #print(mokulabData)
    # #print(mokulabData[0])
    # mokulabDataNumpy = mokulabData.to_numpy()
    # print("Max time =", mokulabDataNumpy[-1,0])
    
    # # Second file is signal + noise
    # fileName = folderName + "MatchChirp2e5_1xBoost_20221216_161029.csv"
    # mokulabData2 = pd.read_csv("C:/Users/nemes/Desktop/Project Stuff/Red Pitaya Stuff/%s" % fileName, skiprows=range(8), header=None)
    # mokulabData2 = mokulabData2[:-1] # Last value had some nands in it for some reason
    # #print(mokulabData2)
    # #print(mokulabData2[0])



    # # Plots some signal data with lines to try and match things well
    # # fig, ax = pyplot.subplots(1, 1, figsize=[16,8])
    # # start=0
    # # stop=-1
    # # xcoords = numpy.arange(0.0002,5,step=0.001011)
    # # for xc in xcoords:
    # #     ax.axvline(x=xc)
    # # #ax.plot(mokulabData[0][start:stop], mokulabData[1][start:stop], label="Noise")
    # # # ax.plot(mokulabData2[0][start:stop], mokulabData2[1][start:stop], label="Signal+Noise")
    # # ax.set_xlabel("Time (s)")
    # # ax.set_ylabel("Voltage (V)")
    # # # ax.legend()
    # # ax.set_xlim(0.0,0.4)



    # #print("Times:", mokulabData[0][0], "to", mokulabData[0][len(mokulabData[0])//2000] )
    # stepLength = 1011 # This is the length of a single reference chirp in terms of index number

    # # InvestigatePerformance()
    # noiseData = mokulabDataNumpy[:,1]
    
    # noiseRMS = GetRMS(noiseData)# - numpy.mean(noiseData)
    # print(noiseData)
    
    # xs = numpy.linspace(0, len(noiseData)-1, num=len(noiseData))
    # print("noiseRMS =", noiseRMS)
    
    # figNoise, axNoise = pyplot.subplots(1, 1, figsize=[16,8])
    # axNoise.plot(xs, noiseData)
    
    
    
    
    
    
    #MatchedFilterTriggerAnalysis()
    
    #testFileFolder = "D:/Project Stuff/Red Pitaya Matched Filter Data/DataWithTrigger/33_Cycle_Data/Mid Range Signals/"
    #testFileName = "250Start_1msChirp_33Cycle_0p32SNR_Sig.csv"
    
    #testData = pd.read_csv(testFileFolder+testFileName, skiprows=range(7), header=None).to_numpy()
    
    #print(testData[:5,0])
    #figTest, axTest = pyplot.subplots(1, 1, figsize=[16,8])
    #axTest.plot(testData[1000:2000,0], testData[1000:2000,1])
    #axTest.plot(testData[1000:2000,0], testData[1000:2000,2])
    
    
    
    #LIATriggerAnalysis()
    
    
    
    testFile = "D:/Project Stuff/Red Pitaya LIA Data/1ms Data/0p04SNR_1em4TC_Sig.csv"
    testData = pd.read_csv(testFile, skiprows=range(7), header=None).to_numpy()
    testFile2 = "D:/Project Stuff/Red Pitaya LIA data/1ms Data/0p04SNR_1em4TC_Noise.csv"
    testData2 = pd.read_csv(testFile2, skiprows=range(7), header=None).to_numpy()
    testFile3 = "D:/Project Stuff/Red Pitaya LIA data/1ms Data/0p08SNR_1em4TC_Sig.csv"
    testData3 = pd.read_csv(testFile3, skiprows=range(7), header=None).to_numpy()
    testFig,testAx = pyplot.subplots(1, 1, figsize=[16,8])
    
    timeStartLabel = 0.0184675 + 0.0033
    #testAx.plot(testData[:,0]-timeStartLabel,testData[:,1], label="0.04")
    offset = 0.018 + 0.0033*1
    
    timeRange = 0.0033
    testAx.set_xlim(0+offset-timeStartLabel,0+timeRange+offset-timeStartLabel)
    testAx.set_ylim(-0.05,0.5)
    #testAx.plot(testData[:,0]-timeStartLabel, testData[:,2])
    testAx.plot(testData3[:,0]+0.00056-timeStartLabel,testData3[:,1], label="0.08", color="c", alpha=0.8)
    testAx.plot(testData2[:,0]-timeStartLabel,testData2[:,1], label="Noise", color="r")
    
    testAx.legend(title="SNR")
    testAx.set_xlabel("Time (s)")
    testAx.set_ylabel("LIABT Response (V)")
    #testAx.plot(testData3[:,0]+0.00056,testData3[:,2])
    #testAx.plot(testData[:,0], testData[:,2])
    # testFile = "D:/Project Stuff/Red Pitaya LIA Data/With Trigger/1e-5sTC Trigger Responses 2/Triggers_0p08SNR_0p0075SweepMin_1e_5TC_0p1ms"
    # GetLIATriggerDataPerformance(testFile)
    
    #LIATriggersProcessing()
    # MFTriggersProcessing()