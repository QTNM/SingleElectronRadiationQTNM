import numpy
import matplotlib.pyplot as pyplot
import SpectrumAnalyser
from scipy.signal import correlate
import SignalProcessingFunctions as SPF
import random
import Constants
import copy

from matplotlib.ticker import FormatStrFormatter


def TestMatchedFilterTrigger():
    # Test chirp correlation for two different chirps
    sampleRate = 200e6
    times = numpy.arange(0, 1e-3, step=1/sampleRate)
    signalAmplitude = 3.075179595089654e-08 * 2**0.5
    signal = signalAmplitude * \
        SPF.GenerateChirpSignal(times, 20.2e6, 3.45e8, 0)
    template = SPF.GenerateChirpSignal(times, 20.1e6, 3e8, 0)

    # Add some noise to the data
    temperature = 4  # K
    antennaBandwidth = 1e6
    antennaLowCutoff = 15e6
    kTdB = Constants.k_B * temperature * antennaBandwidth
    print("kTdB:", kTdB)
    resistance = 73.2
    noise = SPF.GenerateVoltageNoise(sampleRate=sampleRate,
                                     numSamples=len(times),
                                     temp=temperature,
                                     resistance=resistance,
                                     antennaLowCutoff=antennaLowCutoff,
                                     antennaBandwidth=antennaBandwidth)
    signal += noise

    fullTimes = numpy.arange(-1e-3+1/sampleRate, 1e-3, step=1/sampleRate)
    # Need to put template first to get correct time order
    correlation = correlate(template, signal, mode='full')
    print(len(times))
    print(len(correlation))

    frequencyGradients = numpy.arange(3.29e8, 3.61e8, step=1e6)
    peakCorrelationsGradient = []
    for i in range(len(frequencyGradients)):
        template = SPF.GenerateChirpSignal(
            times, 20.2e6, frequencyGradients[i], 0)
        correlation = correlate(template, signal, mode='full')
        peakCorrelationsGradient.append(numpy.amax(correlation))

    startFrequencies = numpy.arange(20e6, 21e6, step=0.02e6)
    peakCorrelationsStart = []
    for i in range(len(startFrequencies)):
        template = SPF.GenerateChirpSignal(times, startFrequencies[i], 3e8, 0)
        correlation = correlate(signal, template, mode='full')
        peakCorrelationsStart.append(numpy.amax(correlation))

    figTestCorrelation, axTestCorrelation = pyplot.subplots(1, 1, figsize=[
                                                            18, 8])
    axTestCorrelation.plot(fullTimes, correlation)
    axTestCorrelation.set_xlabel("Time (s)")
    axTestCorrelation.set_ylabel("Correlation")

    SpectrumAnalyser.ApplyFFTAnalyser(times,
                                      signal,
                                      lowSearchFreq=20e6,
                                      highSearchFreq=21e6,
                                      nTimeSplits=10,
                                      nFreqBins=40)

    figPeakCorrelations, axPeakCorrelations = pyplot.subplots(1, 1, figsize=[
                                                              18, 8])
    axPeakCorrelations.plot(
        frequencyGradients, peakCorrelationsGradient/numpy.amax(peakCorrelationsGradient)*100)
    axPeakCorrelations.set_xlabel("Frequency Gradient (Hz/s)")
    axPeakCorrelations.set_ylabel("Peak Correlation")

    figPeakCorrelationsStartFreq, axPeakCorrelationsStartFreq = pyplot.subplots(
        1, 1, figsize=[18, 8])
    axPeakCorrelationsStartFreq.plot(
        startFrequencies, peakCorrelationsStart/numpy.amax(peakCorrelationsStart)*100)
    axPeakCorrelationsStartFreq.set_xlabel("Start Frequency (Hz)")
    axPeakCorrelationsStartFreq.set_ylabel("Peak Correlation")
    return 0


def GetCorrectSignalDetectionRate(numberOfTests, triggerThreshold, templates, sampleRate, times, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth):
    triggerDecisions = []
    for i in range(numberOfTests):
        randomStartFrequency = numpy.random.uniform(
            startFrequencyBounds[0], startFrequencyBounds[1])
        randomFrequencyGradient = numpy.random.uniform(
            frequencyGradientBounds[0], frequencyGradientBounds[1])
        randomPhase = numpy.random.uniform(0, 2*numpy.pi)
        signal = signalAmplitude * \
            SPF.GenerateChirpSignal(
                times, randomStartFrequency, randomFrequencyGradient, randomPhase)
        noise = SPF.GenerateVoltageNoise(sampleRate=sampleRate,
                                         numSamples=len(times),
                                         temp=temperature,
                                         resistance=resistance,
                                         antennaLowCutoff=antennaLowCutoff,
                                         antennaBandwidth=antennaBandwidth,
                                         hidePrints=True)
        signal += noise

        peakCorrelations = []
        for template in templates:
            # ::33 to emulate using multiple clock cycles for a single MF calculation
            peakCorrelations.append(numpy.amax(
                abs(correlate(signal, template)[::129])**0.25))
        peakCorrelation = numpy.amax(peakCorrelations)

        if peakCorrelation > triggerThreshold:
            triggerDecisions.append(True)
        else:
            triggerDecisions.append(False)

    correctSignalDetectionRate = numpy.count_nonzero(
        triggerDecisions) / numberOfTests
    print("Correct signal detection rate:", correctSignalDetectionRate)
    return correctSignalDetectionRate


def GetFalseAlarmRate(numberOfTests, triggerThreshold, templates, sampleRate, times, temperature, resistance, antennaLowCutoff, antennaBandwidth):
    noiseTriggerDecisions = []
    for i in range(numberOfTests):
        noise = SPF.GenerateVoltageNoise(sampleRate=sampleRate,
                                         numSamples=len(times),
                                         temp=temperature,
                                         resistance=resistance,
                                         antennaLowCutoff=antennaLowCutoff,
                                         antennaBandwidth=antennaBandwidth,
                                         hidePrints=True)

        peakCorrelations = []
        for template in templates:
            # ::33 to emulate using multiple clock cycles for a single MF calculation
            peakCorrelations.append(numpy.amax(
                abs(correlate(noise, template)[::129])**0.25))
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
        if i == 0:
            print("0%")
        if numberOfTests/(i+1) == 2:
            print("50%")
        if numberOfTests/(i+1) == 1:
            print("100%")

        randomStartFrequency = numpy.random.uniform(
            startFrequencyBounds[0], startFrequencyBounds[1])
        #print("Random start frequency:",randomStartFrequency)
        randomFrequencyGradient = numpy.random.uniform(
            frequencyGradientBounds[0], frequencyGradientBounds[1])
        #print("Random frequency gradient:", randomFrequencyGradient)
        randomPhase = numpy.random.uniform(0, 2*numpy.pi)
        # print(randomPhase)
        signal = signalAmplitude * \
            SPF.GenerateChirpSignal(
                times, randomStartFrequency, randomFrequencyGradient, randomPhase)
        # cleanSignal = copy.deepcopy(signal)
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

        # ### Aternative Time Shifting ###
        # emptyArray = numpy.zeros(len(times) + 2*len(times)-2)
        # middleStartPoint = len(times)-1
        # middleEndPoint = -len(times)+1
        # signalStartTime = numpy.random.randint(len(emptyArray)-len(times))
        # emptyArray[signalStartTime:signalStartTime+len(times)] += signal
        # signal = emptyArray[middleStartPoint : middleEndPoint]

        ### Third time shifting method (random full signal within interval) ###
        indexStart = 0
        indexEnd = len(times)
        # shift within the first half so the whole signal is still present
        randomIndex = numpy.random.randint(indexStart, indexEnd//2)
        if randomIndex != 0:
            zerosForShift = numpy.zeros(randomIndex)
            signal = numpy.insert(signal, 0, zerosForShift)
            signal = signal[:-randomIndex]
            #print("Signal check:", signal[randomIndex-2:randomIndex+3])

        # noise = SPF.GenerateVoltageNoise(sampleRate = sampleRate,
        #                                  numSamples = len(times),
        #                                  temp = temperature,
        #                                  resistance = resistance,
        #                                  antennaLowCutoff = antennaLowCutoff,
        #                                  antennaBandwidth = antennaBandwidth,
        #                                  hidePrints = True)

        # noise = SPF.GenerateWhiteVoltageNoise(sampleRate, len(times), 290e-3, antennaLowCutoff, antennaBandwidth, hidePrints=True)
        # print(times[-1])
        noise = SPF.GenerateAdderSpectrumNoise(
            times[-1], sampleRate, noiseRMS=0.290)

        # print("sig:",len(signal))
        # print(len(noise))
        # signal = signal[:-1]

        if len(signal) != len(noise):
            if len(signal) > len(noise):
                signal = signal[:len(noise)]
            else:
                noise = noise[:len(signal)]

        # print("SNR:", SPF.GetRMS(signal) / SPF.GetRMS(noise))
        signal += noise
        peakCorrelations = []

        testOutput = False
        if testOutput == True and i == 0:
            signalNoNoise = signalAmplitude * \
                SPF.GenerateChirpSignal(times, 280.0e3, 3.45e8, 0)
            exampleCorrelation = abs(
                correlate(signal, templates[0])[::17])**0.25
            exampleCorrelationNoise = abs(
                correlate(noise, templates[0])[::17])**0.25
            fullTemplateTimes = numpy.arange(
                0, 1/sampleRate*129 * len(exampleCorrelation), step=1/sampleRate*129)
            fullTemplateTimesNoise = numpy.arange(
                0, 1/sampleRate*129 * len(exampleCorrelationNoise), step=1/sampleRate*129)
            if len(fullTemplateTimes) > len(exampleCorrelation):
                fullTemplateTimes = fullTemplateTimes[:-1]
            figTestOutput, axTestOutput = pyplot.subplots(
                2, 1, figsize=[16, 16])
            axTestOutput[0].step(fullTemplateTimes, exampleCorrelation)
            axTestOutput[0].set_title("Test Correlation")
            axTestOutput[1].step(fullTemplateTimesNoise,
                                 exampleCorrelationNoise)

            cleanCorrelation = abs(
                correlate(signalNoNoise, templates[0])) ** 0.25
            cleanCorrelationTimes = numpy.arange(
                0, 1/sampleRate * len(cleanCorrelation), step=1/sampleRate)
            maxCorrelationIndex = numpy.argmax(cleanCorrelation)
            figCloseResponse, axCloseResponse = pyplot.subplots(
                2, 1, figsize=[16, 16])
            pointsToSee = int(1e-5*sampleRate)
            print("Points to see:", pointsToSee)
            print(len(cleanCorrelation))
            print(maxCorrelationIndex)
            if maxCorrelationIndex-pointsToSee < 0:
                lowIndex = 0
            else:
                lowIndex = maxCorrelationIndex-pointsToSee
            if maxCorrelationIndex+pointsToSee < len(cleanCorrelation):
                highIndex = maxCorrelationIndex+pointsToSee
            else:
                highIndex = -1

            pointsToSeeWide = int(1e-4*sampleRate)
            if maxCorrelationIndex-pointsToSeeWide < 0:
                lowIndexWide = 0
            else:
                lowIndexWide = maxCorrelationIndex-pointsToSeeWide
            if maxCorrelationIndex+pointsToSeeWide < len(cleanCorrelation):
                highIndexWide = maxCorrelationIndex+pointsToSeeWide
            else:
                highIndexWide = -1
            axCloseResponse[0].plot(
                cleanCorrelationTimes[lowIndex: highIndex], cleanCorrelation[lowIndex: highIndex])
            axCloseResponse[1].plot(cleanCorrelationTimes[lowIndexWide: highIndexWide],
                                    cleanCorrelation[lowIndexWide: highIndexWide])

            maxima = []
            sampleRates = []
            absoluteMax = numpy.max(cleanCorrelation)
            for i in range(1000):
                maxima.append(
                    numpy.max(cleanCorrelation[::(i+1)]) / absoluteMax)
                sampleRates.append(
                    1/(cleanCorrelationTimes[i+1]-cleanCorrelationTimes[0]))
            figSampleRate, axSampleRate = pyplot.subplots(
                1, 1, figsize=[16, 8])
            axSampleRate.plot(sampleRates, maxima)
            axSampleRate.set_xscale("log")
            axSampleRate.set_xlabel("Sample Rate (Hz)")
            axSampleRate.set_ylabel("Fraction of actual Maximum")
            # this should be one less than the number used for skipping points
            axSampleRate.axvline(sampleRates[130])

            fullMaximum = numpy.amax(cleanCorrelation)
            lowerMaximum = numpy.amax(cleanCorrelation[::17])
            print("Full maximum:", fullMaximum, "\nLow Maximum:", lowerMaximum)

        for template in templates:
            # figTemplateTest, axTemplateTest = pyplot.subplots(4, 1, figsize=[16,32])
            # axTemplateTest[0].plot(times[:len(template)],template)
            # axTemplateTest[0].set_xlim(0,1e-5)
            # FirstFFTFreqs, FirstFFTValues = SPF.FourierTransformWaveform(signal[:1000], times[:1000])
            # LastFFTFreqs, LastFFTValues = SPF.FourierTransformWaveform(signal[-1000:], times[-1000:])
            # fullFFTFreqs, fullFFTValues = SPF.FourierTransformWaveform(signal, times)
            # fullFFTFreqsClean, fullFFTValuesClean = SPF.FourierTransformWaveform(cleanSignal, times)
            # axTemplateTest[1].plot(FirstFFTFreqs, FirstFFTValues)
            # axTemplateTest[1].axvline(280000, linestyle='--')
            # axTemplateTest[2].plot(LastFFTFreqs, LastFFTValues)
            # axTemplateTest[2].axvline(625000, linestyle='--')
            # axTemplateTest[3].plot(fullFFTFreqs[:-1], fullFFTValues)
            # axTemplateTest[3].plot(fullFFTFreqsClean, fullFFTValuesClean)

            peakCorrelations.append(numpy.amax(
                ((correlate(signal, template)[::3])**2)**0.25))

        matchedFilterScores.append(numpy.amax(peakCorrelations))
    return matchedFilterScores


def GetNoiseMFScoreForTrigger(numberOfTests, templates, sampleRate, times, startFrequencyBounds, frequencyGradientBounds, signalAmplitude, temperature, resistance, antennaLowCutoff, antennaBandwidth):
    matchedFilterScores = []
    for i in range(numberOfTests):
        if i == 0:
            print("0%")
        if numberOfTests/(i+1) == 2:
            print("50%")
        if numberOfTests/(i+1) == 1:
            print("100%")
        # noise = SPF.GenerateVoltageNoise(sampleRate = sampleRate,
        #                                  numSamples = len(times),
        #                                  temp = temperature,
        #                                  resistance = resistance,
        #                                  antennaLowCutoff = antennaLowCutoff,
        #                                  antennaBandwidth = antennaBandwidth,
        #                                  hidePrints = True)

        # noise = SPF.GenerateWhiteVoltageNoise(sampleRate, len(times), 290e-3, antennaLowCutoff, antennaBandwidth, hidePrints=True)
        noise = SPF.GenerateAdderSpectrumNoise(
            times[-1], sampleRate, noiseRMS=0.290)

        peakCorrelations = []
        for template in templates:
            peakCorrelations.append(numpy.amax(
                ((correlate(noise, template)[::3])**2)**0.25))
        matchedFilterScores.append(numpy.amax(peakCorrelations))
    return matchedFilterScores


def TriggerStatisticsTest():
    # Generate a bunch of signals within expected frequency gradient/start frequency ranges (maybe antenna bandwidth limited for the start frequencies)
    numberOfTests = 1200
    # startFrequencyBounds = [20e6, 21e6-345000]
    startFrequencyBounds = [280e3, 280.000000001e3]
    frequencyGradientBounds = [3.45e8, 3.45000000001e8]

    sampleRate = 6.25e6
    times = numpy.arange(0, 2e-3, step=1/sampleRate)
    # signalAmplitude = 3.075179595089654e-08 * 2**0.5
    # signalAmplitude *= 1.6273627050985229801358942004775 /4*4 * 1 # Moderate increase from phased array
    signalAmplitude = 11e-3*numpy.sqrt(2)
    signalAmplitude *= 1.08*3  # some phased antenna improvements
    temperature = 8  # K
    antennaBandwidth = sampleRate/2-100000
    antennaLowCutoff = 1e0
    resistance = 73.2

    # print("SNR (RMS):", (signalAmplitude/numpy.sqrt(2)) / numpy.sqrt(4*Constants.k_B*temperature*resistance*antennaBandwidth))
    print("SNR (RMS):", (signalAmplitude / numpy.sqrt(2)) / 0.290)
    # startFrequencies = numpy.arange(19.5e6,20.5e6, step = 0.2e6)

    triggerThresholds = numpy.logspace(-5, 1, num=3000)
    correctSignalDetectionRates = []
    falseAlarmRates = []

    # templateStartFrequencies = numpy.linspace(20.1e6, 20.9e6, num=5)
    templateStartFrequencies = [280.0e3]
    templateFrequencyGradients = numpy.linspace(3.45e8, 3.45e8, num=1)
    templates = []
    templateTimes = numpy.arange(0, 1/sampleRate*2048, step=1/sampleRate)
    for i in range(len(templateStartFrequencies)):
        for j in range(len(templateFrequencyGradients)):
            templates.append(SPF.GenerateChirpSignal(
                templateTimes, templateStartFrequencies[i], templateFrequencyGradients[j], 0))

    # Do multiple tests for different lengths of signal
    maxTimes = numpy.linspace(1e-3, 1e-3, num=1)
    maxTimeLabels = []
    for i in range(len(maxTimes)):
        maxTimeLabels.append(maxTimes[i])
        maxTimeLabels[i] = "{:.1E}".format(maxTimeLabels[i])
    variedSignalLengthFARs = numpy.empty((len(maxTimes)), dtype='object')
    variedSignalLengthPCDs = numpy.empty((len(maxTimes)), dtype='object')

    for t in range(len(maxTimes)):
        times = numpy.arange(0, maxTimes[t], step=1/sampleRate)
        print("len(times) in loop:", len(times))

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
            correctSignalDetectionRate = numpy.count_nonzero(
                signalTriggerDecisions) / numberOfTests
            # print("Correct signal detection rate:", correctSignalDetectionRate)
            correctSignalDetectionRates.append(correctSignalDetectionRate)

            for MFScore in NoiseMFScore:
                # print(mean)
                if MFScore > triggerThreshold:
                    noiseTriggerDecisions.append(True)
                else:
                    noiseTriggerDecisions.append(False)
            falseAlarmRate = numpy.count_nonzero(
                noiseTriggerDecisions) / numberOfTests
            # print("False alarm rate:",falseAlarmRate)
            falseAlarmRates.append(falseAlarmRate)

        variedSignalLengthFARs[t] = falseAlarmRates
        variedSignalLengthPCDs[t] = correctSignalDetectionRates

    figDetectionRates, axDetectionRates = pyplot.subplots(
        1, 1, figsize=[18, 8])
    axDetectionRates.plot(triggerThresholds, correctSignalDetectionRates)
    axDetectionRates.set_xlabel("Trigger Threshold")
    axDetectionRates.set_ylabel("Correct Signal Detection Rate")

    figFAR, axFAR = pyplot.subplots(1, 1, figsize=[18, 8])
    axFAR.plot(triggerThresholds, falseAlarmRates)
    axFAR.set_xlabel("Trigger Threshold")
    axFAR.set_ylabel("False Alarm Rate")

    figPCDvsFAR, axPCDvsFAR = pyplot.subplots(1, 1, figsize=[18, 8])
    axPCDvsFAR.plot(falseAlarmRates, correctSignalDetectionRates)
    axPCDvsFAR.set_xlabel("False Alarm Rate")
    axPCDvsFAR.set_ylabel("Probability of correct detection")

    figVariedTimeROC, axVariedTimeROC = pyplot.subplots(1, 1, figsize=[16, 8])
    for i in range(len(maxTimes)):
        axVariedTimeROC.plot(
            variedSignalLengthFARs[i], variedSignalLengthPCDs[i], label=maxTimeLabels[i]+"s")
    axVariedTimeROC.legend(loc="lower right")
    axVariedTimeROC.set_xlabel("False Alarm Rate")
    axVariedTimeROC.set_ylabel("Correct Signal Detection Rate")

    return 0


def MatchedFilterFrequencyResolution(hidePrints=False):
    sampleRate = 1000e6
    times = numpy.arange(0, 0.001, 1/sampleRate)
    # times = numpy.arange(0, 727.567e-9, 1/sampleRate)
    signalAmplitude = 2 * 11e-3 * numpy.sqrt(2) #3.075179595089654e-08 * 2**0.5
    startFrequency = 600000
    frequencyGradient = 3.45e8
    signal = signalAmplitude * \
        SPF.GenerateChirpSignal(times, startFrequency, frequencyGradient)

    signal[len(signal)//1:] = 0

    temperature = 8
    resistance = 73.2
    antennaLowCutoff = 0.1
    antennaBandwidth = 15e6

    # noise = SPF.GenerateVoltageNoise(sampleRate=sampleRate,
    #                                  numSamples=len(times),
    #                                  temp=temperature,
    #                                  resistance=resistance,
    #                                  antennaLowCutoff=antennaLowCutoff,
    #                                  antennaBandwidth=antennaBandwidth,
    #                                  hidePrints=True)
    
    
    noise = SPF.GenerateAdderSpectrumNoise(times[-1], sampleRate=sampleRate, noiseRMS=0.290)

    if len(noise) != len(signal):
        if len(noise) > len(signal):
            noise = noise[:len(signal)]
        else:
            signal = signal[:len(noise)]

    if hidePrints == False:
        print("Signal/Noise ratio (RMS):", SPF.GetRMS(signal)/SPF.GetRMS(noise))
    signal += noise

    template = 1 * \
        SPF.GenerateChirpSignal(times, startFrequency, frequencyGradient)
        
    template = template[:len(template)//32] # Make the template shorter 
    matchedFilterResult = correlate(signal, template)

    #xShifts = numpy.linspace(-1e-3, 1e-3, num=len(matchedFilterResult))
    # if hidePrints == False:
    #     print("Lengths:")
    #     print(len(signal))
    #     print(len(template))
    #     print(len(matchedFilterResult))
    #     print(len(xShifts))
    #     figMF, axMF = pyplot.subplots(1, 1, figsize = [16,8])
    #     axMF.plot(xShifts, matchedFilterResult)


    # Calculate theoretical full response width:
    lowerResponseFreq = startFrequency + frequencyGradient * 0.0005 - frequencyGradient*1e-3
    highResponseFreq = startFrequency
    lowerResponseDelta = lowerResponseFreq-startFrequency
    highResponseDelta = startFrequency-startFrequency


    # Now repeat for many templates of varying frequency
    matchedFilterPeaks = []
    noisePeaks = []
    frequencyScanLow = startFrequency-400e3
    frequencyScanHigh = startFrequency+400e3
    numFreqsToScan = 2000
    frequencyStep = (frequencyScanHigh-frequencyScanLow) / numFreqsToScan
    frequencies = numpy.arange(
        frequencyScanLow, frequencyScanHigh, step=frequencyStep)
    for frequency in frequencies:
        template = 1 * \
            SPF.GenerateChirpSignal(times, frequency, frequencyGradient)
        matchedFilterResult = correlate(signal, template)
        matchedFilterPeaks.append(numpy.amax(matchedFilterResult))
        noiseResult = correlate(noise, template)
        noisePeaks.append(numpy.amax(noiseResult))

    # if hidePrints == False:
    #     figFreqRes, axFreqRes = pyplot.subplots(1, 1, figsize=[16,8])
    #     axFreqRes.plot(frequencies, matchedFilterPeaks)
    #     axFreqRes.set_xlabel("Frequency (Hz)")
    #     axFreqRes.set_ylabel("Matched Filter Response")

    timeConstantInSamples = 1e-11 * sampleRate
    movingAverageFilter = SPF.SinglePoleIIR(timeConstantInSamples)

    smoothedMatchedFilterPeaks = movingAverageFilter.ApplyFilter(matchedFilterPeaks)
    smoothedNoisePeaks = movingAverageFilter.ApplyFilter(noisePeaks)

    smoothedMatchedFilterPeaks = matchedFilterPeaks
    smoothedNoisePeaks = noisePeaks


    if hidePrints == False:
        print(len(smoothedMatchedFilterPeaks))

        figFreqResSmoothed, axFreqResSmoothed = pyplot.subplots(1, 1, figsize=[
                                                                16, 8])
        axFreqResSmoothed.plot(frequencies-startFrequency,
                               smoothedMatchedFilterPeaks, label="Signal+Noise")
        axFreqResSmoothed.plot(frequencies-startFrequency, smoothedNoisePeaks, label="Noise Only")
        axFreqResSmoothed.legend()
        axFreqResSmoothed.set_xlabel(r"$\Delta f$ (Hz)")
        axFreqResSmoothed.set_ylabel("Matched Filter Response")
        # pyplot.close(figFreqResSmoothed)
        
        #axFreqResSmoothed.axvline(lowerResponseDelta, color="orange")
        axFreqResSmoothed.axvline(highResponseDelta, color="orange")

    if numpy.argmax(smoothedNoisePeaks) > 3*len(frequencies)/4 and hidePrints != False:
        figFreqResSmoothed, axFreqResSmoothed = pyplot.subplots(1, 1, figsize=[
                                                                16, 8])
        axFreqResSmoothed.plot(frequencies-startFrequency,
                               smoothedMatchedFilterPeaks, label="Signal+Noise")
        axFreqResSmoothed.plot(
            frequencies, smoothedNoisePeaks, label="Noise Only")
        axFreqResSmoothed.legend()
        axFreqResSmoothed.set_xlabel(r"$\Delta f$ (Hz)")
        axFreqResSmoothed.set_ylabel("Matched Filter Response")
        # figFreqResSmoothed.close()
    # print(frequencies[numpy.argmax(smoothedMatchedFilterPeaks)])

    return frequencies[numpy.argmax(smoothedMatchedFilterPeaks)]


def TestFrequencyResolution():
    # frequencies = numpy.arange(440000, 560000, step=2000) # make sure this matches frequencyResolution function's frequencies
    frequenciesAtMaximum = []
    for i in range(10):
        frequenciesAtMaximum.append(
            MatchedFilterFrequencyResolution(hidePrints=False))

    figAltFreqRes, axAltFreqRes = pyplot.subplots(1, 1, figsize=[16, 8])
    axAltFreqRes.hist(frequenciesAtMaximum)
    return 0


def TestShortSignal():
    sampleRate = 125e6
    # times = numpy.arange(0, 0.001, 1/sampleRate)
    times = numpy.arange(0, 727.567e-9, 1/sampleRate)
    signalAmplitude = 3.075179595089654e-08 * 2**0.5
    startFrequency = 500000
    frequencyGradient = 3.45e8
    signal = signalAmplitude * \
        SPF.GenerateChirpSignal(times, startFrequency, frequencyGradient)

    temperature = 8
    resistance = 73.2
    antennaLowCutoff = 0.1
    antennaBandwidth = 15e6

    noise = SPF.GenerateVoltageNoise(sampleRate=sampleRate,
                                     numSamples=len(times),
                                     temp=temperature,
                                     resistance=resistance,
                                     antennaLowCutoff=antennaLowCutoff,
                                     antennaBandwidth=antennaBandwidth,
                                     hidePrints=False)
    if len(noise) != len(signal):
        if len(noise) > len(signal):
            noise = noise[:-1]
        else:
            signal = signal[:-1]

    print("Signal/Noise ratio (RMS):", SPF.GetRMS(signal)/SPF.GetRMS(noise))
    signal += noise
    template = 1 * \
        SPF.GenerateChirpSignal(times, startFrequency, frequencyGradient)
    matchedFilterResult = correlate(signal, template)
    noiseResult = correlate(noise, template)
    xShifts = numpy.linspace(-727.567e-9, 727.567e-9,
                             num=len(matchedFilterResult))

    figMF, axMF = pyplot.subplots(1, 1, figsize=[16, 8])
    axMF.plot(xShifts, matchedFilterResult, label="Signal+Noise")
    axMF.plot(xShifts, noiseResult, label="Noise")
    axMF.legend()

    return 0


def MFRateTest():
    sampleRate = 6.25e6  # max sample rate, needs to be high to catch all the detail
    times = numpy.arange(0, 2e-3, step=1/sampleRate)
    signalAmplitude = 3.075179595089654e-08 * 2**0.5
    signalAmplitude *= 1.6273627050985229801358942004775 / \
        4*3  # Moderate increase from phased array

    signalNoNoise = signalAmplitude * \
        SPF.GenerateChirpSignal(times, 280.5e3, 3.45e8, 0)

    templateStartFrequencies = [200e3]
    templateFrequencyGradients = [3.45e8]
    templates = []
    templateTimes = numpy.arange(0, 1/sampleRate*1024, step=1/sampleRate)
    for i in range(len(templateStartFrequencies)):
        for j in range(len(templateFrequencyGradients)):
            templates.append(SPF.GenerateChirpSignal(
                templateTimes, templateStartFrequencies[i], templateFrequencyGradients[j], 0))

    numberOfMFTests = 1000
    numberOfSampleRates = 10000
    sampleRates = []
    maxima = numpy.zeros((numberOfMFTests, numberOfSampleRates))
    for i in range(numberOfMFTests):
        randomFreq = numpy.random.uniform(200e3, 280e3)
        randomPhase = numpy.random.uniform(0, numpy.pi)
        signalNoNoise = signalAmplitude * \
            SPF.GenerateChirpSignal(times, randomFreq, 3.45e8, randomPhase)

        cleanCorrelation = abs(correlate(signalNoNoise, templates[0])) ** 0.25
        cleanCorrelationTimes = numpy.arange(
            0, 1/sampleRate * len(cleanCorrelation), step=1/sampleRate)

        absoluteMax = numpy.max(cleanCorrelation)

        for j in range(numberOfSampleRates):
            # maxima.append( numpy.max( cleanCorrelation[::(i+1)] ) / absoluteMax )
            if i == 0:
                sampleRates.append(
                    1/(cleanCorrelationTimes[j+1]-cleanCorrelationTimes[0]))
                if j == 32:
                    print(sampleRates[j])

            maxima[i, j] = numpy.max(cleanCorrelation[::(j+1)]) / absoluteMax

    maximaAveraged = []
    standardDeviations = []
    for j in range(numberOfSampleRates):
        maximaAveraged.append(numpy.average(maxima[:, j]))
        standardDeviations.append(numpy.std(maxima[:, j]))
    figMFRateTest, axMFRateTest = pyplot.subplots(3, 1, figsize=[16, 24])
    axMFRateTest[0].plot(sampleRates, maximaAveraged)
    # print(sampleRates)
    axMFRateTest[0].set_xlabel("Sample Rate")
    axMFRateTest[0].set_xscale("log")
    axMFRateTest[0].set_ylabel("Fraction of Full Maximum")
    axMFRateTest[1].plot(sampleRates, standardDeviations)
    axMFRateTest[1].set_xscale("log")
    axMFRateTest[1].set_xlabel("Sample Rate")
    axMFRateTest[1].set_ylabel("Standard Deviation")
    # axMFRateTest[2].plot(sampleRates, maximaAveraged)
    axMFRateTest[2].errorbar(sampleRates, maximaAveraged,
                             yerr=standardDeviations, ecolor="black")
    axMFRateTest[2].set_xscale("log")
    axMFRateTest[2].set_xlabel("Sample Rate")
    axMFRateTest[2].set_ylabel("Fraction of Full Maximum")
    axMFRateTest[2].axvline(6.25e6/33)
    axMFRateTest[2].axvline(6.25e6/65)
    axMFRateTest[2].axvline(6.25e6/129)


def MFTimeResponse():
    sampleRate = 125e6# 6.25e6*60  # max sample rate, needs to be high to catch all the detail
    times = numpy.arange(0, 0.5e-3, step=1/sampleRate)
    signalAmplitude = 11e-3 * numpy.sqrt(2) #3.075179595089654e-08 * 2**0.5
    # signalAmplitude *= 1.6273627050985229801358942004775 /4*3 # Moderate increase from phased array
    signalNoNoise = signalAmplitude * SPF.GenerateChirpSignal(times, 300e3, 3.45e8, 0)

    templateMaxTime = 1e-3# 0.16384e-3  # 0.16384e-3
    templateTimes = numpy.arange(0, templateMaxTime, step=1/sampleRate)
    #templateTimesLong = numpy.arange(0, 0.32768e-3, step=1/sampleRate)
    template = SPF.GenerateChirpSignal(templateTimes, 300e3, 3.45e8, 0)
    print(len(signalNoNoise))
    print(len(template))
    #templateLong = SPF.GenerateChirpSignal(templateTimesLong, 280e3, 3.45e8, 0)
    # abs( correlate(signalNoNoise, template) ) ** 0.25
    
    cleanCorrelation = abs(correlate(signalNoNoise, template))#**0.5
    
    
    #cleanCorrelationLong = abs(correlate(signalNoNoise, templateLong))**0.5
    correlationTimeMax = 1/sampleRate * len(cleanCorrelation) / 2
    #longCorrelationTimeMax = 1/sampleRate * len(cleanCorrelationLong) / 2
    cleanCorrelationTimes = numpy.arange(-correlationTimeMax,
                                         correlationTimeMax, step=1/sampleRate)
    #cleanCorrelationTimesLong = numpy.arange(-longCorrelationTimeMax, longCorrelationTimeMax, step=1/sampleRate)
    
    
    figMFTimeTest, axMFTimeTest = pyplot.subplots(1, 1, figsize=[16, 8])
    cleanCorrelationLength = min(len(cleanCorrelationTimes), len(cleanCorrelation))
    #cleanCorrelationLengthLong = min(len(cleanCorrelationTimesLong), len(cleanCorrelationLong))
    cleanCorrelationTimes =  cleanCorrelationTimes[:cleanCorrelationLength]
    cleanCorrelation = cleanCorrelation[:cleanCorrelationLength]
    #cleanCorrelationTimesLong = cleanCorrelationTimesLong[:cleanCorrelationLengthLong]
    #cleanCorrelationLong = cleanCorrelationLong[:cleanCorrelationLengthLong]

    #axMFTimeTest.set_xlabel("Sample Rate")
    # axMFtimeTest.set_xscale("log")
    #axMFtimeTest.set_ylabel("Fraction of Full Maximum")
    axMFTimeTest.plot(cleanCorrelationTimes*1000, cleanCorrelation)#,color="b", alpha=0.5, label="1024pt")
    #axMFTimeTest.plot(cleanCorrelationTimesLong-0.000082, cleanCorrelationLong, color="orange", alpha=0.5, label="2048pt")
    #axMFTimeTest.scatter(cleanCorrelationTimes[::17*60*1], cleanCorrelation[::17*60*1], color="b")
    # axMFTimeTest.scatter(cleanCorrelationTimesLong[320::33*60*1]-0.000082, cleanCorrelationLong[320::33*60*1], color="orange")
    #axMFTimeTest.set_xlim(-0.00049, -0.00035)
    # axMFtimeTest.set_xscale("log")
    axMFTimeTest.set_xlabel("Correlation Time (ms)")
    axMFTimeTest.set_ylabel("Matched Filter Response")
    axMFTimeTest.legend()

    noise = SPF.GenerateAdderSpectrumNoise(times[-1], sampleRate, noiseRMS=0.290)# signalAmplitude/numpy.sqrt(2)*(1/0.16))
    if len(noise)>len(signalNoNoise):
        noise = noise[:len(signalNoNoise)]
    elif len(signalNoNoise)>len(noise):
        signalNoNoise = signalNoNoise[:len(noise)]
    
    
    signalWithNoise = signalNoNoise+noise
    rawCorrelation = correlate(signalWithNoise, template)
    rawCorrelationTimeMax = 1/sampleRate * len(rawCorrelation) / 2
    rawCorrelationTimes = numpy.arange(-rawCorrelationTimeMax,
                                       rawCorrelationTimeMax, step=1/sampleRate)
    figRaw, axRaw = pyplot.subplots(1, 1, figsize=[16, 8])
    axRaw.plot((rawCorrelationTimes)*1000, abs(rawCorrelation))
    #axRaw.set_xlim(-1e-1, 1e-1)
    axRaw.set_xlabel("Correlation Time (ms)")
    axRaw.set_ylabel("Matched Filter Response")
    for label in axRaw.get_xaxis().get_ticklabels()[1::2]:
        label.set_visible(False)
    axRaw.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    figRaw.savefig(
        "SavedGraphs/MatchedFilterWith0.16SNRNoise0.16384e-3sTemplate+Signal.pdf", format="pdf")
    return 0


if __name__ == "__main__":
    # TestMatchedFilterTrigger()

    # TriggerStatisticsTest() # Use this one for trigger analysis

    MatchedFilterFrequencyResolution(hidePrints=False)
    # TestFrequencyResolution()
    # TestShortSignal()
    # MFRateTest()
    # MFTimeResponse()
