import numpy
import SpectrumAnalyser
import SignalProcessingFunctions as SPF
import matplotlib.pyplot as pyplot


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


def FrequencyResolutionTest(sampleRate = 125e6, signalAmplitude = 3.075179595089654e-08 * 2**0.5, temperature = 8, resistance = 73.2):
    lowFreq = 480000
    highFreq = 520000
    lowPassCutoff = 2000
    times = numpy.arange(0,0.001, step = 1/sampleRate)
    frequenciesToTest = numpy.linspace(lowFreq, highFreq, num=(highFreq-lowFreq)//lowPassCutoff * 50)
    signalChirp = GenerateRandomChirpSignal(sampleRate,
                                            times,
                                            startFrequencyBounds = [500000, 500000.001],
                                            frequencyGradientBounds = [3.45e8, 3.45000001e8],
                                            signalAmplitude = signalAmplitude,
                                            temperature = temperature,
                                            resistance = resistance,
                                            antennaLowCutoff = 0.1,
                                            antennaBandwidth = 1e7
                                            )
    
    lockInMeans = []
    for frequency in frequenciesToTest:
        lockInResponse = SpectrumAnalyser.SweepingLockInAmplifier(signalChirp,
                                                                  times,
                                                                  frequency,
                                                                  3.45e8,
                                                                  lowPassCutoff,
                                                                  sampleRate,
                                                                  showGraphs = False,
                                                                  filterType = "Butterworth")
        lockInMeans.append( numpy.mean( lockInResponse ) )
    
    figFreqRes, axFreqRes = pyplot.subplots(1, 1, figsize=[16,8])
    axFreqRes.plot(frequenciesToTest-500000, lockInMeans)
    axFreqRes.set_xlabel(r"$\Delta f$ (Hz)")
    axFreqRes.set_ylabel("Lock-in Response")
    
    
    
    
if __name__ == "__main__":
    
    
    print("Testing frequency resolution:")
    FrequencyResolutionTest()
    print("Finished frequency resolution test.")
    