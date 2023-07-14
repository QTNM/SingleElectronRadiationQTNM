# SingleElectronRadiationQTNM
Python code describing the radiation of a single electron in motion.
Also contains analysis/trigger simulations.

SingleElectronRadiation contains some python functions/code to calculate the fields produced by an electron given its motion.

The analysis and trigger codes are split into matched filter and lock-in amplifier codes.

SignalProcessingFunctions contains functions that are generally useful for signal processing.

SpectrumAnalyser contains a few different methods for analysing spectrums of radiation, including an FFT "waterfall" type plot, consisting of groups of FFTs in order and the lock-in amplifier with a chirping reference signal.
