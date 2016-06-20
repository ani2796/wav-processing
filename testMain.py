# This module is a clay module used to test all other modules. It is subject to 
# change depending on my current project.

import wavProcessing as wP
import os
from scipy.fftpack import fft, ifft
import numpy as np
import pyfftw
import matplotlib.pyplot as plt

pyfftw.interfaces.cache.enable()

audioFiles = []

sampleAudioDirectory = './Sample Audio/'

for sampleFile in os.listdir(sampleAudioDirectory) :
	if(sampleFile.endswith(".wav")) :
		audioFiles.append(sampleFile)

print("The WAV files in the given folder are:\n")
print(audioFiles, "\n\n")

wP.fft_and_blocks(audioFiles, sampleAudioDirectory)
