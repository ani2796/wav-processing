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

audioFiles = wP.get_audio_from_dir(sampleAudioDirectory)

print("The WAV files in the given folder are:\n")
print(audioFiles, "\n\n")


(inputData, outputData) = wP.fft_and_blocks_and_chunks(audioFiles, sampleAudioDirectory)

mean = np.mean(np.mean(inputData, axis=0), axis=0) #Mean across num examples and num timesteps
std = np.sqrt(np.mean(np.mean(np.abs(inputData-mean)**2, axis=0), axis=0)) # STD across num examples and num timesteps
std = np.maximum(1.0e-8, std) #Clamp variance if too tiny
inputData[:][:] -= mean #Mean 0
inputData[:][:] /= std #Variance 1
outputData[:][:] -= mean #Mean 0
outputData[:][:] /= std #Variance 1

np.save('YourMusicLibrary' +'_mean', mean)
np.save('YourMusicLibrary' +'_var', std)
np.save('YourMusicLibrary' +'_x', inputData)
np.save('YourMusicLibrary' +'_y', outputData)
print("Done!")