# This module is a clay module used to test all other modules. It is subject to 
# change depending on my current project.

import wavProcessing as wp
import os
from scipy.fftpack import fft, ifft
import numpy as np
import pyfftw

audioFiles = []

# Storing names of all the WAV files in the audioFiles list and just displaying their names
for sampleFile in os.listdir("./Sample Audio") :
	if(sampleFile.endswith("wav")) :
		audioFiles.append(sampleFile)

print("The WAV files in the given folder are:\n")
print(audioFiles)

# Trying to simply read a wav file and store its sampling rate and waveform data
sampleAudioDirectory = './Sample Audio/'

for fileName in audioFiles :

	(samplingRate, digitalSignal) = wp.read_wav_file(fileName, sampleAudioDirectory)

	# Trying to plot the last stored wav file using matplotlib
	"""wp.plot_wav_file(digitalSignal, samplingRate)"""

	# Trying to write the last stored wav file into the Sample Audio directory
	"""wp.write_wav_file('written.wav', samplingRate, digitalSignal, './Sample Audio/')"""	

	# The number of seconds per training example
	clipLength = 10

	# Trying to transform the sample waveform into blocks
	digitalSignalBlocks = wp.wave_to_blocks(digitalSignal, samplingRate, clipLength)

	fftWave = []

	# Normalizing the input to make the project suitable for multiple audio files
	digitalSignal = wp.normalizing_float32(digitalSignal)

	print("Max element: ", digitalSignal.max())
	# Performing FFT on a sample audio wave
	for block in digitalSignalBlocks :
		fftWave.append(pyfftw.interfaces.numpy_fft.fft(block))

	"""print("order of fftWave: ", len(fftWave), " ", len(fftWave[0]))"""


	#Size of one FFT block
	blockSize = len(fftWave[0])

	wp.blocks_to_training_examples(fftWave, clipLength, blockSize, samplingRate)


	# Plotting the Frequency spectrum of the audio signal
	"""wp.plot_fft(fftWave, samplingRate)"""


	# Performing IFFT on a sample audio wave
	"""fftWaveInverse = pyfftw.interfaces.numpy_fft.ifft(fftWave)"""
