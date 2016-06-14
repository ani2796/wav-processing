# This is a module that provides processing of WAV files. This includes read, write, decompress
# mp3 and perform fourier transforms. The secret goal of this module is to figure out how to 
#convert a WAV file to a tensor file so that it becomes a valid input to a neural network. 

import os
import scipy.io.wavfile as wav 
import matplotlib.pyplot as plt
import numpy as np
import math

# The below method reads the data within a WAV file and returns a list of the data elements
def read_wav_file(fileName, directory) :
	for sampleFile in os.listdir(directory) :
		if sampleFile == fileName :
			filePath = directory + fileName
			(samplingRate, digitalSignal) = wav.read(filePath)
			return samplingRate, digitalSignal	

# The below method writes the data to a new WAV file
def write_wav_file(fileName, samplingRate, digitalSignal, directory):
	filePath = directory + fileName
	wav.write(filePath, samplingRate, digitalSignal)

# The below method plots the samples of the WAV file 
def plot_wav_file(wavFileData, samplingRate) :
	timeRange = np.arange(wavFileData.size)/float(samplingRate)
	plt.plot(timeRange, wavFileData)
	plt.show()			

# The below method takes a complex list as its input and plots the magnitude of the 
# transformed wave in the frequency domain
def plot_fft(fftData, samplingRate) :
	# Set time axis range for easier understanding
	timeRange = np.arange(fftData.size)/float(samplingRate)
	
	# Calculating the amplitude of each frequency
	amplitude = []
	for index in range(len(fftData)) :
		amplitude.append(math.sqrt(np.real(fftData[index])**2 + np.imag(fftData[index])**2))
	
	# Plotting the DFT	
	plt.plot(timeRange, amplitude)
	plt.show()


def wave_to_blocks(digitalSignal, samplingRate) :
	# We first zero pad the wave to make sure blocks are of even size. Zero padding also increases the 
	# number of data points in the time signal and hence the number of data points in the frequency 
	# spectrum (due to the DFT equation)   

	# index is used to loop through block sized chunks of the input data
	index = 0 
	blockSize = samplingRate/4
	blocks = []

	# Zero padding the signal before converting it into blocks
	remainingSize = int(len(digitalSignal)%blockSize)

	print("Original size:", len(digitalSignal))	
	digitalSignal = np.concatenate([digitalSignal, np.zeros(( (int(blockSize) - remainingSize), ))])
	"""for index in range(int(blockSize) - remainingSize) :
		digitalSignal = np.concatenate((digitalSignal, [0])) """
	print("Padded size: ", len(digitalSignal))


	# Actually "blocking" the signal
	index = 0
	while(index < len(digitalSignal)) :
		blocks.append(digitalSignal[index:index+blockSize])
		index += blockSize

	print("block shape: ", len(blocks))

	# Returning the list of all blocks
	return blocks