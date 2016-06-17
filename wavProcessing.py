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
			return digitalSignal, samplingRate	



# The below method writes the data to a new WAV file
def write_wav_file(fileName, digitalSignal, samplingRate, directory):
	filePath = directory + fileName
	wav.write(filePath, samplingRate, digitalSignal)



# The below method plots the samples of the WAV file 
def plot_wav_file(wavFileData, samplingRate) :
	timeRange = np.arange(wavFileData.size)/float(samplingRate)
	plt.plot(timeRange, wavFileData)
	plt.show()			



def plot_fft(fftData, signalSize, plotTitle) :
	dataPoints = len(fftData)

	freqRange = np.arange(0, signalSize, float(signalSize/dataPoints))

	plt.title(plotTitle)
	plt.plot(freqRange, fftData)
	plt.show()



# def wave_to_blocks(digitalSignal, samplingRate, clipLength) :
		


def normalize_float32(digitalSignal) :
	
	# Simply normalize each data point and return the array
	normalizedWave = digitalSignal.astype('float32')/32767.0
	
	return normalizedWave

def denormalize_float32(digitalSignal) :
	
	# Simply denormalize each data point and return the array
	denormalizedWave = digitalSignal.astype('float32')*32767.0 

	return denormalizedWave



# def blocks_to_training_examples(fftBlocks, clipLength, blockSize, samplingRate) :
	