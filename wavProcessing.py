# This is a module that provides processing of WAV files. This includes read, write, decompress
# mp3 and perform fourier transforms. The secret goal of this module is to figure out how to 
#convert a WAV file to a tensor file so that it becomes a valid input to a neural network. 

import os
import scipy.io.wavfile as wav 
import matplotlib.pyplot as plt
import numpy as np
import math
import pyfftw


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



def signal_to_blocks(digitalSignal) :
	blockSize = 11025
	index = 0
	blocks = []

	while((index+blockSize)<len(digitalSignal)) :
		blocks.append(digitalSignal[index:index + blockSize])
		index += blockSize
	appendedZeros = np.zeros(blockSize - (len(digitalSignal) - index))
	lastBlock = np.concatenate((digitalSignal[index:len(digitalSignal)], appendedZeros), )	
	blocks.append(lastBlock)

	print("Size of new digital signal: ", len(blocks)*blockSize)

	return blocks



def blocks_to_examples(blocks) :
	trainingExamples = []
	blocksPerExample = 40
	index = 0

	while((index+blocksPerExample) < len(blocks)) :
		trainingExamples.append(blocks[index:index + blocksPerExample])
		index += blocksPerExample
		
		

def fft_and_blocks(audioFiles, sampleAudioDirectory) :

	fftBlocksInput = []
	fftBlocksOutput = []

	for fileName in audioFiles :
		(digitalSignal, samplingRate) = read_wav_file(fileName, sampleAudioDirectory)
		signalSize = len(digitalSignal)

		print("No. of audio samples in ", fileName, " is ", len(digitalSignal))

		inputBlocks = signal_to_blocks(digitalSignal)
		blockSize = len(inputBlocks[0])

		for block in inputBlocks :
			block = normalize_float32(block)		
			fftBlocksInput.append(pyfftw.interfaces.numpy_fft.fft(block))

		fftBlocksOutput = fftBlocksInput[1:]	
		fftBlocksOutput.append(np.zeros(blockSize))

		digitalSignal = normalize_float32(digitalSignal)
		fftSignal = pyfftw.interfaces.numpy_fft.fft(digitalSignal)
		
		paddedSignal = np.concatenate((digitalSignal, np.zeros(1000000)),) 
		fftSignalPadded = pyfftw.interfaces.numpy_fft.fft(paddedSignal)

		figure = plt.figure(1)

		freqRange = np.arange(44100)
		timeRange = np.arange(len(digitalSignal))
		paddedRange = np.arange(0, 44100, len(digitalSignal)/len(paddedSignal))

		plt.subplot(311)
		plt.plot(timeRange, digitalSignal)
		plt.title('Before FFT')
		plt.xlabel('time')
		plt.ylabel('pressure')
		
		plt.subplot(312)
		plt.plot(freqRange, np.real(fftSignal[:44100])**2 + np.imag(fftSignal[:44100])**2)
		plt.title('After FFT')
		plt.xlabel('frequency')
		plt.ylabel('amplitude')

		plt.subplot(313)
		plt.plot(paddedRange, np.real(fftSignalPadded[:len(paddedRange)])**2 + np.imag(fftSignalPadded[:len(paddedRange)])**2)
		plt.title('Padded signal\'s FFT')
		plt.xlabel('frequency')
		plt.ylabel('amplitude')

		figure.subplots_adjust(hspace=.75)

		plt.show()	

		print("\n\n")

		# write_wav_file(fileName[:-4] + " changed.wav", reformedSignal, samplingRate, sampleAudioDirectory)
		# os.remove(sampleAudioDirectory + fileName[:-4] + " changed.wav")


