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

for fileName in audioFiles :
	(digitalSignal, samplingRate) = wP.read_wav_file(fileName, sampleAudioDirectory)
	signalSize = len(digitalSignal)

	print("No. of audio samples in ", fileName, " is ", len(digitalSignal))

	paddingZeros = np.zeros(2*len(digitalSignal))
	paddedSignal = np.concatenate((digitalSignal, paddingZeros), )

	print("Sample point ", 2000, " before fft for ", fileName, " is ", digitalSignal[2000])

	digitalSignal = wP.normalize_float32(digitalSignal)
	paddedSignal = wP.normalize_float32(paddedSignal)

	fftSignal1 = pyfftw.interfaces.numpy_fft.fft(digitalSignal)
	fftSignal2 = pyfftw.interfaces.numpy_fft.fft(paddedSignal)

	# wP.plot_fft(fftSignal1, signalSize, 'NORMAL')
	# wP.plot_fft(fftSignal2, signalSize, 'PADDED')

	plt.figure(1)

	normalFreq = np.arange(len(fftSignal1))
	plt.subplot(211)
	plt.plot(normalFreq, fftSignal1)
	
	paddedFreq = np.arange(0, len(fftSignal1), float(len(fftSignal1)/len(fftSignal2)))
	plt.subplot(212)
	plt.plot(paddedFreq, fftSignal2)

	plt.show()	

	"""print("No. of frequency points in the regular fft of ", fileName, " is ", len(fftSignal1))
	print("Sample frequency point ", 2000, " of ", fileName,  " has value ", fftSignal1[2000])
	print("No. of frequency points of the padded fft of ", fileName, " is ", len(fftSignal2))
	print("Sample frequency point ", 2000, " of ", fileName, " is ", fftSignal2[2000]) """

	reformedSignal1 = np.real(pyfftw.interfaces.numpy_fft.ifft(fftSignal1))
	reformedSignal2 = np.real(pyfftw.interfaces.numpy_fft.ifft(fftSignal2))

	reformedSignal1 = wP.denormalize_float32(reformedSignal1)
	reformedSignal2 = wP.denormalize_float32(reformedSignal2)

	"""print("No. of reconverted audio samples in ", fileName, " with no zero padding is ", len(reformedSignal1))
	print("No. of reconverted audio samples in ", fileName, " with zero padding is ", len(reformedSignal2), "\n\n")	"""	

	print("Sample point ", 2000, " after fft for ", fileName, " (without zero padding) is ", reformedSignal1[2000])
	print("Sample point ", 2000, " before fft for ", fileName, " (with zero padding) is ", reformedSignal2[2000], "\n\n")

	wP.write_wav_file(fileName[:-4] + " changed.wav", digitalSignal, samplingRate, sampleAudioDirectory)
	os.remove(sampleAudioDirectory + fileName[:-4] + " changed.wav")
