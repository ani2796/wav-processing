# This module is a clay module used to test all other modules. It is subject to 
# change depending on my current project.

import wavProcessing as wP
import os
from scipy.fftpack import fft, ifft
import numpy as np
import pyfftw
import matplotlib.pyplot as plt

wP.process_audio_for_rnn()