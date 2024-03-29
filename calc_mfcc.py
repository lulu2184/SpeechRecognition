import numpy as np
from scipy.fftpack import dct
import math
import os
import dill

def calc_mfcc(waveData, framerate):
	frameSize = len(waveData)
	ham = np.hamming(frameSize)
	
	curFrame = waveData
	curFrame = curFrame * ham
	curSTFT = np.abs(np.fft.fft(curFrame))
	filteredSpectrum = np.dot(curSTFT, mel_group(20, frameSize, framerate)) # with same size
	logSpectrum = np.log(filteredSpectrum)
	dctSpectrum = dct(logSpectrum)
	return dctSpectrum[:16]

def calc_mfcc_with_mel(waveData, mel):
	frameSize = len(waveData)
	ham = np.hamming(frameSize)
	
	curFrame = waveData
	curFrame = curFrame * ham
	curSTFT = np.abs(np.fft.fft(curFrame))
	filteredSpectrum = np.dot(curSTFT, mel) # with same size
	logSpectrum = np.log(filteredSpectrum)
	dctSpectrum = dct(logSpectrum)
	return dctSpectrum[:16]

def mel_group(m, frameSize, framerate):
	if frameSize == 256 and os.path.exists('train_result/mel_group'):
		f = open('train_result/mel_group', 'r')
		tt = dill.loads(f.read())
		f.close()
		return tt

	fmax = 4000.0
	fmin = 50.0
	Bfmax = mel_feq2mel(fmax)
	Bfmin = mel_feq2mel(fmin)
	# fc_array = [float(frameSize) / framerate * mel_b2feq(Bfmin + i * (Bfmax - Bfmin) / (m + 1)) for i in range(m + 1)]
	fc_array = [Bfmin + i * (Bfmax - Bfmin) / (m + 1) for i in range(m + 2)]
	fc_array = [mel_mel2feq(v) for v in fc_array]
	melFilter = []
	for i in range(1, m + 1, 1):
		newFilter = []
		for k in range(frameSize):
			fk = k * float(framerate) / frameSize
			if fk >= fc_array[i - 1] and fk <= fc_array[i]:
				newFilter.append((fk - fc_array[i - 1]) / (fc_array[i] - fc_array[i - 1]))
			elif fk > fc_array[i] and fk <= fc_array[i + 1]:
				newFilter.append((fc_array[i + 1] - fk) / (fc_array[i + 1] - fc_array[i]))
			else:
				newFilter.append(0)
		melFilter.append(newFilter)
	tt = np.transpose(np.array(melFilter))

	if frameSize == 256:
		f = open('train_result/mel_group', 'w')
		f.write(dill.dumps(tt))
		f.close()
	return tt

def mel_feq2mel(freq):
	return 2595 * np.log10(1 + freq / 700.0) 

def mel_mel2feq(b):
	return (10 ** (b / 2595.0) - 1) * 700

def feature_extractor(waveData, framerate):
	step = 64
	frameSize = 256
	mel = mel_group(20, frameSize, framerate)
	wlen = len(waveData)
	frameNum = int(math.ceil((wlen - frameSize) * 1.0 / step))
	feature_array = []
	for i in range(frameNum - 1):
		if i * step + frameSize < wlen:
			feature = calc_mfcc_with_mel(waveData[np.arange(i * step, i * step + frameSize)], mel)
		else:
			feature = calc_mfcc(waveData[np.arange(i * step, wlen)], framerate)
		feature_array.append(feature)
	# calculate Delta mfcc
	return feature_array




		




