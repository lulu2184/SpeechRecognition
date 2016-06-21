import numpy as np

def calc_mfcc(waveData, frameSize, framerate):
	ham = np.hamming(frameSize)
	
	curFrame = waveData
	curFrame = curFrame * ham
	curSTFT = np.square(np.abs(np.fft(curFrame)))
	filteredSpectrum = np.dot(curSTFT, mel_group(20, frameSize, framerate)) # with same size
	logSpectrum = np.log(filteredSpectrum)
	dctSpectrum = dct(logSpectrum)
	return dctSpectrum[:16]

def mel_group(m, frameSize):
	fmax = 20000
	fmin = 20 
	Bfmax = mel_feq2bw(fmax)
	Bfmin = mel_feq2bw(fmin)
	fc = [frameSize * mel_bw2feq(Bfmin + i * (Bfmax - Bfmin) / (m + 1)) for i in range(m)]
	melFilter = np.zeros((m, frameSize))


def feature_extractor(waveData, framerate):
	step = 64
	frameSize = 256
	wlen = len(waveData)
	frameNum = int(math.ceil(wlen * 1.0 / step))
	feature_array = []
	for i in range(frameNum):
		feature = calc_mfcc(waveData[np.arange(i * step, min(i * step + frameSize, wlen))])
		feature_array.append(feature)
	return feature_array




		




