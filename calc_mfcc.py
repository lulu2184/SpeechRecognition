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
	Bfmax = mel_feq2b(fmax)
	Bfmin = mel_feq2b(fmin)
	fc_array = [frameSize * mel_b2feq(Bfmin + i * (Bfmax - Bfmin) / (m + 1)) for i in range(m + 1)]
	melFilter = []
	for i in range(1, m, 1):
		newFilter = np.array([])
		for k in range(frameSize):
			if k >= fc_array(i - 1) and k <= fc_array(i):
				newFilter.append((k - fc_array(i - 1)) / (fc_array(i) - fc_array(i - 1)))
			elif k > fc_array(i) and k <= fc_array(i + 1):
				newFilter.append((fc_array(i + 1) - k) / (fc_array(i + 1) - fc_array(i)))
			else:
				newFilter.append(0)
		melFilter.append(newFilter)
	return melFilter

def mel_b2feq(freq):
	return 2595 * np.log(1 + freq / 700)

def mel_feq2b(b):
	return (10 ** (b / 2595.0) - 1) * 700

def feature_extractor(waveData, framerate):
	step = 64
	frameSize = 256
	wlen = len(waveData)
	frameNum = int(math.ceil(wlen * 1.0 / step))
	feature_array = []
	for i in range(frameNum):
		feature = calc_mfcc(waveData[np.arange(i * step, min(i * step + frameSize, wlen))])
		feature_array.append(feature)
	# calculate Delta mfcc
	return feature_array




		




