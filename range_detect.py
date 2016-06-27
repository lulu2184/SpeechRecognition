import math
import numpy as np

def calVolume(waveData, frameSize, step):
	wlen = len(waveData)
	frameNum = int(math.ceil(wlen * 1.0 / step))
	volume = np.zeros((frameNum, 1))
	for i in range(frameNum):
		curFrame = waveData[np.arange(i * step, min(i * step + frameSize, wlen))]
		curFrame = curFrame - np.mean(curFrame)
		volume[i] = np.sum(np.abs(curFrame)) / frameSize
	return volume

def calZeros(waveData, frameSize, step):
	wlen = len(waveData)
	frameNum = int(math.ceil(wlen * 1.0 / step))
	zeros = np.zeros((frameNum, 1))
	for i in range(frameNum):
		curFrame = waveData[np.arange(i * step, min(i * step + frameSize, wlen))]
		curFrame = curFrame - np.mean(curFrame);
		zeros[i] = np.sum(np.abs(np.sign(curFrame[0:-1]) - np.sign(curFrame[1::]))) / frameSize / 2
	return zeros

def calEnergy(waveData, frameSize, step):
	wlen = len(waveData)
	frameNum = int(math.ceil(wlen * 1.0 / step))
	energy = np.zeros((frameNum, 1))
	for i in range(frameNum):
		curFrame = waveData[np.arange(i * step, min(i * step + frameSize, wlen))]
		# curFrame = curFrame - np.median(curFrame)
		energy[i] = np.sum(np.square(curFrame)) / frameSize
	return energy

def range_detect(waveData, type):
	frameSize = 256
	step = 64
	volume = calVolume(waveData, frameSize, step)
	zeros = calZeros(waveData, frameSize, step)

	maxv = max(volume)
	avz = np.median(zeros)
	print avz

	z0 = 0.1 + avz 
	z1 = 0.15 + avz
	mh = 0.10 * maxv
	ml = 0.03 * maxv
	wlen = len(waveData)

	mh_array = [i for (i, w) in enumerate(volume) if w > mh]
	start = min(mh_array)
	end = max(mh_array)
	array = [(start * step, end * step + frameSize)]

	length = len(volume)

	start = min([i for (i, z) in enumerate(zeros) if z > z1 and i + 20 > start and i < start] + [start - 10])
	end = max([i for (i, z) in enumerate(zeros) if z > z1 and i - 20 < end and i > end] + [end + 10])
	array.append((start * step, end * step + frameSize))

	start = max([i for (i, w) in enumerate(volume) if w < ml and i + 10 > start and i < start] + [start - 5])
	end = min([i for (i, w) in enumerate(volume) if w < ml and i - 10 < end and i > end] + [end + 5])
	array.append((start * step, end * step + frameSize))

	start = max([i for (i, z) in enumerate(zeros) if z < z0 and i < start and i + 10 > start] + [start - 5])
	end = min([i for (i, z) in enumerate(zeros) if z < z0 and i > end and i - 10 < end] + [end + 5])
	array.append((start * step, end * step + frameSize))

	start = start * step
	end = end * step + frameSize
	start = max(start, 0)
	end = min(wlen, end)
	if type == 1:
		return array
	else:
		return (start, end)
