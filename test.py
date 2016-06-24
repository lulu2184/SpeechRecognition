import math
import wave
import sys
import numpy as np
import pylab as pl
import range_detect as rd


if __name__ == "__main__":
	if len(sys.argv) != 2:
		print('wrong params')
		sys.exit()
	fw = wave.open(sys.argv[1], 'r')
	params = fw.getparams()
	print(params)
	nchannels, sampwidth, framerate, nframes = params[:4]
	strData = fw.readframes(nframes)
	orgData = np.fromstring(strData, dtype = np.int16)
	waveData = orgData - np.mean(orgData);
	waveData = waveData * 1.0 / max(abs(waveData))
	waveData = waveData[400:-400]
	nframes = len(waveData)
	fw.close()

	frameSize = 256
	step = 64
	volume11 = rd.calVolume(waveData, frameSize, step)
	volume12 = rd.calZeros(waveData, frameSize, step)

	time = np.arange(0, nframes) * (1.0 / framerate)
	time2 = np.arange(0, len(volume11)) * step * 1.0 / framerate

	# (start, end) = rd.range_detect(waveData)
	array = rd.range_detect(waveData, 1)

	pl.subplot(3, 1, 1)
	pl.plot(time, waveData, color="black")
	colors = ['-r', '-g', '-b', 'yellow']
	print array
	for (i, (start, end)) in enumerate(array):
		start = start * 1.0 / framerate
		end = end * 1.0 / framerate
		pl.plot([start, start], [-1, 1], colors[i])
		pl.plot([end, end], [-1, 1], colors[i])
	pl.ylabel("Amplitude")
	pl.subplot(3, 1, 2)
	pl.plot(time2, volume11)
	for (i, (start, end)) in enumerate(array):
		start = start * 1.0 / framerate
		end = end * 1.0 / framerate
		pl.plot([start, start], [0, 0.5], colors[i])
		pl.plot([end, end], [0, 0.5], colors[i])
	pl.ylabel("absSum")
	pl.subplot(3, 1, 3)
	pl.plot(time2, volume12)
	for (i, (start, end)) in enumerate(array):
		start = start * 1.0 / framerate
		end = end * 1.0 / framerate
		pl.plot([start, start], [0, 0.5], colors[i])
		pl.plot([end, end], [0, 0.5], colors[i])
	pl.ylabel("ZeroCount")
	pl.show()

	start, end = array[3]

	waveData = waveData[start:end]
	# mfcc_feature = calc_mfcc(waveData)

	f = wave.open(r"out.wav", "wb")

	f.setparams(params)
	f.setnframes(end - start);
	f.writeframes(orgData[start:end].tostring())

	f.close()