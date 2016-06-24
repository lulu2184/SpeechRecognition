# -*- coding: utf-8 -*-
import os
import math
import wave
import sys
import hmm
import pickle
import numpy as np
import pylab as pl
import range_detect as rd
import calc_mfcc as mfcc
# import mfcc
import vector_quantization as vq


words = ['数字', '语音', '信号', '分析', '识别',
		 '北京', '背景', '上海', '商行', '复旦',
		'speech', 'voice', 'sound', 'happy', 'lucky',
		'file', 'open', 'close', 'start', 'stop']

def walk_dir(path):
	voice_list = [[] for i in range(20)]
	for fname in os.listdir(path):
		newpath = os.path.join(path, fname)
		if os.path.isdir(newpath):
			result = walk_dir(newpath)
			voice_list = [v1 + v2 for (v1, v2) in zip(voice_list, result)]
		else:
			for (i, keyword) in enumerate(words):
				if fname.find(keyword) >= 0:
					voice_list[i].append(newpath)
					break;
	return voice_list

def read_voice_file(fname):
	fname = unicode(fname, 'utf8')
	fw = wave.open(fname, 'r')
	params = fw.getparams()
	nframes = params[3]
	strData = fw.readframes(nframes)
	orgData = np.fromstring(strData, dtype = np.int16)
	waveData = orgData - np.mean(orgData);
	waveData = waveData * 1.0 / max(abs(waveData))
	fw.close()
	return (waveData, params)

if __name__ == "__main__":
	voice_list = walk_dir('dataset')
	train_set = []
	test_set = []
	for ele in voice_list:
		sp = len(ele) / 3 * 2
		train_set.append(ele[:sp])
		test_set.append(ele[sp:])
	for (i, word) in enumerate(words):
		vqs = vq.VQset();
		features_set = []
		for fname in train_set[i]:
			waveData, params = read_voice_file(fname)
			framerate = params[2]
			start, end = rd.range_detect(waveData, 0)
			waveData = waveData[start:end]
			features = mfcc.feature_extractor(waveData, framerate)
			vqs.add_samples(features)
			features_set.append(features)

		observations = []
		for features in features_set:
			observation = []
			for feature in features:
				observation.append(vqs.quantization(feature))
			observations.append(observation)

		model = hmm.HMM()
		model.pi = np.zeros(20)
		model.pi[0] = 1
		model.A = np.zeros((20, 20))
		for i in range(19):
			model.A[i][i] = 0.5
			model.A[i][i + 1] = 0.5
		model.A[19][19] = 1

		model.B = [[1.0/20 for j in range(vqs.n_clusters)] for i in range(20)]
		model.train(observations, 0.0001)
		picklestring = pickle.dump(model)
		dumpfile = open('train_result/' + word, "w")
		dumpfile.write(picklestring)






