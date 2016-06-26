# -*- coding: utf-8 -*-
import os
import math
import wave
import sys
import numpy as np
import pylab as pl
import range_detect as rd
import calc_mfcc as mfcc
import yahmm
import dill
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
				if fname.lower().find(keyword) >= 0:
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

def get_file_list(dir):
	voice_list = walk_dir(dir)
	train_set = []
	test_set = []
	for ele in voice_list:
		sp = len(ele) / 3 * 2
		train_set.append(ele[:sp])
		test_set.append(ele[sp:])
	trainstr = dill.dumps(train_set)
	f = open('train_result/train_set', 'w')
	f.write(trainstr)
	f.close()

	teststr = dill.dumps(test_set)
	f = open('train_result/test_set', 'w')
	f.write(teststr)
	f.close()

def pre_proc_train():
	f = open('train_result/train_set', 'r')
	setstr = f.read()
	train_set = dill.loads(setstr)

	for (i, word) in enumerate(words):
		if len(train_set[i]) == 0:
			continue
		features_set = []
		for fname in train_set[i]:
			print fname
			waveData, params = read_voice_file(fname)
			waveData = waveData[400:-400]
			framerate = params[2]
			start, end = rd.range_detect(waveData, 0)
			waveData = waveData[start:end]
			features = mfcc.feature_extractor(waveData, framerate)
			# vqs.add_samples(features)
			features_set.append(features)
		f = open('train_result/mfcc__' + str(i), 'w')
		f.write(dill.dumps(features_set))
		f.close()

def vq_train():
	for (i, word) in enumerate(words):
		f = open('train_result/mfcc__' + str(i), 'r')
		features_set = dill.loads(f.read())
		f.close()

		vqs = vq.VQset()
		total = len(features_set)
		for (j, features) in enumerate(features_set):
			print '[add samples][', word, ']', j, '/', total 
			vqs.add_samples(features)
		vqs.training()
		f = open('train_result/vq_' + str(i), 'w')
		f.write(dill.dumps(vqs))
		f.close()

		observations = []
		for (j,features) in enumerate(features_set):
			print '[observation][', word, ']', j, '/', total 
			observation = []
			for feature in features:
				dmin, choice = vqs.quantization(feature)
				observation.append(choice)
			observations.append(observation)
		f = open('train_result/observations_' + str(i), 'w')
		f.write(dill.dumps(observations))
		f.close()

def hmm_train():
	hh = []
	for (i, word) in enumerate(words):
		print 'Training ', i
		f = open('train_result/observations_' + str(i), 'r')
		observations = dill.loads(f.read())
		f.close()

		n_state = 10
		n_features = 50 #n_cluster
		model = yahmm.Model()
		state_set = []
		for j in range(n_state):
			state_set.append(yahmm.State(
				yahmm.DiscreteDistribution({k : 1.0/n_features for k in range(n_features)}),
				name = str(j)))
		for state in state_set:
			model.add_state(state)

		for j in range(n_state - 1):
			model.add_transition(state_set[j], state_set[j], 0.5)
			model.add_transition(state_set[j], state_set[j + 1], 0.5)
		model.add_transition(state_set[n_state - 1], state_set[n_state - 1], 1.0)
		model.add_transition(model.start, state_set[0], 1.0)
		model.bake()
		model.train(observations, algorithm='baum-welch')
		# mm.append(model)

		hh.append(model)
	return hh

def test(hmm):
	f = open('train_result/test_set', 'r')
	setstr = f.read()
	test_set = dill.loads(setstr)

	vv = []
	for i in range(20):
		f = open('train_result/vq_' + str(i), 'r')
		vqs = dill.loads(f.read())
		vv.append(vqs)


	total_all = 0
	right_all = 0
	for (i, word) in enumerate(words):
		if len(test_set[i]) == 0:
			continue	
		total = 0
		right = 0
		ff = open('train_result/test_result_' + str(i), 'w')

		features = []
		for fname in test_set[i]:
			print fname
			waveData, params = read_voice_file(fname)
			waveData = waveData[400:-400]
			framerate = params[2]
			start, end = rd.range_detect(waveData, 0)
			waveData = waveData[start:end]
			features = mfcc.feature_extractor(waveData, framerate)

			maxv = -1e100
			for j in range(20):
				observation = []
				for feature in features:
					dmin, choice = vv[j].quantization(feature)
					observation.append(choice)
				# print observation
				value = hmm[j].log_probability(observation)
				print j, ' ', value
				ff.write(str(j) + ' ' + str(value) + '\n')
				if value > maxv:
					maxv = value
					tag = j
			print words[tag], maxv
			ff.write(words[tag] + str(maxv) + '\n')

			if i == tag:
				right += 1
			total += 1

		total_all += total
		right_all += right
		print word, ' PASSED: ', float(right)/total * 100, '%'
		ff.write(word + ' PASSED: ' + str(float(right)/total * 100) + '%\n')

	print 'all PASSED: ', float(right)/total * 100, '%'

if __name__ == "__main__":
	# get_file_list('data')
	# print "file list finished"
	# pre_proc_train()
	# print "range detect finished"
	# vq_train()
	# print "vq training finished"
	hmm = hmm_train()
	print "hmm training finished"
	test(hmm)


	# vv = []
	# mm = []
	# for (i, word) in enumerate(words):
	# 	if len(train_set[i]) == 0:
	# 		continue
	# 	vqs = vq.VQset();
	# 	features_set = []
	# 	for fname in train_set[i]:
	# 		print fname
	# 		waveData, params = read_voice_file(fname)
	# 		waveData = waveData[400:-400]
	# 		framerate = params[2]
	# 		start, end = rd.range_detect(waveData, 0)
	# 		waveData = waveData[start:end]
	# 		features = mfcc.feature_extractor(waveData, framerate)
	# 		vqs.add_samples(features)
	# 		features_set.append(features)

	# 	vqs.training()
	# 	vv.append(vqs)

	# 	observations = []
	# 	for features in features_set:
	# 		observation = []
	# 		for feature in features:
	# 			dmin, choice = vqs.quantization(feature)
	# 			observation.append(choice)
	# 		observations.append(observation)

	# 	n_state = 10
	# 	n_features = vqs.n_clusters
	# 	model = yahmm.Model()
	# 	state_set = []
	# 	for i in range(n_state):
	# 		state_set.append(yahmm.State(
	# 			yahmm.DiscreteDistribution({i : 1.0/n_features for i in range(n_features)}),
	# 			name = str(i)))
	# 	for state in state_set:
	# 		model.add_state(state)

	# 	for i in range(n_state - 1):
	# 		model.add_transition(state_set[i], state_set[i], 0.5)
	# 		model.add_transition(state_set[i], state_set[i + 1], 0.5)
	# 	model.add_transition(state_set[n_state - 1], state_set[n_state - 1], 1.0)
	# 	model.add_transition(model.start, state_set[0], 1.0)
	# 	model.bake()
	# 	model.train(observations, algorithm='baum-welch')
	# 	mm.append(model)

	# 	picklestring = dill.dumps(model)
	# 	dumpfile = open('train_result/' + word, "w")
	# 	dumpfile.write(picklestring)

	# total_all = 0
	# right_all = 0
	# for (i, word) in enumerate(words):
	# 	if len(test_set[i]) == 0:
	# 		continue	
	# 	total = 0
	# 	right = 0

	# 	features = []
	# 	for fname in test_set[i]:
	# 		print fname
	# 		waveData, params = read_voice_file(fname)
	# 		waveData = waveData[400:-400]
	# 		framerate = params[2]
	# 		start, end = rd.range_detect(waveData, 0)
	# 		waveData = waveData[start:end]
	# 		features = mfcc.feature_extractor(waveData, framerate)

	# 		observation = []
	# 		maxv = -1e100
	# 		for j in range(20):
	# 			for feature in features:
	# 				dmin, choice = vv[j].quantization(feature)
	# 				observation.append(choice)
	# 			value = mm[j].log_probability(observation)
	# 			if value > maxv:
	# 				maxv = value
	# 				tag = j
	# 		print words[tag], maxv

	# 		if i == tag:
	# 			right += 1
	# 		total += 1

	# 	total_all += total
	# 	right_all += right
	# 	print word, ' PASSED: ', float(right)/total * 100, '%'

	# print 'all PASSED: ', float(right)/total * 100, '%'





