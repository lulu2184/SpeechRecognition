# -*- coding: utf-8 -*-
import os


words = [u'数字', u'语音', u'信号', u'分析', u'识别',
		 u'北京', u'背景', u'上海', u'商行', u'复旦',
		'speech', 'voice', 'sound', 'happy', 'lucky',
		'file', 'open', 'close', 'start', 'stop']

def walk_dir(path):
	voice_list = [[] for i in range(20)]
	for fname in os.listdir(path):
		newpath = os.path.join(fname)
		if os.path.isdir(fname):
			result = walk_dir(newpath)
			voice_list = [v1 + v2 for (v1, v2) in zip(voice_list, result)]
		else:
			for (i, keyword) in enumerate(words):
				if fname.find(keyword) >= 0:
					voice_list[i].append(fname)
					break;
	return voice_list

def read_voice_file(fname):
	

if __name__ == "__main__":
	voice_list = walk_dir('dataset')
	train_set = []
	test_set = []
	for ele in voice_list:
		sp = len(ele) / 3 * 2
		train_set.append(ele[:sp])
		test_set.append(ele[sp:])
	for (i, word) in enumerate(words):
		for fname in train_set[i]:
			waveData = read_voice_file(fname)