#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用于测试整个一套语音识别系统的程序
语音模型 + 语言模型
"""
import platform as plat

from cnn_model import ModelSpeech
from LanguageModel2 import ModelLanguage
from keras import backend as K
from readdata25 import data_hparams

datapath = ''
modelpath = 'model_speech'

system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
if(system_type == 'Windows'):
	datapath = 'D:\\语音数据集'
	modelpath = modelpath + '\\'
elif(system_type == 'Linux'):
	datapath = 'dataset'
	modelpath = modelpath + '/'
else:
	print('*[Message] Unknown System\n')
	datapath = 'dataset'
	modelpath = modelpath + '/'


data_args = data_hparams()
# data_args.data_type = 'train'
data_args.datapath = datapath
# data_args.thchs30 = True
# data_args.aishell = False
# data_args.prime = False
# data_args.stcmd = False
# data_args.shuffle = False
# data_args.batch_size = 16

ms = ModelSpeech(data_args)
#修改訓練好的聲學模型的名字
#ms.LoadModel(modelpath + 'm22_2\\0\\speech_model22_e_0_step_257000.model')
ms.LoadModel(modelpath + '/m251/speech_model251_e_59_step_17085.model')

#ms.TestModel(datapath, str_dataset='test', data_count = 64, out_report = True)
#r = ms.RecognizeSpeech_FromFile('D:\\语音数据集\\ST-CMDS-20170001_1-OS\\20170001P00241I0052.wav')
r = ms.RecognizeSpeech_FromFile('dataset/ST-CMDS-20170001_1-OS/20170001P00241I0052.wav')
a = ms.RecognizeSpeech_FromFile('dataset/ST-CMDS-20170001_1-OS/20170001P00241I0053.wav')
b = ms.RecognizeSpeech_FromFile('dataset/ST-CMDS-20170001_1-OS/20170001P00020I0087.wav')
c = ms.RecognizeSpeech_FromFile('dataset/data_thchs30/data/A11_167.wav')
#r = ms.RecognizeSpeech_FromFile('D:\\语音数据集\\data_thchs30\\data\\D4_750.wav')

K.clear_session()

print('*[提示] 语音识别结果：\n',r)
print('*[提示] 语音识别结果：\n',a)
print('*[提示] 语音识别结果：\n',b)
print('*[提示] 语音识别结果：\n',c)

ml = ModelLanguage('model_language')
ml.LoadModel()

#str_pinyin = ['zhe4','zhen1','shi4','ji2', 'hao3','de5']
#str_pinyin = ['jin1', 'tian1', 'shi4', 'xing1', 'qi1', 'san1']
#str_pinyin = ['ni3', 'hao3','a1']
str_pinyin = r
#str_pinyin =  ['su1', 'bei3', 'jun1', 'de5', 'yi4','xie1', 'ai4', 'guo2', 'jiang4', 'shi4', 'ma3', 'zhan4', 'shan1', 'ming2', 'yi1', 'dong4', 'ta1', 'ju4', 'su1', 'bi3', 'ai4', 'dan4', 'tian2','mei2', 'bai3', 'ye3', 'fei1', 'qi3', 'kan4', 'zhan4']
r = ml.SpeechToText(str_pinyin)
a = ml.SpeechToText(a)
b = ml.SpeechToText(b)
c = ml.SpeechToText(c)
print('语音转文字结果：\n',r)
print('语音转文字结果：\n',a)
print('语音转文字结果：\n',b)
print('语音转文字结果：\n',c)














