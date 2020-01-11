#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

用于训练语音识别系统语音模型的程序

"""
import platform as plat
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from cnn_model import ModelSpeech
from readdata25 import data_hparams
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#进行配置，使用95%的GPU
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# # config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
# set_session(tf.Session(config=config))

datapath = ''
modelpath = 'model_speech'

if(not os.path.exists(modelpath)): # 判断保存模型的目录是否存在
	os.makedirs(modelpath) # 如果不存在，就新建一个，避免之后保存模型的时候炸掉

system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
if(system_type == 'Windows'):
	datapath = 'C:\\Users\\user\\Downloads\\DeepSpeechRecognition-master\\dataset'
	modelpath = modelpath + '\\'
elif(system_type == 'Linux'):
	datapath = 'dataset'
	modelpath = modelpath + '/'
else:
	print('*[Message] Unknown System\n')
	datapath = 'dataset'
	modelpath = modelpath + '/'

#---------------------------------
data_args = data_hparams()
# data_args.data_type = 'train'
data_args.datapath = datapath
data_args.thchs30 = True
data_args.aishell = True
data_args.prime = False
data_args.stcmd = False
data_args.shuffle = False
data_args.batch_size = 16
# print(data_args.vocabNum)
# print(ms.list_symbol[0],ms.list_symbol[1],ms.list_symbol[-1])
#---------------------------------
ms = ModelSpeech(data_args)
ms.TrainModel(datapath, epoch = 50, batch_size = 16, save_step = 1000)




