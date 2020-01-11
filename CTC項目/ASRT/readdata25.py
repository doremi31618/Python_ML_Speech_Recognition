#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform as plat

import tensorflow as tf
import os
from tqdm import tqdm
import numpy as np
from general_function.file_wav import *
from general_function.file_dict import *
from random import shuffle
from scipy.io import wavfile
import random
import numpy as np
#import scipy.io.wavfile as wav
from scipy.fftpack import fft

def data_hparams():
    params = tf.contrib.training.HParams(
    # data_type='train',
    datapath='',
    thchs30=True,
    aishell=True,
    prime=True,
    stcmd=True,
    batch_size=1,
    # vocabNum = 0,
    # data_length=10,
    shuffle=True)
    return params

class DataSpeech():

    def __init__(self,args, type, LoadToMem = False, MemWavCount = 10000):
        '''
        初始化
        参数：
            path：数据存放位置根目录
        '''
        system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
        self.datapath = args.datapath; # 数据存放位置根目录
        self.type = type # 数据类型，分为三种：训练集(train)、验证集(dev)、测试集(test)
        self.slash = ''
        if(system_type == 'Windows'):
            self.slash='\\' # 反斜杠
        elif(system_type == 'Linux'):
            self.slash='/' # 正斜杠
        else:
            print('*[Message] Unknown System\n')
            self.slash='/' # 正斜杠
        #'C:\\Users\\user\\Downloads\\DeepSpeechRecognition-master\\dataset\\'
        if(self.slash != self.datapath[-1]): # 在目录路径末尾增加斜杠
            self.datapath = self.datapath + self.slash

        self.dic_wavlist_all = {}
        self.dic_symbollist_all = {}
        self.dic_han_all = {}
        self.list_han_all = []
        self.list_wavnum_all = []  # wav文件标记列表
        self.list_symbolnum_all = []  # symbol标记列表
        self.SymbolNum = 0 # 记录拼音符号数量
        #self.list_symbol = self.GetSymbolList() # 全部汉语拼音符号列表
        self.DataNum = 0 # 记录数据量
        self.list_symbol,self.han_vocab = Get_lm_vocab()
        self.thchs30 = args.thchs30
        self.aishell = args.aishell
        self.prime = args.prime
        self.stcmd = args.stcmd
        self.LoadToMem = LoadToMem
        self.MemWavCount = MemWavCount
        self.shuffle = args.shuffle
        self.batch_size = args.batch_size
        # print(len(self.list_symbol))
        # #print(self.SymbolNum)
        # print(self.GetPny_vocabNum())
        print(self.list_symbol[0],self.list_symbol[1],self.list_symbol[-1])
        pass

    def GetType(self):
        return self.type

    def LoadDataList(self):
        '''
        加载用于计算的数据列表
        参数：
            type：选取的数据集类型
                train 训练集
                dev 开发集
                test 测试集
        '''
        print('get source list...')
        read_files = []
        read_labels = []
        read_han = []
        #if self.type == 'train':
        if self.thchs30 == True:
            filename_wavlist_thchs30 = 'thchs30' + self.slash + self.type + '.wav.lst'
            filename_symbollist_thchs30 = 'thchs30' + self.slash + self.type + '.syllable.txt'
            filename_hanlist_thchs30 = 'thchs30' + self.slash + self.type + '.han.txt'
            read_files.append(filename_wavlist_thchs30)
            read_labels.append(filename_symbollist_thchs30)
            read_han.append(filename_hanlist_thchs30)
        if self.aishell == True:
            filename_wavlist_aishell = 'aishell' + self.slash + 'aishell_' + self.type + '.wav.txt'
            filename_symbollist_aishell = 'aishell' + self.slash + 'aishell_' + self.type + '.syllable.txt'
            filename_hanlist_aishell = 'aishell' + self.slash + 'aishell_' + self.type + '.han.txt'
            read_files.append(filename_wavlist_aishell)
            read_labels.append(filename_symbollist_aishell)
            read_han.append(filename_hanlist_aishell)
        if self.prime == True:
            filename_wavlist_prime = 'prime' + self.slash + self.type + '.wav.txt'
            filename_symbollist_prime = 'prime' + self.slash + self.type + '.syllable.txt'
            filename_hanlist_prime = 'prime' + self.slash + self.type + '.han.txt'
            read_files.append(filename_wavlist_prime)
            read_labels.append(filename_symbollist_prime)
            read_han.append(filename_hanlist_prime)
        if self.stcmd == True:
            filename_wavlist_stcmds = 'st-cmds' + self.slash + self.type + '.wav.txt'
            filename_symbollist_stcmds = 'st-cmds' + self.slash + self.type + '.syllable.txt'
            filename_hanlist_stcmds = 'st-cmds' + self.slash + self.type + '.han.txt'
            read_files.append(filename_wavlist_stcmds)
            read_labels.append(filename_symbollist_stcmds)
            read_han.append(filename_hanlist_stcmds)

        self.dic_wavlist_all,self.list_wavnum_all=get_wav_list(self.datapath,read_files)
        self.dic_symbollist_all,self.list_symbolnum_all = get_wav_symbol(self.datapath,read_labels)
        self.dic_han_all,self.list_han_all = get_wav_han(self.datapath,read_han)
        self.DataNum = self.GetDataNum()

    def GetDataNum(self):
        '''
        获取数据的数量
        当wav数量和symbol数量一致的时候返回正确的值，否则返回-1，代表出错。
        '''
        if (len(self.dic_wavlist_all) == len(self.dic_symbollist_all)):
            DataNum = len(self.list_wavnum_all)
        else:
            DataNum = -1
        return DataNum

    def GetData(self,n_start,n_amount=1):
        '''
        读取数据，返回神经网络输入值和输出值矩阵(可直接用于神经网络训练的那种)
        参数：
            n_start：从编号为n_start数据开始选取数据
            n_amount：选取的数据数量，默认为1，即一次一个wav文件
        返回：
            三个包含wav特征矩阵的神经网络输入值，和一个标定的类别矩阵神经网络输出值
        '''
        n = 0
        filename = self.dic_wavlist_all[self.list_wavnum_all[n_start+n]]
        list_symbol = self.dic_symbollist_all[self.list_symbolnum_all[n_start+n]]

        # print("1",filename)
        # print("1",list_symbol)
        if('Windows' == plat.system()):
            filename = filename.replace('/','\\') # windows系统下需要执行这一行，对文件路径做特别处理
        
        #wavsignal,fs=read_wav_data(self.datapath + filename)
        try:
            fs, wav1 = wavfile.read(self.datapath + filename)
        except Exception:
            print( 'the file is open error : ',self.datapath + filename)
            n = n + self.batch_size
            filename = self.dic_wavlist_all[self.list_wavnum_all[n_start+n]]
            list_symbol = self.dic_symbollist_all[self.list_symbolnum_all[n_start+n]]
            fs, wav1 = wavfile.read(self.datapath + filename)
        # print('2',filename)
        # print('2',list_symbol)
        
        wavsignal = wav1.astype(np.float32) / np.iinfo(np.int16).max
        data_input = compute_fbank(wavsignal,fs)
        
        feat_out=[]
        #print("数据编号",n_start,filename)
        for i in list_symbol:
            if(''!=i):
                # 拼音找词典位置 拼音表
                n=self.SymbolToNum(i)
                #v=self.NumToVector(n)
                #feat_out.append(v)
                feat_out.append(n)
        #print('feat_out:',feat_out)
        
        # 获取输入特征
        
        #data_input = np.array(data_input)
        #-----------------------------------------------------------------------------
        pad_fbank = np.zeros((data_input.shape[0] // 8 * 8 + 8, data_input.shape[1]))
        pad_fbank[:data_input.shape[0], :] = data_input
    
        #data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)

        data_label = np.array(feat_out)
        return pad_fbank, data_label


    def data_genetator(self, batch_size=32, audio_length = 1600):
        '''
        数据生成器函数，用于Keras的generator_fit训练
        batch_size: 一次产生的数据量
        需要再修改。。。
        '''

        #labels = np.array(labels, dtype = np.float)
        labels = np.zeros((batch_size,1), dtype = np.float)
        #print(input_length,len(input_length))
        
        while True:
            X = np.zeros((batch_size, audio_length, 200, 1), dtype = np.float)
            #y = np.zeros((batch_size, 64, self.SymbolNum), dtype=np.int16)
            y = np.zeros((batch_size, 64), dtype=np.int16)
            
            #generator = ImageCaptcha(width=width, height=height)
            input_length = []
            label_length = []
            wav_data_lst = []
            label_data_lst = []
            ran_num = random.randint(0,self.DataNum - 1) # 获取一个随机数
            num_bias = 0
        
            for i in range(batch_size):
                #ran_num = random.randint(0,self.DataNum - 1) # 获取一个随机数
                data_input, data_labels = self.GetData((ran_num + i + num_bias) % self.DataNum)  # 通过随机数取一个数据
                #data_input, data_labels = self.GetData(ran_num + i)  # 从随机数开始连续向后取一定数量数据
                #num_bias = 0
                label_ctc_len = self.ctc_len(data_labels)

                while(data_input.shape[0] // 8 < label_ctc_len):
                    print('*[Error]','wave data lenghth of num',(ran_num + i+ num_bias) % self.DataNum, 'is too short.','\n' )
                    num_bias += 1
                    data_input, data_labels = self.GetData((ran_num + i + num_bias) % self.DataNum)  # 从随机数开始连续向后取一定数量数据
                # 关于下面这一行取整除以8 并加8的余数，在实际中如果遇到报错，可尝试只在有余数时+1，没有余数时+0，或者干脆都不加，只留整除
                #print('data id is',(ran_num + i+ num_bias))
                #input_length.append(data_input.shape[0] // 8 + data_input.shape[0] % 8)

                # label_ctc_len = self.ctc_len(data_labels)
                # if data_input.shape[0] // 8 >= label_ctc_len:
                wav_data_lst.append(data_input)
                label_data_lst.append(data_labels)
            pad_wav_data, input_length = self.wav_padding(wav_data_lst)
            pad_label_data, label_length = self.label_padding(label_data_lst)
            yield [pad_wav_data, pad_label_data, input_length, label_length ], labels

                # input_length.append(data_input.shape[0] // 8 )
                # X[i,0:len(data_input)] = data_input
                # y[i,0:len(data_labels)] = data_labels
                # label_length.append([len(data_labels)])

                # label_length = np.matrix(label_length)
                # input_length = np.array([input_length]).T
                # yield [X, y, input_length, label_length ], labels
            #batch_size =8 , labels = <class 'tuple'>: (8, 1) [[0.][0.][0.][0.][0.][0.][0.][0.]]
        pass

    def wav_padding(self, wav_data_lst):
        wav_lens = [len(data) for data in wav_data_lst]
        wav_max_len = max(wav_lens)
        wav_lens = np.array([leng // 8 for leng in wav_lens])
        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
        return new_wav_data_lst, wav_lens

    def label_padding(self, label_data_lst):
        label_lens = np.array([len(label) for label in label_data_lst])
        max_label_len = max(label_lens)
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
        for i in range(len(label_data_lst)):
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
        return new_label_data_lst, label_lens

    def ctc_len(self, label):
        add_len = 0
        label_len = len(label)
        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1
        return label_len + add_len

    def get_lm_batch(self,batch_size):
        batch_num = self.GetDataNum() // batch_size
        for k in range(batch_num):
            begin = k * batch_size
            end = begin + batch_size
            input_batch = self.FindDictList(begin, end, self.dic_symbollist_all, self.list_symbolnum_all, )
            # print(input_batch)
            label_batch = self.FindDictList(begin, end, self.dic_han_all, self.list_han_all)
            # print(label_batch)
            max_len = max([len(line) for line in input_batch])
            input_batch = np.array(
                [self.pny2id(line) + [0] * (max_len - len(line)) for line in input_batch])
            label_batch = np.array(
                [self.han2id(line) + [0] * (max_len - len(line)) for line in label_batch])
            #
            yield input_batch, label_batch

    def FindDictList(self, begin, end, dic={}, lis=[]):
        return [dic[lis[i]] for i in range(begin, end)]

    def pny2id(self, line):
        label_id = []
        for symbol in line:
            if (symbol != ''):
                if not symbol.endswith(('1', '2', '3', '4', '5')):
                    symbol = symbol + '5'
                if symbol not in self.list_symbol:
                    label_id.append(1)
                else:
                    label_id.append(self.list_symbol.index(symbol))
        return label_id

    def han2id(self, line):
        label_id = []
        if (line != ''):
            for i in range(len(line)):
                if line[i] not in self.han_vocab:
                    label_id.append(1)
                else:
                    label_id.append(self.han_vocab.index(line[i]))
        return label_id

    # def GetSymbolList(self) :
    #     txt_obj=open('dict.txt','r',encoding='UTF-8') # 打开文件并读入
    #     txt_text=txt_obj.read()
    #     txt_lines=txt_text.split('\n') # 文本分割
    #     list_symbol=['<PAD>'] # 初始化符号列表
    #     list_symbol.append('UNK')
    #     for i in txt_lines:
    #         if(i!=''):
    #             txt_l=i.split('\t')
    #             list_symbol.append(txt_l[0])
    #     txt_obj.close()
    #     list_symbol.append('_')
    #     self.SymbolNum = self.GetPny_vocabNum()
    #     print('list_symbol[-1]',list_symbol[-1])
    #     return list_symbol

    def GetPny_vocabNum(self):
        return len(self.list_symbol)

    def GetHan_vocabNum(self):
        return len(self.han_vocab)

    def SymbolToNum(self,symbol):
        '''
        符号转为数字
        '''
        if(symbol != ''):
            if not symbol.endswith(('1','2','3','4','5')):
                symbol = symbol+'5'
            if symbol not in self.list_symbol:
                symbol = 'UNK'
            return self.list_symbol.index(symbol)
        return 1
    
    def NumToVector(self,num):
        '''
        数字转为对应的向量
        '''
        v_tmp=[]
        for i in range(0,len(self.list_symbol)):
            if(i==num):
                v_tmp.append(1)
            else:
                v_tmp.append(0)
        v=np.array(v_tmp)
        return v

# def compute_fbank(file):
#     x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
#     w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
#     fs, wavsignal = wav.read(file)
#     # wav波形 加时间窗以及时移10ms
#     time_window = 25  # 单位ms
#     wav_arr = np.array(wavsignal)
#     range0_end = int(len(wavsignal) / fs * 1000 - time_window) // 10 + 1 # 计算循环终止的位置，也就是最终生成的窗数
#     data_input = np.zeros((range0_end, 200), dtype=np.float)  # 用于存放最终的频率特征数据
#     data_line = np.zeros((1, 400), dtype=np.float)
#     for i in range(0, range0_end):
#         p_start = i * 160
#         p_end = p_start + 400
#         data_line = wav_arr[p_start:p_end]
#         data_line = data_line * w  # 加窗
#         data_line = np.abs(fft(data_line))
#         data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的
#     data_input = np.log(data_input + 1)
#     # data_input = data_input[::]
#     return data_input

if(__name__=='__main__'):
    #path='E:\\语音数据集'
    #l=DataSpeech(path)
    #l.LoadDataList('train')
    #print(l.GetDataNum())
    #print(l.GetData(0))
    #aa=l.data_genetator()
    #for i in aa:
        #a,b=i
    #print(a,b)
    pass
    