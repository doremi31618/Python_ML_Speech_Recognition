#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reference : 
1. Building a Dead Simple Speech Recognition Engine using ConvNet in Keras
    https://github.com/manashmandal/DeadSimpleSpeechRecognizer
    
2. Day 25：自動語音識別(Automatic Speech Recognition) -- 觀念與實踐
    https://ithelp.ithome.com.tw/articles/10195763
"""
import sys,librosa,os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, load_model , Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical,np_utils
from keras.optimizers import SGD, Adam

DATA_PATH = "./deep-learning-ntut-2019-autumn-speech-recognition/train/train/audio/"
Test_Data_path =os.path.dirname(__file__) +  "/deep-learning-ntut-2019-autumn-speech-recognition/test/test/"
epoche = 250
batch_size = 100
file_name = str(epoche)+'_'+str(batch_size)

#main()
#get_lables()
#train_model()
#predict_data()

#process code
def main():
    get_lables()
    train_model()
    predict_data()
    
def get_lables():
    labels, _, _, = get_labels(DATA_PATH)
    if labels.count('.DS_Store') > 0:
        labels.remove('.DS_Store')
    np.savetxt('name.txt', labels , delimiter=' ', fmt="%s")
    save_data_to_array()

def train_model():
    #get train test data 
    X_train, X_test, y_train, y_test = get_train_test(DATA_PATH,0.99,42)
    X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)
    X_train = X_train.reshape(X_train.shape[0],20,11,1)
    
    #reshape train test
    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)
    
    #cnn model
    model = Sequential()
    model.add(Conv2D(30, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(len(labels), activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epoche, verbose=1, validation_data=(X_test, y_test_hot))

    #save model
    model_path = os.path.dirname(__file__) + '/h5'
    if not os.path.isdir(model_path) :
        os.mkdir(model_path)
    model.save(model_path+'/'+file_name+'.h5')

def predict_data():
    #load moel 
    model = load_model('h5/'+file_name+'.h5')
    prepare_dataset(Test_Data_path)
    
    #data set 
    test_audio = load_prepeare_dataset()
    test_audio = test_audio.reshape(test_audio.shape[0], 20, 11, 1)
    
    #predict 
    predict = model.predict_classes(test_audio, verbose=1)
    label_str = transform(np.loadtxt('name.txt', dtype='str'),predict, test_audio.shape[0])
    
    name_list = np.loadtxt('name.txt', dtype='str')
    for i in range(predict.shape[0]):
        index = predict[i]
        temp = name_list[index]
        if not temp in target_name:
            temp = 'unknown'
        label_str.append(temp)
    
    df = pd.DataFrame({"word": label_str})
    df.index = np.arange(1, len(df) + 1)
    df.index.names = ['id']
    df.to_csv('test.csv')
    
# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    
    labels = os.listdir(path)
    if labels.count('.DS_Store') > 0:
        labels.remove('.DS_Store')
    label_indices = np.arange(0, len(labels))
    print(label_indices,labels)
    return labels, label_indices, to_categorical(label_indices)


# Handy function to convert wav2mfcc
def wav2mfcc(file_path, max_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc

#save wav to mfcc vector
def save_data_to_array(path=DATA_PATH, max_len=11):
    labels, _, _ = get_labels(path)
    
    #[number of Audio folders] 0~30 
    for label in labels:
        if label == '.DS_Store':
            continue
        else:
        # Init mfcc vectors
            mfcc_vectors = []
            wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
            for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
                mfcc = wav2mfcc(wavfile, max_len=max_len)
                mfcc_vectors.append(mfcc)
                
            np.save("./npy/" + label + '.npy', mfcc_vectors)


def get_train_test(path=DATA_PATH,split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(path)

    # Getting first arrays
    X = np.load('./npy/' + labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        if label == '.DS_Store':
            continue
        else:
            x = np.load('./npy/' + label + '.npy')
            X = np.vstack((X, x))
            y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)
    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)


def prepare_dataset(path=DATA_PATH):
    
    mfcc_vectors = []
    max_len = 11
    wavfiles = [path + '/' + wavfile for wavfile in os.listdir(path)]
    for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format("Test Set")):
                mfcc = wav2mfcc(wavfile, max_len=max_len)
                mfcc_vectors.append(mfcc)
    
    np.save('./npy/Test.npy', mfcc_vectors)
    
          

def load_prepeare_dataset():
     X = np.load('./npy/Test.npy') 
     y = np.zeros(X.shape[0])
     assert X.shape[0] == len(y)
     return X

def load_dataset(path=DATA_PATH):
    data = prepare_dataset(path)
    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))

    return dataset[:100]

def transform(listdir, label, lenSIZE):
    label_str = []
    for i in range(lenSIZE):
        temp = listdir[label[i]]
        label_str.append(temp)
    return label_str
#
'''
print(prepare_dataset(Test_Data_path))
#labels, _, _, = get_labels(DATA_PATH)
#if labels.count('.DS_Store') > 0:
#    labels.remove('.DS_Store')
#np.savetxt('name.txt', labels , delimiter=' ', fmt="%s")
#save_data_to_array()


#train model 
#train_model()
##################################################################################
X_train, X_test, y_train, y_test = get_train_test(DATA_PATH,0.99,42)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)
X_train = X_train.reshape(X_train.shape[0],20,11,1)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
model = Sequential()
model.add(Conv2D(30, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(len(labels), activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epoche, verbose=1, validation_data=(X_test, y_test_hot))

model_path = os.path.dirname(__file__) + '/h5'
if not os.path.isdir(model_path) :
    os.mkdir(model_path)
model.save(model_path+'/'+file_name+'.h5')



# predict model 
###################################################################################
model = load_model('h5/'+file_name+'.h5')

test_audio = []

target_name = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']
prepare_dataset(Test_Data_path)
test_audio = load_prepeare_dataset()
test_audio = test_audio.reshape(test_audio.shape[0], 20, 11, 1)
predict = model.predict_classes(test_audio, verbose=1)
label_str = transform(np.loadtxt('name.txt', dtype='str'),predict, test_audio.shape[0])
name_list = np.loadtxt('name.txt', dtype='str')
label_str = []
for i in range(predict.shape[0]):
    index = predict[i]
    temp = name_list[index]
    if not temp in target_name:
        temp = 'unknown'
    label_str.append(temp)

df = pd.DataFrame({"word": label_str})
df.index = np.arange(1, len(df) + 1)
df.index.names = ['id']
df.to_csv('test.csv')

'''