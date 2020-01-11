# from model_language.transformer import Lm, lm_hparams
from transformer import Lm, lm_hparams
from readdata25 import data_hparams,DataSpeech
#from model_language.cbhg import Lm, lm_hparams
import tensorflow as tf
import os
import platform as plat
from tqdm import tqdm

system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
if(system_type == 'Windows'):
	datapath = 'C:\\Users\\user\\Downloads\\DeepSpeechRecognition-master\\dataset'
elif(system_type == 'Linux'):
	datapath = 'dataset'
else:
	print('*[Message] Unknown System\n')
	datapath = 'dataset'
#---------------------------------
data_args = data_hparams()
data_args.datapath = datapath
data_args.thchs30 = True
data_args.aishell = False
data_args.prime = False
data_args.stcmd = False

train_data = DataSpeech(data_args,'train')
train_data.LoadDataList()
lm_args = lm_hparams()
lm_args.num_heads = 8
lm_args.num_blocks = 6
lm_args.input_vocab_size = train_data.GetPny_vocabNum()
lm_args.label_vocab_size = train_data.GetHan_vocabNum()
lm_args.max_length = 100
lm_args.hidden_units = 512
lm_args.dropout_rate = 0.2
lm_args.lr = 0.0003
lm_args.is_training = True
lm = Lm(lm_args)

epochs = 10
batch_size = 16
batch_num = train_data.GetDataNum() // batch_size
with lm.graph.as_default():
    saver =tf.train.Saver()
with tf.Session(graph=lm.graph) as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    add_num = 0
    if os.path.exists('logs_lm/checkpoint'):
        print('loading language model...')
        latest = tf.train.latest_checkpoint('logs_lm')
        add_num = int(latest.split('_')[-1])
        saver.restore(sess, latest)
    writer = tf.summary.FileWriter('logs_lm/tensorboard', tf.get_default_graph())
    for k in range(epochs):
        total_loss = 0
        batch = train_data.get_lm_batch(batch_size)
        for i in range(batch_num):
            input_batch, label_batch = next(batch)
            feed = {lm.x: input_batch, lm.y: label_batch}
            cost,_ = sess.run([lm.mean_loss,lm.train_op], feed_dict=feed)
            total_loss += cost
            if (k * batch_num + i) % 10 == 0:
                rs=sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, k * batch_num + i)
        print('epochs', k+1, ': average loss = ', total_loss/batch_num)
    saver.save(sess, 'logs_lm/model_%d' % (epochs + add_num))
    writer.close()