### Avoid warning ###
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

### Essential ###
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from keras.models import load_model

import math
import gzip

number = '1'
bs=512
N=16
k=20
m=3
l=8

# bs = batch_size
# N = filter number
# k = filter size , motif_size
# m = pooling size
# l = neuron number of fully connected layer

def set_convolution_layer():
    input_shape = (98+k , 256)

    model = models.Sequential()
    model.add(layers.Conv1D(N, k, padding='valid',input_shape = input_shape))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=m))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv1D(N, int(k/2), padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=m))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(l, activation='relu'))
    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))
    
    model.summary()
    return model


from sklearn.preprocessing import LabelEncoder

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = tf.keras.utils.to_categorical(y)
    return y, encoder

def load_label_seq(seq_file):
    encoding = 'utf-8'
    label_list = []
    seq = ''

    #with gzip.open(seq_file, 'r') as fp:
    fp = open(seq_file,'r')
    for line in fp:
        #line = str(line, encoding)
        #print(line, line[0])
        if line[0] == '>':
            #print(line, line[0])
            name = line[1:-1]
            posi_label = name.split(';')[-1]
            label = posi_label.split(':')[-1]
            #print(label)
            label_list.append(int(label))
           
    return np.array(label_list)

def read_seq(seq_file,k):
    encoding = 'utf-8'
    degree = 4
    encoder = buildseqmapper(degree)
    seq_list = []
    seq = ''
 
    #with gzip.open(seq_file, 'r') as fp:
    fp = open(seq_file,'r')
    for line in fp:
        #print(line)
        #line = str(line, encoding)
        if line[0] == '>':
            name = line[1:-1]
            if len(seq):
                seqdata = GetSeqDegree(seq.upper(),degree,k)
                seq_array = embed(seqdata,encoder)
                seq_list.append(seq_array)
            seq = ''
        else:
            seq = seq + line[:-1]
    if len(seq):
        seqdata = GetSeqDegree(seq.upper(), degree,k)
        seq_array = embed(seqdata, encoder)
        seq_list.append(seq_array)
   
    return np.array(seq_list)

def buildseqmapper(degree):
    length = degree
    alphabet = ['A', 'C', 'G', 'T']
    mapper = ['']
    while length > 0:
        mapper_len = len(mapper)
        temp = mapper
        for base in range(len(temp)):
            for letter in alphabet:
                mapper.append(temp[base] + letter)
        # delete the original conents
        while mapper_len > 0:
            mapper.pop(0)
            mapper_len -= 1

        length -= 1

    code = np.eye(len(mapper), dtype=int)
    encoder = {}
    for i in range(len(mapper)):
        encoder[mapper[i]] = list(code[i, :])

    number = int(math.pow(4, degree))
    encoder['N'] = [1.0 / number] * number
    return encoder

def GetSeqDegree(seq, degree, motif_len):
    half_len = int(motif_len/2)
    length = len(seq)
    row = (length + motif_len - degree + 1)
    seqdata = []
    for i in range (half_len):
        multinucleotide = 'N'
        seqdata.append(multinucleotide)

    for i in range(length - degree + 1):
        multinucleotide = seq[i:i + degree]
        seqdata.append(multinucleotide)

    for i in range (row-half_len,row):
        multinucleotide = 'N'
        seqdata.append(multinucleotide)

    return seqdata

def embed(seq, mapper):
    mat = []
    for element in seq:
        if element in mapper:
            mat.append(mapper.get(element))
        elif "N" in element:
            mat.append(mapper.get("N"))
        else:
            print (element,"wrong")

    return np.asarray(mat)


def load_data_file(data_file , k):
    tmp = []
    tmp.append(read_seq(data_file,k))
    
    data = dict()
    data["seq"] = tmp
    data["Y"] = load_label_seq(data_file)

    return data

def train_HOCNN(data_file):
    #data_file="sequence.fa.gz"
    #data_file="FASTA.txt"
    
    data = load_data_file(data_file, k)
    train_Y = data["Y"]
    seq_data = data["seq"][0]

    y = preprocess_labels(train_Y)

    print(len(y[0]))
    print(len(data["seq"][0][0][0]))
    print(len(data["seq"][0][0]))

    my_classifier = set_convolution_layer()
    my_classifier.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
    my_classifier.fit(seq_data, y[0], batch_size=bs, epochs=100)
    print(my_classifier.summary())

    my_classifier.save('seqcnn3_model.pkl')

def test_HOCNN(data_file):
    outfile = 'HOCNNLB/prediction.txt'
    fprfile = 'HOCNNLB/fpr.txt'
    tprfile = 'HOCNNLB/tpr.txt'
    metricsfile = 'HOCNNLB/metrics_file.txt'
    if not os.path.exists('HOCNNLB/'):
        os.makedirs('HOCNNLB/')

    print ('model prediction')

    data = load_data_file(data_file, k)
    true_y = data["Y"]

    testing = data["seq"][0]  # it includes one-hot encoding sequence and structure
    model = load_model('seqcnn3_model.pkl')
    
    predictions = model.predict(testing)
    predictions_label = transfer_label_from_prob(predictions[:, 1])

    fw = open(outfile, 'w')
    myprob = "\n".join(map(str, predictions[:, 1]))
    # fw.write(mylabel + '\n')
    fw.write(myprob)
    fw.close()

    fpr,tpr,thresholds = roc_curve(true_y,predictions[:, 1])
    with open(fprfile, 'w') as f:
        writething = "\n".join(map(str, fpr))
        f.write(writething)
    with open(tprfile, 'w') as f:
        writething = "\n".join(map(str, tpr))
        f.write(writething)

    acc, sensitivity, specificity, MCC = calculate_performance(len(true_y), predictions_label, true_y)
    roc_auc = auc(fpr, tpr)

    out_rel = ['acc', acc, 'sn', sensitivity, 'sp', specificity, 'MCC', MCC, 'auc', roc_auc]
    with open(metricsfile, 'w') as f:
        writething = "\n".join(map(str, out_rel))
        f.write(writething)

    print ("acc,  sensitivity, specificity, MCC,auc : ", acc, sensitivity, specificity, MCC, roc_auc)   

def transfer_label_from_prob(proba):
    label = [0 if val <= 0.5 else 1 for val in proba]
    return label

def calculate_performance(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num
    # precision = float(tp) / (tp + fp)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return acc,sensitivity, specificity, MCC

if __name__ == '__main__':
    # download lncRBPdata.zip from https://github.com/NWPU-903PR/HOCNNLB
    # data_file="./RBPdata1201/01_HITSCLIP_AGO2Karginov2013a_hg19/train/1/sequence.fa.gz"  was renamed
    train_HOCNN(data_file= "Pyfeat_FASTA.txt")
    #train_HOCNN(data_file= "Hocnnlb_train.txt")
    test_HOCNN(data_file = "Hocnnlb_test.txt")