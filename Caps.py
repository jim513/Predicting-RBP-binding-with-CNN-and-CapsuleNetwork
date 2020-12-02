### Avoid warning ###
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# -*- coding: utf-8 -*-
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
from tensorflow.keras.models import load_model,Sequential,Model
from tensorflow.keras.backend import *
from tensorflow.keras.layers import *
import time
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping,  ModelCheckpoint,ReduceLROnPlateau,Callback
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
import os
from Bio import SeqIO,motifs
from Bio.Seq import Seq
global keys
import random
from sklearn.preprocessing import LabelEncoder
from keras_utils import Capsule
from keras import utils as np_utils

import datetime

np.random.seed(0)
from tensorflow.random import set_seed
set_seed(1)
np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt


batch_size = 32
num_epochs = 30
max_len1=97
max_features = 100
DNAelements = 'ACGT'
RNAelements = 'ACGU'

filename = ['00_PAI244','01_HITSCLIP_AGO2Karginov2013a_hg19', '02_HITSCLIP_AGO2Karginov2013c_hg19',
'03_HITSCLIP_AGO2Karginov2013b_hg19','04_HITSCLIP_AGO2Karginov2013d_hg19','05_HITSCLIP_AGO2Karginov2013f_hg19',
'06_HITSCLIP_AGO2Karginov2013e_hg19','07_PARCLIP_AGO2Gottwein2011a_hg19','08_PARCLIP_AGO2Gottwein2011b_hg19','09_PARCLIP_AGO2Skalsky2012a_hg19','10_PARCLIP_AGO2Skalsky2012b_hg19',
'11_PARCLIP_AGO2Skalsky2012e_hg19', '12_HITSCLIP_DGCR8Macias2012d_hg19','13_HITSCLIP_DGCR8Macias2012c_hg19','14_HITSCLIP_DGCR8Macias2012a_hg19','15_HITSCLIP_DGCR8Macias2012b_hg19',
'16_HITSCLIP_EIF4A3Sauliere2012a_hg19','17_HITSCLIP_EIF4A3Sauliere2012b_hg19','18_HITSCLIP_EWSR1Paronetto2014_hg19','19_PARCLIP_FMR1_Ascano2012b_hg19','20_PARCLIP_FMR1_Ascano2012a_hg19',
'21_PARCLIP_FMR1_Ascano2012c_hg19','22_HITSCLIP_FUSNakaya2013c_hg19','23_HITSCLIP_FUSNakaya2013d_hg19','24_HITSCLIP_FUSNakaya2013e_hg19','25_PARCLIP_HuR_mukherjee2011_hg19',
'26_PARCLIP_IGF2BP123_Hafner2010d_hg19','27_PARCLIP_RBM10Wang2013a_hg19','28_PARCLIP_RBM10Wang2013b_hg19','29_iCLIP_TDP-43_tollervey2011_hg19','30_iCLIP_TIAL1_wang2010b_hg19',
'31_PARCLIP_ZC3H7B_Baltz2012e_hg19']

number = '1'

def pseudoKNC(x, k):
        ### k-mer ###
        ### A, AA, AAA
    T=[]
    for i in range(1, k + 1, 1):
        v = list(itertools.product(DNAelements, repeat=i))
        # seqLength = len(x) - i + 1
        for i in v:
            # print(x.count(''.join(i)), end=',')
            T.append(x.count(''.join(i)))
    return T                

def gcContent(x, seqType):
    T=[]
    if seqType == 'DNA' or seqType == 'RNA':

        if seqType == 'DNA':
            TU = x.count('T')
        else:
            if seqType == 'RNA':
                TU = x.count('U')
            else:
                None

        A = x.count('A')
        C = x.count('C')
        G = x.count('G')

        T.append( (G + C) / (A + C + G + TU)  * 10.0 )
    return T   
def zCurve(x, seqType):
        ### Z-Curve ### total = 3
        T=[]
        if seqType == 'DNA' or seqType == 'RNA':

            if seqType == 'DNA':
                TU = x.count('T')
            else:
                if seqType == 'RNA':
                    TU = x.count('U')
                else:
                    None

            A = x.count('A'); C = x.count('C'); G = x.count('G');

            x_ = (A + G) - (C + TU)
            y_ = (A + C) - (G + TU)
            z_ = (A + TU) - (C + G)
            # print(x_, end=','); print(y_, end=','); print(z_, end=',')
            T.append(x_); T.append(y_); T.append(z_) 
        return T   
                  
def calculate_auc(net_one,x1,model_input1, x_train, y_train , model_name = None):
    #pdb.set_trace()
#    auc = roc_auc_score(y_test, predict)
#    print ("Test AUC: ", auc)
    return model
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
#    precision = float(tp) / (tp + fp)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    PPV = float(tp) / (tp + fp)
    NPV = float(tn) / (tn + fn)
    return acc,sensitivity, specificity, MCC, PPV, NPV

def transfer_label_from_prob(proba):
    label = [0 if val <= 0.5 else 1 for val in proba]
    return label


def load_data_file(inputfile,seq=True):
    
    path = os.path.dirname(inputfile)
    data = dict()
    if seq:       
        data["seq"] = read_seq_new(inputfile)

    data["Y"] = load_label_seq(inputfile)
    return data

def load_data(data_file):
    seq_data=load_data_file(data_file,5)
    sentences = seq_data["seq"]
    seq_label = seq_data["Y"]

    sentences=np.array(sentences)
#    totalsentences=np.array(totalsentences)
    seq_label=np.array(seq_label)
    y, encoder1 = preprocess_labels(seq_label)
    return sentences,y ,encoder1

def padding_sequence(seq, max_len = 101, repkey = 'N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq
def load_test_data(data_file):
   
    seq_data=load_data_file(data_file,5)
    seq_label = seq_data["Y"]
    sentences = seq_data["seq"]  
    sentences=np.array(sentences)
    seq_label=np.array(seq_label)
    return sentences,seq_label 

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)  
    y = encoder.transform(labels).astype(np.int32) 
    if categorical:
        y = np_utils.to_categorical(y) 
    return y, encoder 

def split_training_validation(classes, validation_size = 0.2, shuffle = False):
    """split sampels based on balnace classes"""
    num_samples=len(classes)
    classes=np.array(classes)
    classes_unique=np.unique(classes)
    num_classes=len(classes_unique)
    indices=np.arange(num_samples)
    training_indice = []
    training_label = []
    validation_indice = []
    validation_label = []
    for cl in classes_unique:
        indices_cl=indices[classes==cl]
        num_samples_cl=len(indices_cl)
        if shuffle:
            random.shuffle(indices_cl) # 
        num_samples_each_split=int(num_samples_cl*validation_size)
        res=num_samples_cl - num_samples_each_split
        
        training_indice = training_indice + [val for val in indices_cl[num_samples_each_split:]]
        training_label = training_label + [cl] * res
        
        validation_indice = validation_indice + [val for val in indices_cl[:num_samples_each_split]]
        validation_label = validation_label + [cl]*num_samples_each_split

    training_index = np.arange(len(training_label))
    random.shuffle(training_index)
    training_indice = np.array(training_indice)[training_index]
    training_label = np.array(training_label)[training_index]
    
    validation_index = np.arange(len(validation_label))
    random.shuffle(validation_index)
    validation_indice = np.array(validation_indice)[validation_index]
    validation_label = np.array(validation_label)[validation_index]    
    
            
    return training_indice, training_label, validation_indice, validation_label  
def get_6_trids():  
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**6
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        n=n//base
        ch2=chars[n%base]
        n=n//base
        ch3=chars[n%base]
        n=n//base
        ch4=chars[n%base]
        n=n//base
        ch5=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3 + ch4 + ch5)
    return  nucle_com   

def get_4_nucleotide_dict_composition(tris, seq, ordict):
    seq_len = len(seq)
    tri_feature = []
    k = len(tris[0])
    for x in range(len(seq) + 1- k):
        kmer = seq[x:x+k]
        if kmer in tris:
            ind = tris.index(kmer)
            tri_feature.append(ordict[str(ind)])
        else:
            tri_feature.append(-1)
    return np.asarray(tri_feature)

def embed(seq, mapper):
    mat = []
    for element in seq:
        if element in mapper:
            mat.append(mapper.get(element))
        else:
            print ("wrong")
    return np.asarray(mat)

def GetSeqDegree(seq, degree,motif_len):
   
    length = len(seq)
    row = (length + motif_len - degree + 1)
    seqdata = []
    for i in range(length - degree + 1):
        multinucleotide = seq[i:i + degree]
        seqdata.append(multinucleotide)
    return seqdata
    
def buildseqmapper(degree,kemerseq):
    length = degree
    alphabet = ['A', 'C', 'G', 'T']
    #alphabet = ['A', 'C', 'G', 'U']
    mapper = [''] 
    while length > 0:
        mapper_len = len(mapper)
        temp = mapper
        for base in range(len(temp)):
            for letter in alphabet:
                mapper.append(temp[base] + letter)
        while mapper_len > 0:
            mapper.pop(0)
            mapper_len -= 1

        length -= 1
    code = np.eye(len(mapper), dtype=int) 
    encoder = {}
    for i in range(len(mapper)):
        code[i, :][i]=1+kemerseq.count(mapper[i])
        encoder[mapper[i]] = list(code[i, :])
    return encoder

def kmers(seq, k):
        v = []
        for i in range(len(seq) - k + 1):
            v.append(seq[i:i + k])
        return v

def read_seq_new(seq_file,k):
    degree = 5
    seq_list = []
    seq_totallist = []
    seq = ''
    for line in  SeqIO.parse(seq_file,"fasta"):
       seq=str(line.seq)   
       kemerseq=kmers(seq,5)
       encoder = buildseqmapper(degree,kemerseq)
       seqdata = GetSeqDegree(seq.upper(),degree,k)
       seq_array = embed(seqdata,encoder)
       seq_list.append(seq_array)  
    return np.array(seq_list)

def load_label_seq(seq_file):
    label_list = []
    seq = ''
    for line in  SeqIO.parse(seq_file,"fasta"):
        ids=line.description
        posi_label = ids.split(';')[-1]
        label = posi_label.split(':')[-1]
        label_list.append(int(label))
    
    return np.array(label_list)

def load_data_file(inputfile,k,seq=True):
    """
        Load data matrices from the specified folder.
    """
    path = os.path.dirname(inputfile)
    data = dict()
    if seq:       
        data["seq"]= read_seq_new(inputfile,k)

    data["Y"] = load_label_seq(inputfile)
    return data

def attention(inputs,Input_Dim,name):
    attention_probs = Dense(Input_Dim, activation='softmax')(inputs)
    output_attention_mul = merge([inputs,attention_probs],output_shape=1024, name="attention_mul",mode="concat")
    return output_attention_mul


def get_cnn_network_one(maxlen):
    print ('configure cnn network')

    model = Sequential()
    input_shape=(maxlen,1024)
    #input_shape=(maxlen,256)
    model_input = Input(shape=input_shape)
    #x=Conv1D(nb_filter=64,  
    #                    filter_length=14,
    #                    border_mode="valid",
    #                    subsample_length=1)(model_input)
    x=Conv1D(filters=64,kernel_size=14 ,padding ="valid")(model_input)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    #x=MaxPooling1D(pool_length=3)(x)
    x=MaxPooling1D(pool_size=3)(x)
    x = Dropout(0.5)(x)
    x = Capsule( num_capsule=14, dim_capsule=16, routings=3, share_weights=True)(x)   
    x = Dropout(0.25)(x)
    x= Flatten()(x)
    return model,x,model_input

def read_rna_dict():
    odr_dict = {}
    with open('rna_vocab_dict', 'r') as fp:
        for line in fp:
            values = line.rstrip().split(',') 
            for ind, val in enumerate(values): 
                val = val.strip() 
                odr_dict[val] = ind
    return odr_dict     

def run_network(net_one,x1,model_input1, x_train, y_train ):
    #MODEL_PATH='./'
    MODEL_PATH='Caps/'
    filepath = os.path.join(MODEL_PATH,'my_net_model.h5')
    if not os.path.exists(MODEL_PATH): 
        os.makedirs(MODEL_PATH) 

    model_output = Dense(2, activation="softmax")(x1)
    adam = Adam(lr=0.001)
    model=Model(inputs=model_input1, outputs=model_output)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    print ('model training')
    earlystopper = EarlyStopping(monitor='loss', patience=5, verbose=0)
    s_time = time.strftime("%Y%m%d%H%M%S", time.localtime())  
    logs_path = "logs/my_board/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        os.makedirs(logs_path)
    except:
        pass
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True,
                            mode='max')
    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5,  
                                            min_lr=0.00001)
    
    history=model.fit(x_train,y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, callbacks=[learning_rate_reduction,checkpoint,TensorBoard(log_dir=logs_path,histogram_freq=0,write_graph=True, write_images=True),earlystopper])
    model.summary()
    #print(model.summary())
    model.save(filepath)
    return  model

def train_caps(data_file):
    
    print("Load data...")
    x_train, y_train,encoder1= load_data(data_file)
    #print("x_dev shape:", x_dev.shape)

    x_train=x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2]))

    net_one,x1,model_input1= get_cnn_network_one(max_len1 )
    model = run_network(net_one,x1,model_input1, x_train, y_train )

   

def test_caps(data_file):
    outfile='Caps/prediction.txt'
    fprfile='Caps/fpr_file.txt'
    tprfile='Caps/tpr_file.txt'
    metricsfile="Caps/metrics_file.txt"
    positivefile = 'Positive_sequence.txt'
    negativefile = 'Negative_sequence.txt'
    if not os.path.exists('Caps/'):
        os.makedirs('Caps/')

    print("Load test data...")
    x_test, y_test= load_test_data(data_file)
    print("x_test shape:", x_test.shape)
    x_test= x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2]))

    model = load_model('Caps/my_net_model.h5',custom_objects={'Capsule': Capsule})
    
    test_predictions = model.predict(x_test,verbose=1)

    predictions_label = transfer_label_from_prob(test_predictions[:, 1])
    pos, neg =classify_with_predict_label(predictions_label,data_file )
    
    with open(positivefile, 'w') as f:
        writething = "\n".join(map(str, pos))
        f.write(writething)
    with open(negativefile, 'w') as f:
        writething = "\n".join(map(str, neg))
        f.write(writething)
    
    fw = open(outfile, 'w')
    myprob = "\n".join(map(str, test_predictions))
    fw.write(myprob)
    fw.close()
    
    #print(y_test)

    fpr,tpr,thresholds = roc_curve(y_test,test_predictions[:, 1])
    with open(fprfile, 'w') as f:
        writething = "\n".join(map(str, fpr))
        f.write(writething)
    with open(tprfile, 'w') as f:
         writething = "\n".join(map(str, tpr))
         f.write(writething)

    acc, sensitivity, specificity, MCC ,PPV, NPV = calculate_performance(len(y_test), predictions_label, y_test)
    roc_auc = auc(fpr, tpr)
    out_rel = ['acc', acc, 'sn', sensitivity, 'sp', specificity, 'MCC', MCC, 'auc', roc_auc , 'PPV' ,PPV , 'NPV' , NPV]

    with open(metricsfile, 'w') as f:
         writething = "\n".join(map(str, out_rel))
         f.write(writething)

    print("ACC——%.4f" %acc)       
    print("sensitivity——%.4f" %sensitivity )       
    print("specificity——%.4f "%specificity )      
    print("PPV——%.4f " %PPV)       
    print("NPV——%.4f " %NPV)       
    print("MCC——%.4f " %MCC)       
    print("roc_auc——%.4f" %roc_auc)       
    
    plt.plot(
            fpr,
            tpr,
            linestyle='-',
            label='{} ({:0.3f})'.format("AUC_CNN", roc_auc), lw=2.0)

    auROCplot()

def classify_with_predict_label(label, data_file):
    
    positive_seq = []
    negative_seq = []
    fp = open(data_file,'r')
    cnt = 0
    seq_id = ''
    for line in fp:
        if line[0] == '>':
            seq_id= line[0:-1]
        else:
            line = line[0:-1]
            if label[cnt] == 1:
                positive_seq.append(seq_id)
                positive_seq.append(line)
            else :
                negative_seq.append(seq_id)
                negative_seq.append(line)
            cnt = cnt +1 

    return positive_seq, negative_seq


def auROCplot():
    ### auROC ###
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
    plt.title('Caps Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig('auROC_caps.png', dpi=300)
    plt.show()
    ### --- ###

if __name__ == "__main__":
    train_caps(data_file= "Datasets/%s/train/%s/sequence.fa"%(filename[25],number))
    test_caps(data_file= "Datasets/%s/test/%s/sequence.fa"%(filename[25],number))
