from CNN import embed,load_label_seq,preprocess_labels,GetSeqDegree,transfer_label_from_prob,classify_with_predict_label,calculate_performance
from CNN import filename ,number

from tensorflow.keras import datasets, layers, models
from keras.models import load_model
from sklearn.metrics import roc_curve, auc

import numpy as np
import math
import os

k=20
N=16
m=3
l=8

def set_convolution_layer():
    #input_shape = (98+k , 256)     
    input_shape = (98+k , 2401)     


    model = models.Sequential()
    model.add(layers.Conv1D(N, k, padding='valid',input_shape = input_shape))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=m))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv1D(N, int(k/2), padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=m))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(l, activation='relu'))
    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))
    
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

    return model

def read_structure(structure_file,k):
    encoding = 'utf-8'
    degree = 4
    encoder = buildstructuremapper(degree)
    structure_list = []
    structure = ''

    fp = open(structure_file,'r')
    for line in fp:
        if line[0] == '>':
            name = line[1:-1]
            if len(structure):
                structuredata = GetSeqDegree(structure,degree,k)
                structure_array = embed(structuredata,encoder)
                structure_list.append(structure_array)
            structure = ''
        else:
            structure = structure + line[:-1]
    if len(structure):
        structuredata = GetSeqDegree(structure, degree,k)
        structure_array = embed(structuredata, encoder)
        structure_list.append(structure_array)
    return np.array(structure_list)

def load_data_file(data_file , k):
    tmp = []
    tmp.append(read_structure(data_file,k))
    data = dict()
    data["struct"] = tmp
    data["Y"] = load_label_seq(data_file)
    return data
    
def buildstructuremapper(degree):
    length = degree
    alphabet = ['S','H','M','I','B','X','E']  
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

    number = int(math.pow(7, degree))
    encoder['N'] = [1.0 / number] * number
    return encoder

def train_sturcture(data_file):
    data = load_data_file(data_file, k)
    train_Y = data["Y"]
    struct_data = data["struct"][0]
    
    y = preprocess_labels(train_Y)

    my_classifier= set_convolution_layer()

    print(np.shape(struct_data))

    my_classifier.fit(struct_data, y[0], epochs=50)
    my_classifier.save('structure_cnn_model.pkl')


def test_structure(data_file):
    outfile = 'Capsnet/prediction.txt'
    fprfile = 'Capsnet/fpr.txt'
    tprfile = 'Capsnet/tpr.txt'
    metricsfile = 'Capsnet/metrics_file.txt'
    positivefile = 'Positive_structure.txt'
    negativefile = 'Negative_structure.txt'
    if not os.path.exists('Capsnet/'):
        os.makedirs('Capsnet/')

    print ('model prediction')

    data = load_data_file(data_file, k)
    true_y = data["Y"]

    testing = data["struct"][0]  # it includes one-hot encoding sequence and structure
    model = load_model('structure_cnn_model.pkl')
    
    predictions = model.predict(testing)
    predictions_label = transfer_label_from_prob(predictions[:, 1])
    pos, neg =classify_with_predict_label(predictions_label,data_file )
    #print(pos,neg)
    with open(positivefile, 'w') as f:
        writething = "\n".join(map(str, pos))
        f.write(writething)
    with open(negativefile, 'w') as f:
        writething = "\n".join(map(str, neg))
        f.write(writething)
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

    acc, sensitivity, specificity, MCC ,PPV , NPV = calculate_performance(len(true_y), predictions_label, true_y)
    roc_auc = auc(fpr, tpr)

    out_rel = ['acc', acc, 'sn', sensitivity, 'sp', specificity, 'MCC', MCC, 'auc', roc_auc ,'PPV', PPV , 'NPV',NPV]
    with open(metricsfile, 'w') as f:
        writething = "\n".join(map(str, out_rel))
        f.write(writething)

    print ("acc,  sensitivity, specificity, MCC, auc, PPV, NPV : ", acc, sensitivity, specificity, MCC, roc_auc, PPV, NPV)   
if __name__ == "__main__":
    #train_sturcture(data_file="Structure_train.txt")
    #test_structure(data_file="Structure_test.txt")
    train_sturcture(data_file= "Datasets/%s/train/%s/structure.txt"%(filename[25],number))
