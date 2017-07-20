#%matplotlib inline
import numpy as np
#import matplotlib.pyplot as plt
from six.moves import cPickle 
import time

def load_cifar10(dataset_number):

    for i in range(dataset_number):
        filename = '/home/alvin/notebook_home/ai_lab_image/corpus/cifar-10-batches-py/data_batch_' +str(i+1)
        f = open(filename, 'rb')
        datadict = cPickle.load(f,encoding='latin1')
        f.close()
    
        DX= datadict["data"] # 1000 x (32x32x3)
        DY= datadict['labels']
        #X = DX.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
        X[i*10000:(i+1)*10000] = np.array(DX)
        Y[i*10000:(i+1)*10000] = np.array(DY)
        #break
    return X,Y

def l1_dist(input_vec, train_set):
    l1_formula = lambda x: sum(abs(input_vec - x))
    l1_dist_vec= l1_formula(train_set)
    return l1_dist_vec

def l2_dist(input_vec, train_set):
    l2_formula = lambda x: np.sqrt(sum((input_vec - x) ** 2))
    l2_dist_vec= l2_formula(train_set)
    return l2_dist_vec

cifar_label = { '0':'airplane',
                '1':'automobile',
                '2':'bird',
                '3':'cat',
                '4':'deer',
                '5':'dog',
                '6':'frog',
                '7':'horse',
                '8':'ship',
                '9':'truck'}

dataset_number = 1
X = np.zeros(shape=(dataset_number*10000,32*32*3),dtype='float64')
Y = np.empty([dataset_number * 10000], dtype='uint8')
X,Y = load_cifar10(dataset_number)

XG = X.reshape(10000*dataset_number, 3, 1024).transpose(0,2,1).astype("float64") #10000 x 3 x 1024 --> 10000 x 1024 x 3
XG = np.mean(XG,axis=2)                                         #10000 x 1024

# training set and testing test
t_rate   = 0.8
train_set_num = int(len(X)*t_rate)
train_set   = XG[:train_set_num]
train_label = Y[:train_set_num]
test_set    = XG[train_set_num+1:]
test_label  = Y[train_set_num+1:]

Stime = time.time()
k = 51
correct_num = 0
pred = np.zeros(shape=(len(test_set),1),dtype='uint8')
for i in range(len(test_set)):
    test_sample = test_set[i]
    dict_vec    = l1_dist(test_sample, train_set)
    idx         = np.argpartition(dict_vec,k)[0:k]
    vote        = train_label[idx[0:k]]
    #tmp = np.bincount(vote)
    #print(str(np.bincount(vote)))
    #print(str(np.argmax(tmp)))
    pred[i]     = np.argmax(np.bincount(vote))
    #if i > 100: break
    vostr       = ''.join(str(vote))
    print(str(test_label[i]) + ' , ' + vostr + ' --> ' + str(pred[i]) + '\t progress ' + str(i) + '/' + str(len(test_set)))
    
    if test_label[i] == train_label[idx[0]]:
        correct_num += 1
Etime = time.time()
accuracy = correct_num/len(test_set)
print('knn accuarcy is ' + str(float("{0:.3f}".format(accuracy*100)))+'%')
print('exectution time knn w/ k = ' +str(k)+ ' is ' + str(float(("{0:.2f}".format(Etime-Stime)))) + 's')


