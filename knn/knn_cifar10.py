import numpy as np
#import matplotlib.pyplot as plt
from six.moves import cPickle 
import time
import scipy.spatial.distance as sp

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
    #l1_formula = lambda x: sp.cdist(input_vec,x,'city_block')
    #l1_dist_vec= l1_formula(train_set)
    l1_dist_vec = sp.cdist(train_set,[input_vec],'cityblock')
    return l1_dist_vec

def l2_dist(input_vec, train_set):
    l2_dist_vec = sp.cdist(train_set,[input_vec],'euclidean')
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

###############################################################
#   knn main code
###############################################################
###################################
# allocate space and load dataset
###################################
dataset_number = 5
X = np.zeros(shape=(dataset_number*10000,32*32*3),dtype='float64')
Y = np.empty([dataset_number * 10000], dtype='uint8')
X,Y = load_cifar10(dataset_number)


#####################################################
# change color to gray
#####################################################
XG = X.reshape(10000*dataset_number, 3, 1024).transpose(0,2,1).astype("float64") #10000 x 3 x 1024 --> 10000 x 1024 x 3
XG = np.mean(XG,axis=2)  #10000 x 1024
XG /= np.std(XG,axis=0)
#####################################################
# assign traing set and testing set and run cross validation
#####################################################

t_rate   = 0.8
test_set_num  = int(round(len(X)*(1-t_rate),0))
for iter_num in range(5):
    test_index  = np.arange(iter_num*test_set_num, (iter_num+1)*test_set_num, 1)
    train_set   = np.delete(XG,test_index,0)
    train_label = np.delete(Y,test_index,0)
    test_set    = np.array(XG[test_index])
    test_label  = np.array(Y[test_index])
    
    #####################################################
    # run KNN algorithm
    #####################################################

    Stime = time.time()
    k = 1
    correct_num = 0
    pred = np.zeros(shape=(len(test_set),1),dtype='uint8')
    for i in range(len(test_set)):
        test_sample = test_set[i]
        dict_vec    = l1_dist(test_sample, train_set)
        #print(dict_vec)
        idx         = np.argpartition(dict_vec.reshape((-1,)),k)[0:k]
        vote        = train_label[idx[0:k]]
        pred[i]     = np.argmax(np.bincount(vote))
        #if i > 2: break
        #print(str(test_label[i]) + ' , ' + str(vote) + ' --> ' + str(pred[i]) + '\t progress ' + str(i) + '/' + str(len(test_set)))
    
        if test_label[i] == train_label[idx[0]]:
            correct_num += 1
    Etime = time.time()
    accuracy = correct_num/len(test_set)
    print('knn accuarcy is ' + str(float("{0:.3f}".format(accuracy*100)))+'%')
    print('exectution time knn w/ k = ' +str(k)+ ' is ' + str(float(("{0:.2f}".format(Etime-Stime)))) + 's')
