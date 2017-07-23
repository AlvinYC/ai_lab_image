{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "%matplotlib inline\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from six.moves import cPickle \n",
    "import time\n",
    "import scipy.spatial.distance as sp"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def load_cifar10(dataset_number):\n",
    "\n",
    "    for i in range(dataset_number):\n",
    "        filename = '/home/alvin/notebook_home/ai_lab_image/corpus/cifar-10-batches-py/data_batch_' +str(i+1)\n",
    "        f = open(filename, 'rb')\n",
    "        datadict = cPickle.load(f,encoding='latin1')\n",
    "        f.close()\n",
    "    \n",
    "        DX= datadict[\"data\"] # 1000 x (32x32x3)\n",
    "        DY= datadict['labels']\n",
    "        #X = DX.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype(\"uint8\")\n",
    "        X[i*10000:(i+1)*10000] = np.array(DX)\n",
    "        Y[i*10000:(i+1)*10000] = np.array(DY)\n",
    "        #break\n",
    "    return X,Y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def dataset_crossvalidate()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def l1_dist(input_vec, train_set):\n",
    "    #l1_formula = lambda x: sp.cdist(input_vec,x,'city_block')\n",
    "    #l1_dist_vec= l1_formula(train_set)\n",
    "    l1_dist_vec = sp.cdist(train_set,[input_vec],'cityblock')\n",
    "    return l1_dist_vec\n",
    "\n",
    "def l2_dist(input_vec, train_set):\n",
    "    l2_dist_vec = sp.cdist(train_set,[input_vec],'euclidean')\n",
    "    return l2_dist_vec"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "cifar_label = { '0':'airplane',\n",
    "                '1':'automobile',\n",
    "                '2':'bird',\n",
    "                '3':'cat',\n",
    "                '4':'deer',\n",
    "                '5':'dog',\n",
    "                '6':'frog',\n",
    "                '7':'horse',\n",
    "                '8':'ship',\n",
    "                '9':'truck'}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# show each class sample number on training data\n",
    "print('dataset count = ' + str(len(X)))\n",
    "for i in range(10):\n",
    "    print('number of class '+ str(i) + ': ' + str(Y[np.where(Y==i)].size))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#[option] show label information\n",
    "Y[10000:10010]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#[option] display cifar image with color\n",
    "#Visualizing CIFAR 10 method 2\n",
    "X3 = X.reshape(10000*dataset_number, 3, 32, 32).transpose(0,2,3,1)\n",
    "pic_dim = 6\n",
    "fig, axes1 = plt.subplots(pic_dim,pic_dim,figsize=(6,6))\n",
    "for i in range(pic_dim*pic_dim):\n",
    "    #print('i/pic_dim: ' + str(i/pic_dim) + 'i%pic_dim: ' + str(i%pic_dim))\n",
    "    axes1[int(i/pic_dim)][int(i%pic_dim)].set_axis_off()\n",
    "    axes1[int(i/pic_dim)][int(i%pic_dim)].imshow(X3[i])\n",
    "    axes1[int(i/pic_dim)][int(i%pic_dim)].annotate(cifar_label[str(Y[i])],xy=(0.2,0.6))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#[option] display cifar image with gray\n",
    "#Visualizing CIFAR 10 method 2\n",
    "XG = X.reshape(10000*dataset_number, 3, 1024).transpose(0,2,1).astype(\"float64\") #1000 x 3 x 1024 --> 1000 x 1024 x 3\n",
    "XG = np.mean(XG,axis=2)                                         #1000 x 1024\n",
    "XG = XG.reshape(10000*dataset_number, 32, 32)\n",
    "pic_dim = 6\n",
    "fig, axes1 = plt.subplots(pic_dim,pic_dim,figsize=(6,6))\n",
    "for i in range(pic_dim*pic_dim):\n",
    "    #print('i/pic_dim: ' + str(i/pic_dim) + 'i%pic_dim: ' + str(i%pic_dim))\n",
    "    axes1[int(i/pic_dim)][int(i%pic_dim)].set_axis_off()\n",
    "    axes1[int(i/pic_dim)][int(i%pic_dim)].imshow(XG[i], cmap='gray')\n",
    "    axes1[int(i/pic_dim)][int(i%pic_dim)].annotate(cifar_label[str(Y[i])],xy=(0.2,0.6))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "###############################################################\n",
    "#   knn main code\n",
    "###############################################################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "###################################\n",
    "# allocate space and load dataset\n",
    "###################################\n",
    "dataset_number = 5\n",
    "X = np.zeros(shape=(dataset_number*10000,32*32*3),dtype='float64')\n",
    "Y = np.empty([dataset_number * 10000], dtype='uint8')\n",
    "X,Y = load_cifar10(dataset_number)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "#####################################################\n",
    "# change color to gray\n",
    "#####################################################\n",
    "XG = X.reshape(10000*dataset_number, 3, 1024).transpose(0,2,1).astype(\"float64\") #10000 x 3 x 1024 --> 10000 x 1024 x 3\n",
    "XG = np.mean(XG,axis=2)  #10000 x 1024"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "knn accuarcy is 0.0%\n",
      "exectution time knn w/ k = 1 is 0.11s\n"
     ]
    }
   ],
   "source": [
    "#####################################################\n",
    "# assign traing set and testing set and run cross validation\n",
    "#####################################################\n",
    "\n",
    "t_rate   = 0.8\n",
    "test_set_num  = int(round(len(X)*(1-t_rate),0))\n",
    "for iter_num in range(1):\n",
    "    test_index  = np.arange(iter_num*test_set_num, (iter_num+1)*test_set_num, 1)\n",
    "    train_set   = np.delete(XG,test_index,0)\n",
    "    train_label = np.delete(Y,test_index,0)\n",
    "    test_set    = np.array(XG[test_index])\n",
    "    test_label  = np.array(Y[test_index])\n",
    "    \n",
    "    #####################################################\n",
    "    # run KNN algorithm\n",
    "    #####################################################\n",
    "\n",
    "    Stime = time.time()\n",
    "    k = 1\n",
    "    correct_num = 0\n",
    "    pred = np.zeros(shape=(len(test_set),1),dtype='uint8')\n",
    "    for i in range(len(test_set)):\n",
    "        test_sample = test_set[i]\n",
    "        dict_vec    = l1_dist(test_sample, train_set)\n",
    "        #print(dict_vec)\n",
    "        idx         = np.argpartition(dict_vec.reshape((-1,)),k)[0:k]\n",
    "        vote        = train_label[idx[0:k]]\n",
    "        pred[i]     = np.argmax(np.bincount(vote))\n",
    "        #if i > 2: break\n",
    "        #print(str(test_label[i]) + ' , ' + str(vote) + ' --> ' + str(pred[i]) + '\\t progress ' + str(i) + '/' + str(len(test_set)))\n",
    "    \n",
    "        if test_label[i] == train_label[idx[0]]:\n",
    "            correct_num += 1\n",
    "    Etime = time.time()\n",
    "    accuracy = correct_num/len(test_set)\n",
    "    print('knn accuarcy is ' + str(float(\"{0:.3f}\".format(accuracy*100)))+'%')\n",
    "    print('exectution time knn w/ k = ' +str(k)+ ' is ' + str(float((\"{0:.2f}\".format(Etime-Stime)))) + 's')    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#dataset_number = 2 \n",
    "#20000, testing sample 4000\n",
    "'''\n",
    "knn accuarcy is 28.775%\n",
    "exectution time knn w/ k = 1 is 96.42s\n",
    "knn accuarcy is 27.825%\n",
    "exectution time knn w/ k = 1 is 95.66s\n",
    "knn accuarcy is 27.6%\n",
    "exectution time knn w/ k = 1 is 95.76s\n",
    "knn accuarcy is 28.475%\n",
    "exectution time knn w/ k = 1 is 95.18s\n",
    "knn accuarcy is 29.225%\n",
    "exectution time knn w/ k = 1 is 94.74s\n",
    "'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "dict_vec.reshape((-1,))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# X uint8\n",
    "# knn gray with k=3, 20,000[16,000, 4000],  dtype=uint8  = 10.578%\n",
    "# knn gray with k=3, 30,000[24,000, 6000],  dtype=uint8  = 10.502% (1827.51s->30.45m)\n",
    "\n",
    "# X float64\n",
    "# knn gray with k=3, 10,000[ 8,000, 2000], dtype=float64 = 11.056% (214.57s-> 3.57m)\n",
    "# knn gray with k=3, 20,000[16,000, 4000], dtype=float64 = 10.578% (815.93s->13.59m)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "k= 1\n",
    "test_sample = test_set[0]\n",
    "l1_dict_vec = l1_dist(test_sample,train_set)\n",
    "idx = np.argpartition(l1_dict_vec,k)\n",
    "#\n",
    "print(Y[train_set_num+0])\n",
    "print(Y[train_set_num+idx[0]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "int_v = test_set[0]/255\n",
    "trn_v = train_set[0]/255\n",
    "print(int_v)\n",
    "print(trn_v)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "print('np.linalg.norm = ' + str(np.linalg.norm((int_v - trn_v), ord=2)))\n",
    "dist_l1 = sum(abs(int_v - trn_v))\n",
    "print('sum(abs(a-b)) = ' + str(dist_l1))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "int_v = np.array([1,1,1,1,1])\n",
    "tring_set = np.array([[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4],[5,5,5,5,5]])\n",
    "#dist_lambda = lambda x: np.linalg.norm((int_v - x), ord=1)\n",
    "dist_lambda = lambda x: sum(abs(int_v - x))\n",
    "#dist_vec = map{lamda x: np.linalg.norm((int_v - x), ord=1),train_set}\n",
    "dist_vec = dist_lambda(tring_set)\n",
    "dist_vec"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "a = np.array([[1,0,1,0,1]])\n",
    "b = np.array([[2,2,2,2,2],[3,3,3,3,3]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "l1_dist_vec = sp.cdist(b,a,'cityblock')\n",
    "l1_dist_vec"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "a = np.array([1,2,3,4,5])\n",
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "b = np.delete([a],[1,3],1)\n",
    "b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "t_rate   = 0.8\n",
    "test_set_num  = int(round((len(X)*(1-t_rate)),0))\n",
    "iter_number = 2\n",
    "test_index = np.arange(iter_number*test_set_num, (iter_number+1)*test_set_num, 1)\n",
    "train_set   = np.delete(XG,test_index,0)\n",
    "train_lable = np.delete(Y,test_index,0)\n",
    "test_set    = np.array(XG[test_index])\n",
    "test_label  = np.array(Y[test_index])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_set.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.delete(np.linspace(1,10,10,dtype='uint8'),np.linspace(5,8,4,dtype='uint8'),0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_set.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_index = np.arange(iter_num*test_set_num, (iter_number+1)*test_set_num, 1)\n",
    "test_index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "iter_num*test_set_num"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_index[1600]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
