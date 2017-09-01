import time
import numpy as np
import cv2
import skimage.feature
import keras
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mpld3
from PIL import Image 
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model
from tempfile import TemporaryFile

r = 0.4     #scale down
width = 100 #patch size 
start_time = time.time()

img_path = "../corpus/KaggleNOAASeaLions_small"

Tbeg_time = time.time()

def GetData(filename):
    # read the Train and Train Dotted images
    image_1 = cv2.imread(img_path  + '/TrainDotted/' + filename)
    image_2 = cv2.imread(img_path + '/Train/' + filename)    
    img1 = cv2.GaussianBlur(image_1,(5,5),0)

    # absolute difference between Train and Train Dotted
    if(image_1.shape !=  image_2.shape):
        #skip due to size not equal
        return np.array([]),np.array([]),'error_no_2'
    image_3 = cv2.absdiff(image_1,image_2)
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 50] = 0
    mask_1[mask_1 > 0] = 255
    image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)

    # convert to grayscale to be accepted by skimage.feature.blob_log
    image_6 = np.max(image_4,axis=2)
  
    # alvin: check blog_log
    rmse = np.sqrt((image_3**2).mean())
    if(rmse > 8.0):
        #print('skip due to blob failed')
        return np.array([]),np.array([]),'error_no_1'
    
    
    # detect blobs
    blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.05)

    h,w,d = image_2.shape

    res=np.zeros((int((w*r)//width)+1,int((h*r)//width)+1,5), dtype='int16')

    for blob in blobs:
        # get the coordinates for each blob
        y, x, s = blob
        # get the color of the pixel from Train Dotted in the center of the blob
        b,g,R = img1[int(y)][int(x)][:]
        x1 = int((x*r)//width)
        y1 = int((y*r)//width)
        # decision tree to pick the class of the blob by looking at the color in Train Dotted
        if R > 225 and b < 25 and g < 25: # RED
            res[x1,y1,0]+=1
        elif R > 225 and b > 225 and g < 25: # MAGENTA
            res[x1,y1,1]+=1
        elif R < 75 and b < 50 and 150 < g < 200: # GREEN
            res[x1,y1,4]+=1
        elif R < 75 and  150 < b < 200 and g < 75: # BLUE
            res[x1,y1,3]+=1
        elif 60 < R < 120 and b < 50 and g < 75:  # BROWN
            res[x1,y1,2]+=1

    ma = cv2.cvtColor((1*(np.sum(image_1, axis=2)>20)).astype('uint8'), cv2.COLOR_GRAY2BGR)
    img = cv2.resize(image_2 * ma, (int(w*r),int(h*r)))
    h1,w1,d = img.shape

    trainX = []
    trainY = []

    for i in range(int(w1//width)):
        for j in range(int(h1//width)):
            trainY.append(res[i,j,:])
            trainX.append(img[j*width:j*width+width,i*width:i*width+width,:])
            #print(str(i)+',',str(j),' ==>\t')
            #print(str(j*width)+':'+str(j*width+width)+' , ' + str(i*width)+':'+str(i*width+width) + ' , :')
    return np.array(trainX), np.array(trainY), 'ok'
    #return trainX,trainY

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

model = load_model('vgg16_full.h5')



filelist = os.listdir(img_path + "/Train/")
filelist = filter(lambda x: x.endswith('.jpg'), filelist)
filelist = list(filelist)
filelist.sort(key=lambda x : int(x[:len(x)-4]))
filelist = list(filelist)
print('Total tesing file = ' + str(len(filelist)))

sel_fid  = '2.jpg'
 
n_train      = len(filelist)
trainset     = np.arange(n_train)
five_rmse_vec= np.array([])
dist_vec     = np.zeros((1,5))
    
trainX, trainY,flag = GetData(sel_fid)
if(flag == 'ok'):
    print('Testing '+ sel_fid)
else:
    print('Testing\t' + sel_fid + '\t' + flag)
    exit()

#result = model.predict(trainX)
#np.save('demo-1', result)    
result = np.load('demo-2.npy')
ground_truth_vec = np.sum(trainY, axis=0)
prediction_vec   = np.sum(result*(result>0.3), axis=0).astype('int')
five_class_rmse  = rmse(prediction_vec,ground_truth_vec)
five_rmse_vec    = np.append(five_rmse_vec,five_class_rmse)
dist_vec        += np.abs(prediction_vec-ground_truth_vec)


    
print(sel_fid +' ==> CNN VGG16 output--')
print('    ground truth: ', ground_truth_vec)
print('  evaluate count: ', prediction_vec)
print('      difference: ', str(np.abs(prediction_vec-ground_truth_vec)))
print('            rmse: ', str(five_class_rmse))


# re-generate image
#print(result)

image_1 = cv2.imread(img_path + '/TrainDotted/' + sel_fid);

h,w,d = image_1.shape
#print('w:'+str(w)+', h:'+str(h)+' ,d:' + str(d))
fig, ax = plt.subplots(1,1,figsize=(15,15))
new_width = int(width / r)

ax.imshow(image_1[:,:,[2,1,0]])

sealion_loc_list = []
invisiblex       = []
invisibley       = []

for i in range(result.shape[0]):
    raw_block  = int(h/new_width)
    x_idx  = int(i/raw_block)
    y_idx  = i%raw_block
    #print(str(i) + '--> ' + str(y_idx) + ',' + str(x_idx))
    result_i = result[i][:]
    #recx, recy, width, high = [x_idx*new_width,y_idx*new_width,new_width,new_width]

    if(sum(result_i*(result_i>0.3))):
        recx, recy, width, high = [x_idx*new_width,y_idx*new_width,new_width,new_width]
        rect = patches.Rectangle((recx,recy),width,high,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        ax.text(recx+new_width/15,recy+new_width/4,str(i),fontsize='8')
        sealion_loc_list.append(i)
        invisiblex.append(recx+int(new_width/2))
        invisibley.append(recy+int(new_width/2))
line, = ax.plot(invisiblex,invisibley,'o',picker=int(new_width/10),alpha=0)

def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    points = tuple(zip(xdata[ind], ydata[ind]))
    # print('onpick points:', points)
    
    ### mouse point to block x,y, retrieve block image 
    loc_id = int(points[0][0]/new_width)*raw_block + int(points[0][1]/new_width)
    x_idx  = int(loc_id/raw_block)
    y_idx  = loc_id%raw_block
    img_loc_id = image_1[y_idx*new_width:y_idx*new_width+new_width, x_idx*new_width:x_idx*new_width+new_width, [2,1,0]]
   
    # close previos figure 
    fig_vect = plt.get_fignums()
    if(fig_vect[-1] > 1): plt.close(fig_vect[-1])

    # plot detail image
    fig2, ax2 = plt.subplots(1,2,sharey=True,figsize=(10,5))
    ax2[0].imshow(img_loc_id)
    
    ax2[1].set_xlim([0,new_width])
    ax2[1].set_ylim([0,new_width])
    ax2[1].set_axis_off()
    #plot lable color discrition
    h_w_size   =  40
    rect_red   =  patches.Rectangle((20+h_w_size*0,new_width-h_w_size-10),h_w_size,h_w_size,color='red')
    rect_man   =  patches.Rectangle((20+h_w_size*1,new_width-h_w_size-10),h_w_size,h_w_size,color='magenta')
    rect_brown =  patches.Rectangle((20+h_w_size*2,new_width-h_w_size-10),h_w_size,h_w_size,color='brown')
    rect_blue  =  patches.Rectangle((20+h_w_size*3,new_width-h_w_size-10),h_w_size,h_w_size,color='blue')
    rect_green =  patches.Rectangle((20+h_w_size*4,new_width-h_w_size-10),h_w_size,h_w_size,color='green')
    ax2[1].add_patch(rect_red)
    ax2[1].add_patch(rect_man)
    ax2[1].add_patch(rect_brown)
    ax2[1].add_patch(rect_blue)
    ax2[1].add_patch(rect_green)
    ax2[1].text(20+h_w_size*0+int(h_w_size/2),new_width-h_w_size-22,'adult males',rotation=-45,fontsize=12)
    ax2[1].text(20+h_w_size*1+int(h_w_size/2),new_width-h_w_size-22,'subadult male',rotation=-45,fontsize=12)
    ax2[1].text(20+h_w_size*2+int(h_w_size/2),new_width-h_w_size-22,'adult females',rotation=-45,fontsize=12)
    ax2[1].text(20+h_w_size*3+int(h_w_size/2),new_width-h_w_size-22,'juveniles',rotation=-45,fontsize=12)
    ax2[1].text(20+h_w_size*4+int(h_w_size/2),new_width-h_w_size-22,'pup',rotation=-45,fontsize=12)
    ax2[1].text(5, 40,'img process label: '+ str(trainY[loc_id][:]), fontsize=12)
    result_i = result[loc_id][:]
    result_i = result_i*(result_i>0.3)
    result_i = [round(float(np.float32(x)),3) for x in result_i]
    ax2[1].text(5, 20 ,'Prediction  : '+ str(result_i), fontsize=12)
    #print('result_i.shape is ' + str(result_i.shape))
    fig2.subplots_adjust(wspace=0)
    fig2.show()
    return True
    
fig.canvas.mpl_connect('pick_event', onpick)

'''
num_of_sealion_block = len(sealion_loc_list)

#plt.figure(1)
num_of_sealion_block = 2
fig2, ax = plt.subplots(num_of_sealion_block,1,figsize=(6,num_of_sealion_block*5))

for i in range(num_of_sealion_block):
    loc_id = sealion_loc_list[i]
    x_idx  = int(loc_id/raw_block)
    y_idx  = loc_id%raw_block
    img_i  = image_1[y_idx*new_width:y_idx*new_width+new_width, x_idx*new_width:x_idx*new_width+new_width, [2,1,0]]
    ax[i].imshow(img_i)
'''
def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    points = tuple(zip(xdata[ind], ydata[ind]))
    print('onpick points:', points)
    #print('alvin '+str(xdata)+','+str(ydata))

fig.canvas.mpl_connect('pick_event', onpick)
plt.show()



end_time = time.time()
print('spend ' + str(round(end_time-start_time,1))+'s')

#mpld3.show()
