#####Note:run in python2######


import math
import numpy as np
import sys
from PIL import Image
import matplotlib.pyplot as plt 

np.random.seed(42)



def exceptions(img,conv_filter):

    if len(img.shape) > 2 or len(conv_filter.shape) > 3: # Check if number of image channels matches the filter depth.
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()

    ###
    #as discussed in the class that the centre of kernel must exist for the mathematical operation of convolution to be defined
    #thus they should odd size
    ###
    if conv_filter.shape[1]%2==0 or conv_filter.shape[2]%2==0: # Check if filter diemnsions are odd.
        print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
        sys.exit()



def pad_input(img, Mrow, Mcol, stride):

    N = img.shape[0]

    ##padding along rows
    p1 = int(math.ceil(((img.shape[0]-1)*(stride[0]) + Mrow -img.shape[0])/2))
    for i in range(0,p1):
        img=np.vstack((np.zeros((1,img.shape[1],img.shape[2])),img))
        img=np.vstack((img,np.zeros((1,img.shape[1],img.shape[2]))))

    #padding along column
    p2 = int(math.ceil(((img.shape[1]-1)*(stride[1]) + Mcol -img.shape[1])/2))
    for i in range(0,p2):
        img=np.hstack((np.zeros((img.shape[0],1,img.shape[2])),img))
        img=np.hstack((img,np.zeros((img.shape[0],1,img.shape[2]))))

    return img

def apply_activationCONV(Activation_function,inp):
    
    #activation functions
    if Activation_function == "relu":
        return max(0,inp)
    elif Activation_function == "tanh":
        return np.tanh(inp)
    elif Activation_function == "sigmoid":
        return 1.0/(1+np.exp(-1.0*inp))


def conv(img, conv_filter, padding, stride, Activation_function):

    # print img.shape
    if padding == 'same':

        img = np.array(pad_input(img,conv_filter.shape[1],conv_filter.shape[2],stride))
        # print img.shape

    # see for exceptions
    exceptions(img,conv_filter)

    # An empty feature map to hold the output of convolving the filter(s) with the image.
    feature_maps = np.zeros((   (img.shape[0]-conv_filter.shape[1])/stride[0] + 1 , 
                                (img.shape[1]-conv_filter.shape[2])/stride[1] + 1 , conv_filter.shape[0] ))

    # Convolving the image by the filter(s).
    for filter_num in range(conv_filter.shape[0]):

        curr_filter = conv_filter[filter_num, :]

        filter_sizeRow = curr_filter.shape[0]

        filter_sizeCol = curr_filter.shape[1]

        #Applying the convolution operation.
        for r in np.arange(filter_sizeRow/2,img.shape[0]-filter_sizeRow/2,stride[0]):
            for c in np.arange(filter_sizeCol/2,img.shape[1]-filter_sizeCol/2,stride[1]):
                #Region required
                curr_region = img[r-filter_sizeRow/2:r+filter_sizeRow/2+1,c-filter_sizeCol/2:c+filter_sizeCol/2+1,:]
                #Element-wise multipliplication between the current region and the filter.
                curr_result = curr_region * curr_filter
                #Summing the result of multiplication.
                feature_maps[(r-(filter_sizeRow/2))/stride[0],
                             (c-(filter_sizeCol/2))/stride[1], filter_num] =apply_activationCONV(Activation_function,np.sum(curr_result)) 

    return feature_maps # Returning all feature maps.


#pooling funcion 
def pooling(feature_map,size=2,stride=2,pool_function='max'):

    #Preparing the output of the pooling operation.
    pool_out = np.zeros(((feature_map.shape[0]-size)/stride+1,
                            (feature_map.shape[1]-size)/stride+1,feature_map.shape[-1]))

    r2=0
    #variables r2 and c2 are for keeping a track of rows and columns in final pool_out 
    for row in np.arange(0,feature_map.shape[0]-size+1, stride):
        c2 = 0
        for col in np.arange(0, feature_map.shape[1]-size+1, stride):
            if pool_function == 'max':
                pool_out[r2, c2, :] = np.amax(feature_map[row:row+size,  col:col+size,:],(0,1)).reshape((1,1,feature_map.shape[-1]))
            if pool_function == 'average':
                pool_out[r2, c2, :] = np.mean(feature_map[row:row+size,  col:col+size,:],(0,1)).reshape((1,1,feature_map.shape[-1]))
            c2 = c2 + 1
        r2 = r2 +1

    return pool_out




#########
#composition of convolution layer functions
def compConv(img,NoOfConv,kernelMatrix,strideVect,padVect,actfnVect,poolfnVect):

    feature_maps = []

    poolOut = []

    feature_maps.append(np.array(conv(img, kernelMatrix[0], padVect[0], strideVect[0], actfnVect[0]))) 
   
    poolOut.append(np.array(pooling(feature_maps[0],2,2,poolfnVect[0])))
    
    for i in range(0,NoOfConv-1):
        
        feature_maps.append(np.array(conv(poolOut[i], kernelMatrix[i+1], padVect[i+1], strideVect[i+1], actfnVect[i+1]))) 
        poolOut.append(np.array(pooling(feature_maps[i+1],2,2,poolfnVect[i+1])))

    return feature_maps,poolOut

##########
#display the output of the layers

def display(feature_maps,poolOut,NoOfConv,kernelMatrix):
    
    print('Displaying output per channel of a given volume')

    for i in range(0,NoOfConv):
        print('LAYER: '+str(i+1))
        print('Kenels per layer shape'+str((kernelMatrix[i]).shape))
        print('Convolution output'+str((feature_maps[i]).shape))
        print('Pooling output'+str((poolOut[i]).shape))
        for j in range(0,feature_maps[i].shape[2]):
            plt.title('Layer:'+str(i+1)+' Activation_maps[:,:,'+str(j)+']')
            plt.imshow((feature_maps[i])[:,:,j],cmap="gray")
            plt.show()
            plt.title('After pooling PoolOut[:,:,'+str(j)+']')
            plt.imshow((poolOut[i])[:,:,j],cmap="gray")
            plt.show()





##########
#unraveling function
def unraveling(inp,L):

    M=(inp.shape[0])*(inp.shape[1])*(inp.shape[2])

    WMatrix=1.0/M*np.random.normal(0,1,(L,M))

    flatten = np.array(inp.flatten())

    return np.array(np.matmul(WMatrix,flatten))


#########
#MLP

# initialisation of the weights
def weights(noHiddenLayers,sizeOfLayers):

    W=[]
    b=[]

    for i in range(0,noHiddenLayers+1):
        W.append(np.random.normal(1e-4,1,(sizeOfLayers[i],sizeOfLayers[i+1])))
        b.append(np.random.normal(1e-4,1,(sizeOfLayers[i+1],1)))

    W=np.array(W)
    b=np.array(b)
    return W,b

#layer
def layer(w,x,b):
    out = np.dot(x,w)+b.T
    return out

def apply_activationMLP(Activation_function,inp):
    
    #activation functions
    if Activation_function == "relu":
        return np.where(inp<0,0,inp)
    elif Activation_function == "tanh":
        return np.tanh(inp)
    elif Activation_function == "sigmoid":
        return 1.0/(1+np.exp(-1.0*inp))
    elif Activation_function == "softmax":
        return (1.0/(np.sum(np.exp(inp),axis=1)))*(np.exp(inp))

#forward path
def forward_path(noHiddenLayers,sizeOfLayers,X,W,b,Activation_function='relu'):

    out=[]

    z=apply_activationMLP(Activation_function,np.array(layer(W[0],X,b[0])))
    out.append(np.array(z))

    for i in range(1,noHiddenLayers):
        z=apply_activationMLP(Activation_function,np.array(layer(W[i],out[i-1],b[i])))
        out.append(np.array(z))

    z=apply_activationMLP('softmax',np.array(layer(W[-1],out[-1],b[-1])))
    out.append(np.array(z))

    y_pred = out[-1]

    return out,y_pred

########################################################################
#initialise all parameters fo CNN

def init(img,NoOfConv):

    kernelMatrix=[] 
    strideVect  =[]
    padVect     =[]
    actfnVect   =[]
    poolfnVect  =[]
    temp2        =img.shape[2]

    for i in range(0,NoOfConv):
        
        temp1=np.random.randint(1,5)+2#some random number
        #a random number being generated that random number is the number of filters in the convolutional layer
        #you may change it to what ever you need
        
        kernelMatrix.append(np.random.normal(1e-4,1,(temp1,3, 3 ,temp2)))
        temp2=temp1#to match the depth of activation map after convolution layer and kernel depth

        strideVect.append((1,1))
        
        padVect.append('valid')

        actfnVect.append('relu')       
        
        poolfnVect.append('max')

    return kernelMatrix,strideVect,padVect,actfnVect,poolfnVect


################################

img = (np.array(Image.open('check.jpg')))

plt.imshow(img)
plt.show()

img=(img-np.mean(img))/np.std(img)
img=img.reshape([img.shape[0],img.shape[1],3])

NoOfConv = 2

kernelMatrix,strideVect,padVect,actfnVect,poolfnVect=init(img,NoOfConv)

###############################
# CNN
feature_maps,poolOut = compConv(img,NoOfConv,kernelMatrix,strideVect,padVect,actfnVect,poolfnVect)

display(feature_maps,poolOut,NoOfConv,kernelMatrix)

################################
# MLP

L  = 7
X=unraveling(feature_maps[-1],L)

noHiddenLayers=3

#also includes the input vector dimension and output vector dimension
sizeOfLayers=[L,6,5,7,3]

weights,bias=weights(noHiddenLayers,sizeOfLayers)

out,y_pred=forward_path(noHiddenLayers,sizeOfLayers,X,weights,bias,Activation_function='relu')

print ('output probability'+str(y_pred))
