#####Note:run in python2######


import math
import numpy as np
import sys
from PIL import Image
import matplotlib.pyplot as plt 
import cv2


np.random.seed(78)


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
    bias=np.random.normal(0,0.01,(feature_maps.shape[0],feature_maps.shape[1],feature_maps.shape[2]))
    # Convolving the image by the filter(s).
    for filter_num in range(conv_filter.shape[0]):

        curr_filter = conv_filter[filter_num, :]

        filter_sizeRow = curr_filter.shape[0]

        filter_sizeCol = curr_filter.shape[1]

        #Applying the convolution operation.
        for r in np.arange(filter_sizeRow/2,img.shape[0]-filter_sizeRow/2+1,stride[0]):
            for c in np.arange(filter_sizeCol/2,img.shape[1]-filter_sizeCol/2+1,stride[1]):
                #Region required
                curr_region = img[r-filter_sizeRow/2:r+filter_sizeRow/2+1,c-filter_sizeCol/2:c+filter_sizeCol/2+1,:]
                #Element-wise multipliplication between the current region and the filter.
                curr_result = curr_region * curr_filter
                #Summing the result of multiplication.
                feature_maps[(r-(filter_sizeRow/2))/stride[0],
                             (c-(filter_sizeCol/2))/stride[1], filter_num] =apply_activationCONV(Activation_function,np.sum(curr_result))+bias[(r-(filter_sizeRow/2))/stride[0],
                             (c-(filter_sizeCol/2))/stride[1], filter_num]  

    return feature_maps # Returning all feature maps.

##########
#display the output of the layers

def display(feature_maps,conv_filter):
    
    print('Displaying output per channel of a given volume')

    for i in range(0,feature_maps.shape[-1]):
            print('\nKernel filter('+str(i+1)+') shape'+str(conv_filter[i,:,:,:].shape)+'\n')
            print(conv_filter[i,:,:,:])
            plt.title('Activation map feature_maps[:,:,'+str(i)+']')
            plt.imshow(feature_maps[:,:,i],cmap="gray")
            plt.show()



########################################################################

#CHANGE THE PRAMETERS HERE FOR DIFFERENT OUTPUT REQUIRED

img = (np.array(Image.open('check.jpg')))

plt.title('Image')    
plt.imshow(img)
plt.show()

img=(img-np.mean(img))/np.std(img)
#made the image between 0,1 by dividing by 255.0

img=img.reshape([img.shape[0],img.shape[1],3])

#NOTE:
#conv_filter.shape[0] is the number of filters for a convolutional layer
#conv_filter.shape[1], conv_filter.shape[2] filter dimensions 
#conv_filter.shape[2] is the depth of the kernel=input depth

filter_num=3
ker_rows  =3
ker_cols  =3
ker_depth =img.shape[2]

#random weights for convolution filter kernels
conv_filter=np.random.normal(0,1,(filter_num,ker_rows,ker_cols,ker_depth))

#convolution parameters
stride=(1,1)
padding='same'
Activation_function='relu'
feature_maps=conv(img,conv_filter,padding,stride,Activation_function)

print( 'Image Shape'+str(img.shape))
print('Number of filters:' + str(filter_num))
print('feature_map Shape'+str(feature_maps.shape))


display(feature_maps,conv_filter)