#####Note:run in python2######


import math
import numpy as np
import sys
from PIL import Image
import matplotlib.pyplot as plt 

np.random.seed(42)

#pooling funcion as a volume 
def pooling(feature_map,size=2,stride=2,pool_function='max'):

    #Preparing the output of the pooling operation.
    pool_out = np.zeros(((feature_map.shape[0]-size)/stride+1,
                            (feature_map.shape[1]-size)/stride+1,feature_map.shape[-1]))

    r2=0
    #variables r2 and c2 are for keeping a track of rows and columns in final pool_out 
    for row in np.arange(0,feature_map.shape[0]-size, stride):
        c2 = 0
        for col in np.arange(0, feature_map.shape[1]-size, stride):
            if pool_function == 'max':#takes the max per chanel in the volume and stores them as a volume
                pool_out[r2, c2, :] = np.amax(feature_map[row:row+size,  col:col+size,:],(0,1)).reshape((1,1,feature_map.shape[-1]))
                #this function returns  the maximum of the elements depthwise
            if pool_function == 'average':#takes the average per chanel in the volume and stores them as a volume
                pool_out[r2, c2, :] = np.mean(feature_map[row:row+size,  col:col+size,:],(0,1)).reshape((1,1,feature_map.shape[-1]))
                #this function returns  the average of the elements depthwise
            c2 = c2 + 1
        r2 = r2 +1

    return pool_out

##########
#display the output of the layers

def display(feature_maps,poolOut):
    
    print('Displaying output per channel of a given volume')

    for i in range(0,feature_maps.shape[-1]):

            plt.title('Activation map feature_maps[:,:,'+str(i)+']')
            plt.imshow(feature_maps[:,:,i],cmap="gray")
            plt.show()
            plt.title('Pooling output [:,:,'+str(i)+']')
            plt.imshow(poolOut[:,:,i],cmap="gray")
            plt.show()


########################################################################

#CHANGE THE PRAMETERS HERE FOR DIFFERENT OUTPUT REQUIRED

##I have previously run the algorithm of convolution and stored the output of convolution with 1 filter in a image file
featureMap = (np.array(Image.open('activationVolume.jpg')))/255.0
#made the image between 0,1 by dividing by 255.0


featureMap = featureMap.reshape((featureMap.shape[0],featureMap.shape[1],featureMap.shape[-1]))
#here the featureMap.shape[-1] is the depth of the activation layer 

poolOut=pooling(featureMap,pool_function='max')

print('Activation Map Shape'+str(featureMap.shape))
print('Pooling output Shape'+str(poolOut.shape))

display(featureMap,poolOut)
