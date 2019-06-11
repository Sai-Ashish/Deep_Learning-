#####Note:run in python2######


import math
import numpy as np
import sys
from PIL import Image
import matplotlib.pyplot as plt 

np.random.seed(42)



##########
#unraveling function
def unraveling(inp,L):

	M=(inp.shape[0])*(inp.shape[1])*(inp.shape[2])

	WMatrix=1.0/M*np.random.normal(0,1,(L,M))

	flatten = np.array(inp.flatten())

	return np.array(np.matmul(WMatrix,flatten))

#########

#i have stored the output from the pooling of a convolutional layer dimension(225,225,3) in activationVolume.jpg
# now i read it and perform the unraveling on it 

feature_maps = (np.array(Image.open('PoolingVolume.jpg')))/255.0

#unraveling function
L  = 7
X=unraveling(feature_maps,L)

print(str(feature_maps.shape)+'->'+str(X.shape))

print('Output vector :\n'+ str(X))