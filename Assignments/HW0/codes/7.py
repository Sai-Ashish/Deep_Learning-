#####Note:run in python2######

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(11)

# initialisation of the weights
def weights(X,noHiddenLayers,sizeOfHiddenLayers,sizeofOutput):

	W=[]
	b=[]
	W.append(np.random.normal(1e-4,1,(X.shape[1],sizeOfHiddenLayers[0])))
	b.append(np.random.normal(1e-4,1,(sizeOfHiddenLayers[0],1)))
	
	for i in range(0,noHiddenLayers-1):
		W.append(np.random.normal(1e-4,1,(sizeOfHiddenLayers[i],sizeOfHiddenLayers[i+1])))
		b.append(np.random.normal(1e-4,1,(sizeOfHiddenLayers[i+1],1)))

	
	W.append(np.random.normal(1e-4,1,(sizeOfHiddenLayers[-1],sizeofOutput)))
	b.append(np.random.normal(1e-4,1,(sizeofOutput,1)))

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
def forward_path(noHiddenLayers,X,W,b,Actfnvect):

	out=[]

	z=apply_activationMLP(Actfnvect[0],np.array(layer(W[0],X,b[0])))
	out.append(np.array(z))

	for i in range(1,noHiddenLayers):
		z=apply_activationMLP(Actfnvect[i],np.array(layer(W[i],out[i-1],b[i])))
		out.append(np.array(z))

	z=apply_activationMLP(Actfnvect[-1],np.array(layer(W[-1],out[-1],b[-1])))
	out.append(np.array(z))

	y_pred = out[-1]

	return out,y_pred

#####################################################

###
#MLP PARAMETERS
noHiddenLayers=3

X=np.array([1,2,3,5]).reshape((1,4))
#also includes the input vector dimension and output vector dimension
sizeOfHiddenLayers=[4,3,4]

sizeofOutput=2

Actfnvect = ['relu','relu','relu','softmax']

###

# MLP FUNCTION
#with softmax
weights,bias=weights(X,noHiddenLayers,sizeOfHiddenLayers,sizeofOutput)

out,y_pred=forward_path(noHiddenLayers,X,weights,bias,Actfnvect)
print('Y out with softmax')
print y_pred
print np.sum(y_pred)


# MLP FUNCTION
#without softmax
Actfnvect = ['relu','relu','relu','relu']

out,y_pred=forward_path(noHiddenLayers,X,weights,bias,Actfnvect)
print('Y out without softmax and relu used')
print y_pred
print np.sum(y_pred)