import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import cusignal
import cupy as cp
from tqdm.notebook import tqdm
import cnn_data_set

class input_data(np.ndarray):
    '''
    class for input data
    '''
    pass

    def __new__(self,data):
        '''
        Load data from on experiment
        '''
        res = np.array(data)
        new = res.view(self)
        return new

    def normalized_data(self,scaler=None):
        '''
        Normalized the data set
        '''
        if scaler==None:
            scaler = StandardScaler()
            n_data = scaler.fit_transform(self)
        else:
            n_data=scaler.transform(self)


        n_data = input_data(np.clip(n_data, -5, 5))
        
        return n_data,scaler
    
    def create_sub_image(self,dim,column_BI=0):
        '''split data set of one image in data set of sub image'''
        im_class=self[:,:,column_BI]
        dataIA=np.array(self)
        dataIA=np.delete(dataIA,column_BI,axis=2)


        ss=np.shape(dataIA)
        for i in tqdm(range(ss[-1])):
            tmp=split_mat(dataIA[:,:,i],dim)
            if i==0:
                st=np.shape(tmp)
                res=np.zeros([st[0],st[1],ss[-1],st[2]])
            res[:,:,i,:]=tmp

        kernel=np.zeros([dim,dim])
        kernel[int((dim-1)/2),int((dim-1)/2)]=1
        BiClass=cp.asnumpy(cusignal.convolve2d(im_class,kernel,mode='valid').flatten())
        
        input_list=[]
        ss=np.shape(res)
        for i in range(ss[2]):
            input_list.append(np.transpose(res[:,:,i,:]))
        
        
        return cnn_data_set.cnn_data_set(input_list),BiClass


#-------------------------------------
#----------- function ----------------
#-------------------------------------
def split_mat(mat,dim):
    '''
    Split one image in sub-image of dimension dim x dim
    '''    
    for i in range(dim):
        for j in range(dim):
            kernel=np.zeros([dim,dim])
            kernel[i,j]=1
            tmp=cusignal.convolve2d(mat,kernel,mode='valid').flatten()
            if i+j==0:
                nb_img=len(tmp)
                res=np.zeros([int(dim**2),nb_img])
            
            res[i*dim+j,:]=cp.asnumpy(tmp)
            
    return res.reshape([dim,dim,nb_img])[::-1,::-1,:]