import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import cusignal
import cupy as cp
import random

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
        ss=self.shape
        fdata=np.zeros([ss[0]*ss[1],ss[2]])
        for i in range(ss[2]):
            fdata[:,i]=self[:,:,i].flatten()
            
        if scaler==None:
            scaler = StandardScaler()
            n_data = scaler.fit_transform(fdata)
        else:
            n_data=scaler.transform(fdata)


        n_data = input_data(np.clip(n_data, -5, 5))
        
        res_data=n_data.reshape(ss)
        
        return res_data,scaler
    
    def create_sub_image(self,dim,column_BI=0):
        '''split data set of one image in data set of sub image'''
        im_class=self[:,:,column_BI]
        dataIA=np.array(self)
        dataIA=np.delete(dataIA,column_BI,axis=2)


        ss=np.shape(dataIA)
        for i in range(ss[-1]):
            tmp=split_mat(dataIA[:,:,i],dim)
            if i==0:
                st=np.shape(tmp)
                res=np.zeros([st[0],st[1],st[2],ss[-1]])
            res[:,:,:,i]=tmp

        kernel=np.zeros([dim,dim])
        kernel[int((dim-1)/2),int((dim-1)/2)]=1
        BiClass=cp.asnumpy(cusignal.convolve2d(im_class,kernel,mode='valid').flatten())
        
        
        return input_data(res),BiClass
    
    def split_dataset(self,im_class,rpc=0.2):
        nb_input=self.shape[0]
        test_id=np.int32(np.array(random.sample(range(0,nb_input-1), int(rpc*nb_input))))
        allid=np.int32(np.linspace(0,nb_input-1,nb_input))

        train_id = np.int32(np.setdiff1d(allid,test_id))
        
        return self[train_id,:,:,:],im_class[train_id],self[test_id,:,:,:],im_class[test_id]
    
    def merge(self,self2):
        
        return input_data(np.concatenate([self,self2],axis=0))


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
                res=np.zeros([nb_img,int(dim**2)])
            
            res[:,i*dim+j]=cp.asnumpy(tmp)
            
    return res.reshape([nb_img,dim,dim])[:,::-1,::-1]
