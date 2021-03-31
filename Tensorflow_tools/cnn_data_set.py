import numpy as np
import random

class cnn_data_set(list):
    '''
    class for input data
    '''
    pass

    def split_dataset(self,im_class,rpc=0.2):
        nb_input=len(self[0])
        test_id=np.int32(np.array(random.sample(range(0,nb_input-1), int(rpc*nb_input))))
        allid=np.int32(np.linspace(0,nb_input-1,nb_input))

        train_id = np.int32(np.setdiff1d(allid,test_id))
        train=[]
        test=[]
        for i in range(len(self)):
            train.append(self[i][train_id,:,:])
            test.append(self[i][test_id,:,:])
        
        return train,im_class[train_id],test,im_class[test_id]