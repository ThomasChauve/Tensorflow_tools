import Tensorflow_tools.input_data as mtfl
import numpy as np
import pandas as pd

def load_data(adr,shape):
    data = pd.read_csv(adr,delimiter=' ')
    # Correct the misAngle data
    id=data['misAngle']>np.pi/2
    data['misAngle'][id]=np.pi-data['misAngle'][id]
    #select variable
    field=['RX','eqStrain','eqStress','Sys_pr','dist_to_GB','misAngle','Schmid_factor']
    cdata=mtfl.input_data(np.array(data[field]).reshape([shape[0],shape[1],len(field)]))
    
    return cdata