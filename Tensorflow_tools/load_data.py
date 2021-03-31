import input_data

def load_data(adr,shape):
    data = pd.read_csv(adr,delimiter=' ')
    # Correct the misAngle data
    id=data['misAngle']>np.pi/2
    data['misAngle'][id]=np.pi-data['misAngle'][id]
    #select variable
    field=['RX','eqStrain','eqStress','Sys_pr','dist_to_GB','misAngle','Schmid_factor']
    cdata=input_data.input_data(data[field])
    
    return cdata
    