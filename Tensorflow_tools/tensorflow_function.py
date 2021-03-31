from tensorflow import keras

def cnn_build(filter,input_size,kernel_size=(3,3)):
    visible = keras.Input(shape=(input_size,input_size,1))
    for ifil in filter :
        cnn2d = keras.layers.Conv2D(filters=ifil,kernel_size=kernel_size,activation='relu')(visible)
        cnn2d = keras.layers.MaxPooling2D(pool_size=(2,2))(cnn2d)
    
    cnn2d = keras.layers.Flatten()(cnn2d)
    
    return cnn2d,visible