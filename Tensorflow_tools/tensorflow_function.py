from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def cnn_build(filter,input_size,kernel_size=(3,3)):
    visible = keras.Input(shape=(input_size,input_size,1))
    for ifil in filter :
        cnn2d = keras.layers.Conv2D(filters=ifil,kernel_size=kernel_size,activation='relu')(visible)
        cnn2d = keras.layers.MaxPooling2D(pool_size=(2,2))(cnn2d)
    
    cnn2d = keras.layers.Flatten()(cnn2d)
    
    return cnn2d,visible

def plot_metrics(history,color='r'):
  metrics = ['loss', 'auc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color, label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color, linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()