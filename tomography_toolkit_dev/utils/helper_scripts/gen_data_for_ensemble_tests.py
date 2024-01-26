import numpy as np
import os


# Some settings:
n_data_samples = 1
sample_sizes = [1000,10000,100000]
sample_names = ['1k','10k','100k']
data_dir = "ensemble_data"

if os.path.exists(data_dir) == False:
    os.mkdir(data_dir)

# Load the proxy app data sample:
sample_data_package = np.load('../../sample_data/events_data.pkl.npy',allow_pickle=True)

# Get the norms out of the way:
norm1 = sample_data_package[1]
norm2 = sample_data_package[2]

# Get the data:
data = sample_data_package[0]

def get_sample(X,n_events):
    size_x = X.shape[1]
    do_replacing = False

    if n_events >= size_x:
        do_replacing = True

    idx = np.random.choice(size_x,size=(n_events,),replace=do_replacing)
    return X[:,idx]


#+++++++++++++++++++++++++
for s in range(1,1+n_data_samples):
    
    #+++++++++++++++++++++++++
    for n in range(len(sample_sizes)):
        current_sample = get_sample(data,sample_sizes[n])
        current_data_package = np.array((current_sample,norm1,norm2),dtype=object)

        save_name = data_dir + '/data_sample' + str(s) + '_' + sample_names[n] + '.npy'
        np.save(save_name,current_data_package)
    #+++++++++++++++++++++++++

#+++++++++++++++++++++++++


