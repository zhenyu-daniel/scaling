import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':20})

add_name = '_winit.npy'
#add_name = '.npy'

m1 = np.load('parameter_residuals_mean_5'+add_name)
s1 = np.load('parameter_residuals_std_5'+add_name)

m2 = np.load('parameter_residuals_mean_128'+add_name)
s2 = np.load('parameter_residuals_std_128'+add_name)

m3 = np.load('parameter_residuals_mean_500'+add_name)
s3 = np.load('parameter_residuals_std_500'+add_name)

m4 = np.load('parameter_residuals_mean_1000'+add_name)
s4 = np.load('parameter_residuals_std_1000'+add_name)


fig,ax = plt.subplots(6,1,sharex=True)


ensemble_id = np.array([i for i in range(m1.shape[0])])
#++++++++++++++++++++++++++++
for p in range(6):
    ax[p].errorbar(x=np.array(ensemble_id),y=m1[:,p],yerr=s1[:,p],linewidth=3.0,label='N=5')
    ax[p].errorbar(x=np.array(ensemble_id),y=m2[:,p],yerr=s2[:,p],linewidth=3.0,label='N=128')
    ax[p].errorbar(x=np.array(ensemble_id),y=m3[:,p],yerr=s3[:,p],linewidth=3.0,label='N=500')
    ax[p].errorbar(x=np.array(ensemble_id),y=m4[:,p],yerr=s4[:,p],linewidth=3.0,label='N=1000')
    ax[p].grid(True)
    ax[p].set_ylabel('Res. ' + str(p))
    ax[p].legend(fontsize=10)
#++++++++++++++++++++++++++++
ax[5].set_xlabel('Ensemble ID')


plt.show()

