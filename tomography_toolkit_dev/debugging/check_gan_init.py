import torch 
import numpy as np
import tomography_toolkit_dev.generator_module as generators
import matplotlib.pyplot as plt


n_neurons = 128

generator_config={
    "num_layers": 4,
    "num_nodes": [n_neurons]*4,
    "activation": ["LeakyRelu", "LeakyRelu", "LeakyRelu", "LeakyRelu"],
    "dropout_percents": [0., 0., 0., 0.],
    "input_size" : 6,
    "output_size" : 6,
    "learning_rate": 1e-5,
   #  "weight_initialization": ["uniform"]*4,
   #  "bias_initialization": ["normal"]*4
}

dev = "cpu"
#torch.manual_seed(0)

mean_store_name = 'parameter_residuals_mean_' + str(n_neurons) + '_winit2_binit2.npy'
std_store_name = 'parameter_residuals_std_' + str(n_neurons) + '_winit2_binit2.npy'


true_params = torch.as_tensor([0.72916667, 0.25, 0.6, 0.36458333, 0.25, 0.8],device=dev)
parmin = torch.as_tensor([0.0, -1.0, 0.0, 0.0, -1.0, 0.0],device=dev)
parmax = torch.as_tensor([3.0, 1.0, 5.0, 3.0, 1.0, 5.0],device=dev)

ensemble_size = 50

parameter_mean = np.zeros((ensemble_size,6))
parameter_std = np.zeros((ensemble_size,6))

residual_mean = np.zeros((ensemble_size,6))
residual_std = np.zeros((ensemble_size,6))

ensemble_id = []
#++++++++++++++++++++++++++++++
for i in range(ensemble_size):
   generator = generators.make("torch_mlp_generator_hvd", config=generator_config, module_chain=[],devices=dev)

#    if i == ensemble_size-1:
#       print(generator) 

   noise = torch.normal(mean=0.0,std=1.0,size=(10000,6),device=dev)
   params = generator.generate(noise).to(dev)

   residuals = true_params - params

   res_mu =  np.mean(residuals.detach().cpu().numpy(),axis=0)
   res_std =  np.std(residuals.detach().cpu().numpy(),axis=0)

   mu =  np.mean(params.detach().cpu().numpy(),axis=0)
   std =  np.std(params.detach().cpu().numpy(),axis=0)
   #++++++++++++++++++++++++++++
   for p in range(6):
      parameter_mean[i][p] = mu[p]
      parameter_std[i][p] = std[p]

      residual_mean[i][p] = res_mu[p]
      residual_std[i][p] = res_std[p]
   #++++++++++++++++++++++++++++

   ensemble_id.append(i+1) 
#++++++++++++++++++++++++++++++

plt.rcParams.update({'font.size':20})

fig,ax = plt.subplots(6,1,sharex=True)
figr,axr = plt.subplots(6,1,sharex=True)



#++++++++++++++++++++++++++++
for p in range(6):
    ax[p].errorbar(x=np.array(ensemble_id),y=parameter_mean[:,p],yerr=parameter_std[:,p],linewidth=3.0)
    ax[p].grid(True)
    ax[p].set_ylabel('Param. ' + str(p))

    axr[p].errorbar(x=np.array(ensemble_id),y=residual_mean[:,p],yerr=residual_std[:,p],linewidth=3.0)
   # axr[p].set_ylim(-0.35,0.35)
    axr[p].grid(True)
    axr[p].set_ylabel('Res. ' + str(p))
#++++++++++++++++++++++++++++
ax[5].set_xlabel('Ensemble ID')
axr[5].set_xlabel('Ensemble ID')

plt.show()

# np.save(mean_store_name,residual_mean)
# np.save(std_store_name,residual_std)

