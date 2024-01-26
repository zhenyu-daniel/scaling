import numpy as np
import matplotlib.pyplot as plt

n_workers = [2,4,8,16]
worker_colors = ['k','r','m','g']
worker_lines = ['-','--','-.',':']
#core_name = 'gan_training_results'
core_name = 'gan_perf_ana'
loss_dir = 'data_science_npy'
phys_dir = "physics_npy"


plt.rcParams.update({'font.size':20})
fig,ax = plt.subplots(1,3,figsize=(17,8))
fig.subplots_adjust(wspace=0.5)

figp,axp = plt.subplots(1,2,figsize=(12,8))
figp.subplots_adjust(wspace=0.5)

#+++++++++++++++++
for i in range(len(n_workers)):
    add_name = str(n_workers[i])

    # if i > 1:
    #     add_name = '2_np'+str(n_workers[i])
    disc_real_loss = np.load(core_name+'_np'+add_name+'/'+loss_dir+'/discriminator_real_loss.npy')
    disc_fake_loss = np.load(core_name+'_np'+add_name+'/'+loss_dir+'/discriminator_fake_loss.npy')
    gen_loss = np.load(core_name+'_np'+add_name+'/'+loss_dir+'/generator_loss.npy')
    training_time = np.load(core_name+'_np'+add_name+'/'+loss_dir+'/training_time.npy')

    # pdf_x = np.load(core_name+'_np'+str(n_workers[i])+'/'+phys_dir+'/pdf_x.npy')
    # true_u = np.load(core_name+'_np'+str(n_workers[i])+'/'+phys_dir+'/true_u.npy')
    # true_d = np.load(core_name+'_np'+str(n_workers[i])+'/'+phys_dir+'/true_d.npy')

    # gen_u = np.load(core_name+'_np'+str(n_workers[i])+'/'+phys_dir+'/generated_u.npy')
    # gen_d = np.load(core_name+'_np'+str(n_workers[i])+'/'+phys_dir+'/generated_d.npy')
    

    current_label = 'N(GPUs) = ' + str(n_workers[i])
    n_show = 50

    ax[0].plot(training_time[:n_show],gen_loss[:n_show],color=worker_colors[i],linestyle=worker_lines[i],linewidth=2.0,label=current_label)
    ax[0].set_xlabel('Training Time [s]')
    ax[0].set_ylabel('Normalized Training Loss')
    ax[0].set_title('Generator')
    ax[0].grid(True)
    ax[0].legend(fontsize=15)
    ax[0].set_xscale('log')

    ax[1].plot(training_time[:n_show],disc_real_loss[:n_show],color=worker_colors[i],linestyle=worker_lines[i],linewidth=2.0,label=current_label)
    ax[1].set_xlabel('Training Time [s]')
    ax[1].set_ylabel('Normalized Training Loss')
    ax[1].set_title('Discriminator Real')
    ax[1].grid(True)
    ax[1].legend(fontsize=15)
    ax[1].set_xscale('log')

    ax[2].plot(training_time[:n_show],disc_fake_loss[:n_show],color=worker_colors[i],linestyle=worker_lines[i],linewidth=2.0,label=current_label)
    ax[2].set_xlabel('Training Time [s]')
    ax[2].set_ylabel('Normalized Training Loss')
    ax[2].set_title('Discriminator Fake')
    ax[2].grid(True)
    ax[2].legend(fontsize=15)
    ax[2].set_xscale('log')
#+++++++++++++++++


plt.show()
