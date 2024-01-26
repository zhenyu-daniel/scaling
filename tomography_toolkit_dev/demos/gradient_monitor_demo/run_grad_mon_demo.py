import torch
from torch import nn
from torch.utils.data import DataLoader
from tomography_toolkit_dev.utils.data_science_mon.torch_gradient_monitor import Weight_Gradient_Monitor
import numpy as np
import matplotlib.pyplot as plt

"""
Demo to run the gradient monitor tool. Here, we try to solve a simple classification problem, via two classification neural networks.
One network uses dropout while the other one does not. The gradient monitor tool allows us to compare the gradient flow as a function of the training epoch.

The data set used here consists of two classes (signal and background), each characterized by two (normal distributed) features.

All results produced by this script are stored as .png files.
"""

print(" ")
print("**************************")
print("*                        *")
print("*   Gradient Flow Demo   *")
print("*                        *")    
print("**************************")   
print(" ") 

# Define the classification model, that we wish to inspect

class cl_model(nn.Module):
    
    def __init__(self,dropout_rate):
        super(cl_model, self).__init__()
        self.dropout_rate = dropout_rate
        
        self.cl_layers = nn.Sequential(
            nn.Linear(2,5),
            nn.Dropout(p=self.dropout_rate),
            nn.ReLU(),
            nn.Linear(5,3),
            nn.Dropout(p=self.dropout_rate),
            nn.ReLU(),
            nn.Linear(3,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.cl_layers(x)

# Basic settings:
#///////////////////////////////////////////////////
n_events = 1000000 #--> Number of events in each data set
n_epochs = 200 #--> Number of training epochs
mon_epoch = 20 #--> Print out some basic results, every mon_epoch
batch_dim = 256 #--> Batch size
#///////////////////////////////////////////////////

# Set up the models:
#*******************************************************
print("Set up classification models...")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_1 = cl_model(0.0).to(device)
model_2 = cl_model(0.05).to(device)

loss_fn = nn.BCELoss()
opt1 = torch.optim.Adam(model_1.parameters(),lr=0.01)
opt2 = torch.optim.Adam(model_2.parameters(),lr=0.01)

print("...done!")
print(" ")
#*******************************************************


# Load gradient monitor tool:
#*******************************************************
print("Load gradient monitor...")

grad_mon_1 = Weight_Gradient_Monitor(model=model_1)
grad_mon_2= Weight_Gradient_Monitor(model=model_2)

print("...done!")
print(" ")
#*******************************************************

# Set and plot data set
#*******************************************************
print("Set up and visualize the training data set...")

# Signal class:
sig_feat_1 = torch.normal(mean=1.0,std=0.7,size=(n_events,1))
sig_feat_2 = torch.normal(mean=2.0,std=0.8,size=(n_events,1))

sig_data = torch.cat([sig_feat_1,sig_feat_2],dim=1).to(device)
sig_labels = torch.ones(size=(sig_data.size()[0],1)).to(device)

# Background class:
bkg_feat_1 = torch.normal(mean=-1.0,std=0.7,size=(n_events,1))
bkg_feat_2 = torch.normal(mean=-2.0,std=0.8,size=(n_events,1))

bkg_data = torch.cat([bkg_feat_1,bkg_feat_2],dim=1).to(device)
bkg_labels = torch.zeros(size=(bkg_data.size()[0],1)).to(device)

train_dataloader = DataLoader([[sig_data,sig_labels],[bkg_data,bkg_labels]], batch_size=64,shuffle=True)

plt.rcParams.update({'font.size':20})

fig,ax = plt.subplots(1,2,figsize=(15,8),sharey=True)
fig.suptitle('Classification Data Set')

ax[0].hist(sig_data[:,0].cpu().numpy(),100,alpha=0.5,label='Signal')
ax[0].hist(bkg_data[:,0].cpu().numpy(),100,alpha=0.5,label='Background')
ax[0].legend(fontsize=15)
ax[0].grid(True)
ax[0].set_xlabel('Feature Variable 1')
ax[0].set_ylabel('Entries')

ax[1].hist(sig_data[:,1].cpu().numpy(),100,alpha=0.5,label='Signal')
ax[1].hist(bkg_data[:,1].cpu().numpy(),100,alpha=0.5,label='Background')
ax[1].legend(fontsize=15)
ax[1].grid(True)
ax[1].set_xlabel('Feature Variable 2')

fig.savefig('classification_data.png')
plt.close(fig)

print("...done!")
print(" ")
#*******************************************************

# Train the models
#*******************************************************
print("Run training...")

loss1_mon = []
loss2_mon = []
#+++++++++++++++++++++++++++
for epoch in range(1,1+n_epochs):

    if epoch % mon_epoch == 0:
       print(" ")
       print("   Epoch: " + str(epoch) + "/" + str(n_epochs))

    #++++++++++++++++++++++++++++
    for _, (x_train,y_train) in enumerate(train_dataloader):
        n_events = x_train.size()[0]

        y_pred_1 = model_1(x_train)
        loss_1 = loss_fn(y_pred_1,y_train)

        opt1.zero_grad()
        loss_1.backward()
        opt1.step() 

        # Notes: 
        # (i) We need to call the gradient watch function AFTER we called the '.backward()' command
        # (ii) There is no difference if we call the watch function before or after the 'optimizer.step()' command 
        grad_mon_1.watch_gradients_per_batch(sample_size=n_events)

        y_pred_2 = model_2(x_train)
        loss_2 = loss_fn(y_pred_2,y_train)

        opt2.zero_grad()
        loss_2.backward()
        opt2.step() 

        grad_mon_2.watch_gradients_per_batch(sample_size=n_events)

        
    #++++++++++++++++++++++++++++
    current_loss_1 = loss_1.detach().cpu().numpy()
    loss1_mon.append(current_loss_1)

    current_loss_2 = loss_2.detach().cpu().numpy()
    loss2_mon.append(current_loss_2)

    # Print out some information every mon_epoch:
    #-----------------------------------------------
    if epoch % mon_epoch == 0:
       print("   >>> Model 1 <<<")
       print("   >>> Loss: " + str(current_loss_1) + " <<<")
       print("   >>> Avg. gradients: " + str(grad_mon_1.average_gradient_watched.compute()) + " <<<")
       print("   >>> Min. gradients: " + str(grad_mon_1.minimum_gradient_watched.compute()) + " <<<")
       print("   >>> Max. gradients: " + str(grad_mon_1.maximum_gradient_watched.compute()) + " <<<")
       print(" ")

       print("   >>> Model 2 <<<")
       print("   >>> Loss: " + str(current_loss_2) + " <<<")
       print("   >>> Avg. gradients: " + str(grad_mon_2.average_gradient_watched.compute()) + " <<<")
       print("   >>> Min. gradients: " + str(grad_mon_2.minimum_gradient_watched.compute()) + " <<<")
       print("   >>> Max. gradients: " + str(grad_mon_2.maximum_gradient_watched.compute()) + " <<<")
    #-----------------------------------------------

    grad_mon_1.collect_gradients_per_epoch()
    grad_mon_2.collect_gradients_per_epoch()
#+++++++++++++++++++++++++++

print(" ")
print("...done!")
print(" ")
#*******************************************************

# Get the gradients:
#*******************************************************
print("Retrieve gradients for each model...")

model_1_gradients = grad_mon_1.read_out_gradients()
model_2_gradients = grad_mon_2.read_out_gradients()

print("...done!")
print(" ")
#*******************************************************

# Visualize everything
#*******************************************************
print("Plot results...")

sig_pred_1 = None
bkg_pred_1 = None

sig_pred_2 = None
bkg_pred_2 = None

with torch.no_grad():  
   sig_pred_1 = model_1(sig_data)
   bkg_pred_1 = model_1(bkg_data)

   sig_pred_2 = model_2(sig_data)
   bkg_pred_2 = model_2(bkg_data)


figt,axt = plt.subplots(1,3,figsize=(18,8))
figt.suptitle('Model Performance')
figt.subplots_adjust(wspace=0.35)

axt[0].plot(loss1_mon,'k-',linewidth=3.0,label='Model 1')
axt[0].plot(loss2_mon,'r--',linewidth=3.0,label='Model 2')
axt[0].grid(True)
axt[0].legend(fontsize=15)
axt[0].set_xlabel('Epoch')
axt[0].set_ylabel('Training Loss')

axt[1].hist(sig_pred_1.cpu().numpy(),100,alpha=0.5,facecolor='g',log=True,label='Signal')
axt[1].hist(bkg_pred_1.cpu().numpy(),100,alpha=0.5,facecolor='r',log=True,label='Background')
axt[1].legend(fontsize=15)
axt[1].set_xlabel('Model 1 Output')
axt[1].set_ylabel('Entries')

axt[2].hist(sig_pred_2.cpu().numpy(),100,alpha=0.5,facecolor='g',log=True,label='Signal')
axt[2].hist(bkg_pred_2.cpu().numpy(),100,alpha=0.5,facecolor='r',log=True,label='Background')
axt[2].legend(fontsize=15)
axt[2].set_xlabel('Model 2 Output')
axt[2].set_ylabel('Entries')

figt.savefig('model_performance.png')
plt.close(figt)

# The gradient plots are stored within a dictionary. The first element is a pyplot figure, whereas a second element is a pyplot axis (in case one needs to alter the plot settings)
grad1_dict = grad_mon_1.show_gradients(gradient_dict=model_1_gradients,model_name='Model1')
grad2_dict = grad_mon_2.show_gradients(gradient_dict=model_2_gradients,model_name='Model2')

fig_grad1 = grad1_dict['gradient_flow_Model1'][0]
fig_grad2 = grad2_dict['gradient_flow_Model2'][0]

fig_grad1.savefig('gradient_flow_model1.png')
plt.close(fig_grad1)

fig_grad2.savefig('gradient_flow_model2.png')
plt.close(fig_grad2)

print("...done!")
print(" ")
#*******************************************************



