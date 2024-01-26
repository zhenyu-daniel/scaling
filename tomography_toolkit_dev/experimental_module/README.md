# Experimental Module(s) for the SciDAC Quantom Project
This folder contains different modules which shall mimic experimental conditions such as detector noise, resolution effects or reconstruction inefficiencies. Each module is listed and explained below.

## Torch_Simplified_Detector
* As the name suggests, this is a very basic (and not realistic) implementation of a detector which works as follows:
```
X_det = M * X_ideal
``` 
* where X_ideal is the undistorted event vector (i.e. after the theory module). The matrix M mimics detector effects and is defined as follows:
```
M = M_G + c1 * M_c + c2 * (M_r + M_r.T)/2
``` 
* with:
```
M_G: Diagonal matrix where each diagonal element i is ~N(1,sigma_i)
M_c: Symmetric matrix: M_c = 0, if i = j and M_c = 1 else
M_r: Matrix with random entries ~U(r1,r2) and 0 on the diagonal
```
* c1 and c2 are parameters to adjust the magnitude of the constant and variational correlations. r1 and r2 adjust the symmetry / asymmetry of the variational correlations. 
* The parameters: sigma_i, c1, c2, r1 and r2 are provided by the user  
* Input: Tensor with shape: N_events x N_features 
* Output: Tensor with shape: N_events x N_features
* If the input vector happens to have a different shape than the one specified above, the user may transpose the input, by setting the configuration file properly.
