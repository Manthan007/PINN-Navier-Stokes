from pinn_navier_stokes.netwok import Model
import torch
import matplotlib.pyplot as plt
import numpy as np

# Instantiate the model
model = Model(input_dim=3, output_dim=2)

# Loading the saved dict
model.load_state_dict(torch.load("D:\\PINN-Navier-Stokes\\model\\pinn_cylinder_weights.pth"))

# Ready to evaluate the model
model.eval()

# Generating the space time grid
x_grid = np.linspace(1.0, 8.0, 200)
y_grid = np.linspace(-2.0, 2.0, 200)
X, Y = np.meshgrid(x_grid, y_grid)

# flatten
x_flat = X.flatten()[:, None]
y_flat = Y.flatten()[:, None]
t = 10
t_flat = np.full_like(x_flat, t)

# Converting to tensors
x_flat = torch.from_numpy(x_flat).float()
y_flat = torch.from_numpy(y_flat).float()
t_flat = torch.from_numpy(t_flat).float()

inputs = torch.cat([x_flat, y_flat, t_flat], dim=1)
with torch.no_grad(): 
    psi, p = model(inputs)

# p back to numpy array
p_pred_numpy = p.detach().numpy().reshape(X.shape)


plt.figure(figsize=(10, 4))
# contourf fills the contours with color. 'jet' is the standard fluid dynamics colormap.
plt.contourf(X, Y, p_pred_numpy, levels=50, cmap='jet')
plt.colorbar(label='Pressure')
plt.title("PINN Prediction: Pressure Field at t=10s")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

