import torch
import scipy
import os
from pinn_navier_stokes.physics import GetData, derivative
from pinn_navier_stokes.netwok import Model


def NaiverStokesLoss(model, x, y, t):
    inputs = torch.cat([x, y, t], dim=1)
    psi, p = model(inputs)

    # velocities
    u = derivative(psi, y)
    v = -derivative(psi, x)

    # first order derivatives
    u_t = derivative(u, t)
    u_x = derivative(u, x)
    u_y = derivative(u, y)
    v_t = derivative(v, t)
    v_x = derivative(v, x)
    v_y = derivative(v, y)
    p_x = derivative(p, x)
    p_y = derivative(p, y)

    # second order derivaties
    u_x_x = derivative(u_x, x)
    u_y_y = derivative(u_y, y)
    v_x_x = derivative(v_x, x)
    v_y_y = derivative(v_y, y)

    vis = 0.01
    f = u_t + u*u_x + v*u_y + p_x - vis*(u_x_x + u_y_y)
    g = v_t + u*v_x + v*v_y + p_y - vis*(v_x_x + v_y_y)
    
    return torch.mean(f**2) + torch.mean(g**2)

# setting up the model and optimizer
model = Model(input_dim=3, output_dim=2)
adam_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lbfgs_optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=10000)

# fetching the data
full_tensor =  GetData("D:\\PINN-Navier-Stokes\\data\\cylinder_nektar_wake.mat")

# random sampling
N_train = 5000
idx = torch.randperm(full_tensor.shape[0])[:N_train]
tensor = full_tensor[idx]

# x, y, t
x = tensor[:, 0:1].requires_grad_(True)
y = tensor[:, 1:2].requires_grad_(True)
t = tensor[:, 2:3].requires_grad_(True)

# Fetching the collocation points
mat_data = scipy.io.loadmat("D:\\PINN-Navier-Stokes\\data\\cylinder_nektar_wake.mat")
u_star = mat_data["U_star"]
u_flat = torch.from_numpy(u_star[:, 0, :]).float().flatten().unsqueeze(1)
v_flat = torch.from_numpy(u_star[:, 1, :]).float().flatten().unsqueeze(1)
# True velocities
u_true = u_flat[idx]
v_true = v_flat[idx]


print("Starting Adam pre-training...")
epochs = 1500
for epoch in range(epochs):
    adam_optimizer.zero_grad()
    
    # 1. Forward passes & derivatives
    inputs = torch.cat([x, y, t], dim=1)
    psi, p = model(inputs)
    u_pred = derivative(psi, y)
    v_pred = -derivative(psi, x)
    
    # 2. Calculate Losses
    MSE_data = torch.mean((u_pred-u_true)**2) + torch.mean((v_pred-v_true)**2)
    MSE_physics = NaiverStokesLoss(model, x, y, t)
    total_loss = MSE_data + MSE_physics
    
    # 3. Backprop
    total_loss.backward()
    adam_optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Adam Epoch {epoch}: Loss {total_loss.item():.5f}")



def closure():
    lbfgs_optimizer.zero_grad()

    inputs = torch.cat([x, y, t], dim=1)
    psi, p = model(inputs)
    # pred velocities
    u_pred = derivative(psi, y)
    v_pred = -derivative(psi, x)

    # 1. MSE_data
    MSE_data = torch.mean((u_pred-u_true)**2) + torch.mean((v_pred-v_true)**2)
    # 2. MSE_physics
    MSE_physics = NaiverStokesLoss(model, x, y, t)
    # 3. total_loss
    total_loss = MSE_data + MSE_physics

    total_loss.backward()
    print(f"Loss: {total_loss.item():.5f} (Physics: {MSE_physics.item():.5f}, Data: {MSE_data.item():.5f})")
    
    return total_loss


# Run the optimizer
print("Starting L-BFGS training...")
lbfgs_optimizer.step(closure)

# 1. Ensure the 'model' directory exists
os.makedirs("model", exist_ok=True)

# 2. Define the file path
path = "model/pinn_cylinder_weights.pth"

# 3. Save the model's state dictionary (weights)
torch.save(model.state_dict(), path)

print(f"Model weights successfully saved to {path}")