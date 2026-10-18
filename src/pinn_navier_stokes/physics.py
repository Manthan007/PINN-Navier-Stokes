import torch
from pinn_navier_stokes.netwok import Model
import scipy.io

# Data function
def GetData(data_path: str):
    # Fetching the data
    mat_data = scipy.io.loadmat(data_path)
    print("Variables in file:", mat_data.keys())

    # Extracting matrices as numpy arrays
    X_star = mat_data['X_star']
    t_data = mat_data['t']
    u_star = mat_data["U_star"]

    # Convert to PyTorch Tensors
    x_tensor = torch.from_numpy(X_star).float()
    t_tensor = torch.from_numpy(t_data).float()
    u_tensor = torch.from_numpy(u_star).float()

    # Print shapes to inspect dimensions
    print("X_star Shape:", x_tensor.shape)
    print("t Shape:", t_tensor.shape)
    print("u_star Shape: ", u_star.shape)

    N_spatial = x_tensor.shape[0]
    N_time = t_tensor.shape[0]

    X_repeated = x_tensor.repeat(N_time, 1)
    t_repeated = torch.repeat_interleave(t_tensor, N_spatial, dim=0)

    X_Y_T_tensor = torch.cat([X_repeated, t_repeated], dim=1)

    # 5. Verify the results
    print("Final Tensor Shape:", X_Y_T_tensor.shape)
    print("First row (x, y, t_0):", X_Y_T_tensor[0])
    print("Last row (x, y, t_last):", X_Y_T_tensor[-1])

    return X_Y_T_tensor


tensor = GetData("D:\\PINN-Navier-Stokes\\data\\cylinder_nektar_wake.mat")

x = tensor[:, 0:1].requires_grad_(True)
y = tensor[:, 1:2].requires_grad_(True)
t = tensor[:, 2:3].requires_grad_(True)

inputs = torch.cat([x, y, t], dim=1)

model = Model(input_dim=3, output_dim=2)
psi, p = model(inputs)


def derivative(output, input):
    return torch.autograd.grad(
            outputs=output, 
            inputs=input, 
            grad_outputs=torch.ones_like(output), 
            create_graph=True
        )[0]

def get_gradient():

    u = derivative(psi, y)
    
    v = -derivative(psi, x)

    return u, v

# Calculate velocities
u, v = get_gradient()

print("psi shape:", psi.shape)
print("p shape:", p.shape)
print("u shape:", u.shape)
print("v shape:", v.shape)

