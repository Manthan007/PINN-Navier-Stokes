import torch
from pinn_navier_stokes.physics import GetData, derivative
from pinn_navier_stokes.netwok import Model


def NaiverStokesLoss(model, x, y, t):
    inputs = torch.cat([x, y, t], dim=1)

    model = Model(input_dim=3, output_dim=2)
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

def closure(model):
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=50000)


# fetching the data
full_tensor =  GetData("D:\\PINN-Navier-Stokes\\data\\cylinder_nektar_wake.mat")

# random sampling
N_train = 100000
idx = torch.randperm(full_tensor.shape[0])[:N_train]
tensor = full_tensor[idx]

# x, y, t
x = tensor[:, 0:1].requires_grad_(True)
y = tensor[:, 1:2].requires_grad_(True)
t = tensor[:, 2:3].requires_grad_(True)