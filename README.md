# PINN-Navier-Stokes

A PyTorch implementation of a Physics-Informed Neural Network (PINN) designed to solve the 2D incompressible Navier-Stokes equations. This project models fluid flow past a circular cylinder, capturing the formation of the classic von Kármán vortex street without relying on traditional computational fluid dynamics (CFD) mesh solvers.

By leveraging PyTorch's `autograd` engine, this network calculates continuous spatial and temporal derivatives to mathematically enforce the laws of conservation of mass and momentum directly within the loss function.

## Architecture & Tech Stack
* **Framework:** PyTorch
* **Network Design:** Fully Connected MLP (8 hidden layers, 20 neurons/layer) with `tanh` activations to preserve non-zero second-order derivatives.
* **Optimizers:** Dual-stage training pipeline.
  * **Adam:** Global exploration and initial convergence.
  * **L-BFGS:** Second-order fine-tuning for strict mathematical constraint satisfaction.
* **Data Processing:** SciPy and NumPy for handling `.mat` collocation and boundary data.
* **Visualization:** Matplotlib for rendering pressure and velocity contour fields.

## The Physics Loss
The neural network takes spatial and temporal coordinates $(x, y, t)$ as inputs and predicts the latent stream function $\psi(x, y, t)$ and pressure $p(x, y, t)$. From $\psi$, velocity components $u$ and $v$ are derived via automatic differentiation. 

The network is penalized by the Navier-Stokes residuals $f$ and $g$:
$$f = u_t + u u_x + v u_y + p_x - \nu (u_{xx} + u_{yy})$$
$$g = v_t + u v_x + v v_y + p_y - \nu (v_{xx} + v_{yy})$$

Total Loss is computed as:
$$Loss = MSE_{data} + \text{mean}(f^2) + \text{mean}(g^2)$$

## Repository Structure

```text
PINN-Navier-Stokes/
├── .venv/                          # Virtual environment
├── data/
│   └── cylinder_nektar_wake.mat    # Ground truth wake dataset
├── model/
│   └── pinn_cylinder_weights.pth   # Saved model weights
├── src/
│   └── pinn_navier_stokes/         # Main package source
│       ├── __init__.py
│       ├── evaluate.py             # Inference and visualization script
│       ├── netwok.py               # PyTorch nn.Module architecture
│       ├── physics.py              # Autograd derivatives and data loading
│       └── trainer.py              # Dual-stage Adam/L-BFGS training loop
├── .gitignore
├── .python-version
├── LICENSE
├── pyproject.toml                  # Project metadata and dependencies
├── README.md
├── results.png                     # Output contour plots
└── uv.lock                         # UV package manager lockfile
```

## Installation & Setup

This project uses `uv` for fast dependency management, but can also be installed via standard pip.

**Using uv (Recommended based on project structure):**
```bash
uv venv
uv sync
```

**Using standard pip:**
```bash
python -m venv .venv
# Activate environment (Windows)
.venv\Scripts\activate
# Install dependencies
pip install torch scipy numpy matplotlib
```

*Note: Ensure the `cylinder_nektar_wake.mat` dataset is placed in the `data/` directory before running.*

## Usage

**1. Train the Model**
Run the training script to execute the Adam and L-BFGS optimization loops. The script will randomly sample collocation points and save the final `.pth` weights to the `model/` directory upon completion.
```bash
python src/pinn_navier_stokes/trainer.py
```

**2. Evaluate and Visualize**
Once weights are saved, run the evaluation script to query the neural network across a dense space-time grid. This will generate a contour plot of the predicted pressure field ($p$) at a given time step (e.g., $t=10s$).
```bash
python src/pinn_navier_stokes/evaluate.py
```

## Results
A successfully trained model will output a continuous, mesh-free contour map displaying the alternating high and low-pressure stagnation zones and wake structures characteristic of fluid dynamics past a cylinder boundary (see `results.png`).