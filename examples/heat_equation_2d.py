from pinn_generator import NeuralNetwork, PINN, PINNFactory
import torch
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Problem definition
alpha = 0.1  # Thermal difusivity

independent_vars = ['t', 'x', 'y']
dependent_vars = ['u']

pde_residuals_str = [
    f"Derivative(u(t, x, y), t) - {alpha} * (Derivative(u(t, x, y), (x, 2)) + Derivative(u(t, x, y), (y, 2)))"
]

conditions_def = [
    {'equation': 'u(t, x, y) - sin(pi*x)*sin(pi*y)'},
    {'equation': 'u(t, x, y) - 0'}, {'equation': 'u(t, x, y) - 0'},
    {'equation': 'u(t, x, y) - 0'}, {'equation': 'u(t, x, y) - 0'},
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

N_col = 20000
t_col = torch.rand(N_col, 1, device=device)
x_col = torch.rand(N_col, 1, device=device)
y_col = torch.rand(N_col, 1, device=device)

N_ic = 2500
t_ic = torch.zeros(N_ic, 1, device=device)
x_ic = torch.rand(N_ic, 1, device=device)
y_ic = torch.rand(N_ic, 1, device=device)

N_bc = 2500
t_bc_x = torch.rand(N_bc, 1, device=device) # t in x=0 and x=1
y_bc_x = torch.rand(N_bc, 1, device=device) # y in x=0 and x=1

t_bc_y = torch.rand(N_bc, 1, device=device) # t in y=0 and y=1
x_bc_y = torch.rand(N_bc, 1, device=device) # x in y=0 and y=1

data = {
    'collocation': [t_col, x_col, y_col],
    'conditions': [
        [t_ic, x_ic, y_ic],                                # IC u(0,x,y)
        [t_bc_x, torch.zeros_like(y_bc_x), y_bc_x],       # BC u(t,0,y)
        [t_bc_x, torch.ones_like(y_bc_x), y_bc_x],        # BC u(t,1,y)
        [t_bc_y, x_bc_y, torch.zeros_like(x_bc_y)],       # BC u(t,x,0)
        [t_bc_y, x_bc_y, torch.ones_like(x_bc_y)],        # BC u(t,x,1)
    ]
}

net = NeuralNetwork(num_inputs=3, num_outputs=1, num_layers=4, num_neurons=64)
model = PINN(net).to(device)

loss_weights = {'pde': 1.0, 'conditions': 10.0}

factory = PINNFactory(pde_residuals_str, conditions_def, independent_vars, dependent_vars, loss_weights=loss_weights)
loss_fn = factory.generate_loss_function()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 15000
start_time = time.time()
for epoch in range(epochs):
    optimizer.zero_grad()
    loss, loss_components = loss_fn(model, data)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {loss_components['total']:.4e}, "
                f"PDE: {loss_components.get('pde', 0):.4e}, Conds: {loss_components.get('conditions', 0):.4e}")

print(f"Training finished in {time.time() - start_time:.2f} seconds.")

model.eval()
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
fig.suptitle('2D Heat Equation: PINN vs. Analytical Solution', fontsize=16)

grid_res = 50
x = np.linspace(0, 1, grid_res)
y = np.linspace(0, 1, grid_res)
X, Y = np.meshgrid(x, y)
x_flat, y_flat = X.flatten(), Y.flatten()

x_torch = torch.tensor(x_flat, dtype=torch.float32).view(-1, 1).to(device)
y_torch = torch.tensor(y_flat, dtype=torch.float32).view(-1, 1).to(device)

def update_plot(frame_t):
    print(f"Generating frame for t = {frame_t:.2f}...")
    for ax in axes: ax.cla()
    u_true = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-2 * alpha * (np.pi**2) * frame_t)
    t_torch = torch.full_like(x_torch, frame_t)
    with torch.no_grad():
        u_pred = model(t_torch, x_torch, y_torch).cpu().numpy().reshape(X.shape)
    error = np.abs(u_true - u_pred)
    
    im1 = axes[0].imshow(u_true, extent=[0,1,0,1], origin='lower', vmin=0, vmax=1, cmap='inferno')
    axes[0].set_title(f'Analytical at t={frame_t:.2f}')
    im2 = axes[1].imshow(u_pred, extent=[0,1,0,1], origin='lower', vmin=0, vmax=1, cmap='inferno')
    axes[1].set_title(f'PINN at t={frame_t:.2f}')
    im3 = axes[2].imshow(error, extent=[0,1,0,1], origin='lower', cmap='Reds')
    axes[2].set_title(f'Absolute Error at t={frame_t:.2f}')
    for ax in axes: ax.set_xlabel('x'); ax.set_ylabel('y')
    return axes

cax1 = fig.add_axes([axes[0].get_position().x1 + 0.01, axes[0].get_position().y0, 0.02, axes[0].get_position().height])
fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap='inferno'), cax=cax1)
cax2 = fig.add_axes([axes[1].get_position().x1 + 0.01, axes[1].get_position().y0, 0.02, axes[1].get_position().height])
fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap='inferno'), cax=cax2)
cax3 = fig.add_axes([axes[2].get_position().x1 + 0.01, axes[2].get_position().y0, 0.02, axes[2].get_position().height])
fig.colorbar(plt.cm.ScalarMappable(cmap='Reds'), cax=cax3)

time_points = np.linspace(0, 1, 51)
ani = animation.FuncAnimation(fig, update_plot, frames=time_points, blit=False, interval=100)

try:
    print("Saving animation to 'heat_equation_2d.gif'...")
    ani.save('examples/heat_equation_2d.gif', writer='pillow', fps=10)
    print("Done.")
except Exception as e:
    print(f"\nCould not save animation. Error: {e}")
    print("Displaying animation instead.")
    plt.show()