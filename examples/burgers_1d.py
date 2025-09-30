from pinn_generator import NeuralNetwork, PINN, PINNFactory
import torch
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

nu = 0.01 / np.pi 

independent_vars = ['t', 'x']
dependent_vars = ['u']

pde_residuals_str = [
    f"Derivative(u(t, x), t) + u(t, x) * Derivative(u(t, x), x) - {nu} * Derivative(u(t, x), (x, 2))"
]

conditions_def = [
    {'equation': 'u(t, x) + sin(pi*x)'},  # IC: u(0,x) = -sin(pi*x)
    {'equation': 'u(t, x) - 0'},          # BC: u(t,-1) = 0
    {'equation': 'u(t, x) - 0'},          # BC: u(t,1) = 0
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

N_col = 20000
t_col = torch.rand(N_col, 1, device=device)
x_col = (torch.rand(N_col, 1, device=device) * 2) - 1 # x de -1 a 1

N_ic = 1000
t_ic = torch.zeros(N_ic, 1, device=device)
x_ic = (torch.rand(N_ic, 1, device=device) * 2) - 1

N_bc = 1000
t_bc = torch.rand(N_bc, 1, device=device)

data = {
    'collocation': [t_col, x_col],
    'conditions': [
        [t_ic, x_ic],
        [t_bc, -torch.ones_like(t_bc)], # BC x=-1
        [t_bc, torch.ones_like(t_bc)],  # BC x=1
    ]
}

net = NeuralNetwork(num_inputs=2, num_outputs=1, num_layers=6, num_neurons=64)
model = PINN(net).to(device)

loss_weights = {'pde': 1.0, 'conditions': 20.0}

factory = PINNFactory(pde_residuals_str, conditions_def, independent_vars, dependent_vars, loss_weights=loss_weights)
loss_fn = factory.generate_loss_function()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 20000
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
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel("x")
ax.set_ylabel("u(t, x)")
ax.set_title("1D Burgers' Equation Solution")
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.grid(True)

line, = ax.plot([], [], 'b-', lw=2, label="PINN Prediction")
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

x_plot = torch.linspace(-1, 1, 400, device=device).view(-1, 1)

x_initial_plot = np.linspace(-1, 1, 400)
u_initial_plot = -np.sin(np.pi * x_initial_plot)
ax.plot(x_initial_plot, u_initial_plot, 'r--', label="Initial Condition t=0")
ax.legend()

def update_plot(frame_t):
    t_plot = torch.full_like(x_plot, frame_t)
    with torch.no_grad():
        u_pred = model(t_plot, x_plot).cpu().numpy()
    
    line.set_data(x_plot.cpu().numpy(), u_pred)
    time_text.set_text(f'Time = {frame_t:.2f} s')
    return line, time_text

time_points = np.linspace(0, 1, 101)
ani = animation.FuncAnimation(fig, update_plot, frames=time_points, blit=True, interval=50)

try:
    print("Saving animation to 'burgers_equation_1d.gif'...")
    ani.save('examples/burgers_equation_1d.gif', writer='pillow', fps=20)
    print("Done.")
except Exception as e:
    print(f"\nCould not save animation. Error: {e}")
    print("Displaying animation instead.")
    plt.show()