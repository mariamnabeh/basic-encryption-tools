import numpy as np
import matplotlib.pyplot as plt

# 1. Define Lorenz System differential equations
def lorenz_deriv(state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])

# 2. Implement RK4 (Runge-Kutta 4th Order) numerical solver
def rk4_step(state, dt):
    k1 = lorenz_deriv(state)
    k2 = lorenz_deriv(state + 0.5 * dt * k1)
    k3 = lorenz_deriv(state + 0.5 * dt * k2)
    k4 = lorenz_deriv(state + dt * k3)
    
    # Calculate the next state using weighted average of k1, k2, k3, k4
    next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return next_state

# 3. Simulation Parameters & Initialization
dt = 0.01          # Time step size
steps = 10000      # Number of iteration steps
initial_state = np.array([1.0, 1.0, 1.0])  # Initial conditions (x0, y0, z0)

# Array to store the chaotic trajectory
trajectory = np.empty((steps, 3))
trajectory[0] = initial_state

# 4. Run the Simulation Loop (Generating the Key Stream)
current_state = initial_state
for i in range(1, steps):
    current_state = rk4_step(current_state, dt)
    trajectory[i] = current_state

# 5. Plotting the 3D Lorenz Attractor
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')

# Plotting the trajectory line
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], lw=0.6, color='purple')

ax.set_title("3D Lorenz Attractor (Chaos-based Key Generation)", fontsize=12)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")

plt.show()