import torch
import matplotlib.pyplot as plt

# 非凸函数：多个局部最优 + 鞍点
def loss_fn(x, y):
    return torch.exp(-x**2 - y**2) * torch.sin(5 * x) * torch.sin(5 * y)


def optimize(optimizer_name, steps=500, lr=0.05):
    x = torch.tensor([0.0], requires_grad=True)
    y = torch.tensor([0.0], requires_grad=True)
    path = []

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam([x, y], lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD([x, y], lr=lr)
    else:
        raise ValueError("Unsupported optimizer")

    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(x, y)
        loss.backward()
        optimizer.step()
        path.append((x.item(), y.item(), loss.item()))

    return path


def plot_paths():
    import numpy as np

    # Create grid for contour plot
    x_grid = np.linspace(-2, 2, 200)
    y_grid = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.sin(3 * X) * np.sin(3 * Y) + 0.1 * (X ** 2 + Y ** 2)

    # Run optimizers
    adam_path = optimize('adam')
    sgd_path = optimize('sgd')

    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Loss')

    # Plot paths
    adam_x, adam_y, _ = zip(*adam_path)
    sgd_x, sgd_y, _ = zip(*sgd_path)
    plt.plot(adam_x, adam_y, 'r.-', label='Adam')
    plt.plot(sgd_x, sgd_y, 'w.-', label='SGD')

    plt.legend()
    plt.title("Adam vs SGD Optimization Path")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


plot_paths()
