# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import torch

from model import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_point(model_path):
    loaded_model = torch.load(model_path, map_location=torch.device('cpu')).to(device)

    # Set the model to evaluation mode if you plan to use it for inference
    loaded_model.eval()

    len_eval = 1000

    with torch.no_grad():
        x_eval = -1.0 + (torch.arange(1, len_eval + 1).to(device) / len_eval - (0.5 / len_eval)) * 2.0
        t_eval = torch.arange(1, len_eval + 1).to(device) / len_eval - (0.5 / len_eval)
        # Cartesian product
        all_points = torch.cartesian_prod(x_eval, t_eval).to(device)  # (x, t)
        # torch.cartesian_prod: The behavior is similar to python’s itertools.product
        # assert all_points.shape == (len_eval * len_eval, 2)
        all_results = loaded_model(all_points[:, 0:1], all_points[:, 1:2]).detach().squeeze()
        # assert all_results.shape == torch.Size([len_eval * len_eval])

        return all_points, all_results


def draw_Burger(model_path):
    all_points, all_results = get_point(model_path)

    plt.figure(figsize=(12, 6))

    # Create scatter plot
    plt.scatter(all_points[:, 1:2].squeeze().cpu(), all_points[:, 0:1].squeeze().cpu(), c=all_results.cpu(),
                cmap='rainbow')

    # Add a color bar
    plt.colorbar()

    # Add labels and title if needed
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title("u(t, x)")

    # Show the plot
    plt.show()


def draw_Diffusion(model_path):
    all_points, all_results = get_point(model_path)

    plt.figure(figsize=(12, 6))

    # Create scatter plot
    plt.scatter(all_points[:, 1:2].squeeze().cpu(), all_points[:, 0:1].squeeze().cpu(), c=all_results.cpu(), cmap='bwr')

    # Add a color bar
    plt.colorbar()

    # Add labels and title if needed
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Diffusion Equation')

    # Show the plot
    plt.show()


def draw_allan(model_path):
    all_points, all_results = get_point(model_path)

    plt.figure(figsize=(12, 6))

    # Create scatter plot
    plt.scatter(all_points[:, 1:2].squeeze().cpu(), all_points[:, 0:1].squeeze().cpu(), c=all_results.cpu(), cmap="bwr")

    # Add a color bar
    plt.colorbar()

    # Add labels and title if needed
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title("u(t, x)")

    # Show the plot
    plt.show()


def draw_fisher(model_path):
    loaded_model = torch.load(model_path, map_location=torch.device('cpu')).to(device)

    len_eval = 1000

    with torch.no_grad():
        # x_eval = -1.0 + (torch.arange(1, len_eval + 1).to(device) / len_eval - (0.5 / len_eval)) * 2.0
        x_eval = torch.arange(1, len_eval + 1).to(device) / len_eval - (0.5 / len_eval)
        t_eval = torch.arange(1, len_eval + 1).to(device) / len_eval - (0.5 / len_eval)
        # Cartesian product
        all_points = torch.cartesian_prod(x_eval, t_eval).to(device)  # (x, t)
        # torch.cartesian_prod: The behavior is similar to python’s itertools.product
        assert all_points.shape == (len_eval * len_eval, 2)
        all_results = loaded_model(all_points[:, 0:1], all_points[:, 1:2]).detach().squeeze()
        assert all_results.shape == torch.Size([len_eval * len_eval])

    plt.figure(figsize=(12, 6))

    # Create scatter plot
    plt.scatter(all_points[:, 1:2].squeeze().cpu(), all_points[:, 0:1].squeeze().cpu(), c=all_results.cpu(), cmap="bwr")

    # Add a color bar
    plt.colorbar()

    # Add labels and title if needed
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title("u(t, x)")

    # Show the plot
    plt.show()


def draw_Schrodinger(model_path):
    loaded_model = torch.load(model_path, map_location=torch.device('cpu')).to(device)

    len_eval = 1000

    x_min = -5.0
    x_max = 5.0
    t_min = 0.0
    t_max = torch.pi / 2.0

    with torch.no_grad():
        t_eval = t_min + (torch.arange(1, len_eval + 1).to(device) / len_eval - (0.5 / len_eval)) * (t_max - t_min)
        x_eval = x_min + (torch.arange(1, len_eval + 1).to(device) / len_eval - (0.5 / len_eval)) * (x_max - x_min)
        # Cartesian product
        all_points = torch.cartesian_prod(t_eval, x_eval).to(device)  # (t, x)
        # torch.cartesian_prod: The behavior is similar to python’s itertools.product
        assert all_points.shape == (len_eval * len_eval, 2)
        real_part, imaginary_part = loaded_model(all_points[:, 0:1], all_points[:, 1:2])
        all_results = torch.sqrt(real_part.squeeze() ** 2 + imaginary_part.squeeze() ** 2)
        assert all_results.shape == torch.Size([len_eval * len_eval])

    plt.figure(figsize=(18, 8))

    # Create scatter plot
    plt.scatter(all_points[:, 0:1].squeeze().cpu(), all_points[:, 1:2].squeeze().cpu(), c=all_results.cpu(),
                cmap='viridis_r')
    # 'viridis' is a color map, you can choose others
    # cmap='plasma'

    # Add a color bar
    plt.colorbar()

    # Add labels and title if needed
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('|h(t, x)|')

    # Show the plot
    plt.show()


def part_graph(model_path, t):
    loaded_model = torch.load(model_path, map_location=torch.device('cpu')).to(device)

    # Set the model to evaluation mode if you plan to use it for inference
    loaded_model.eval()

    len_eval = 1000

    with torch.no_grad():
        x_eval = -1.0 + (torch.arange(1, len_eval + 1).to(device) / len_eval - (0.5 / len_eval)) * 2.0
        t_eval = torch.arange(1, len_eval + 1).to(device) / len_eval - (0.5 / len_eval)
        # Cartesian product
        all_points = torch.cartesian_prod(x_eval, t_eval).to(device)  # (x, t)
        # torch.cartesian_prod: The behavior is similar to python’s itertools.product
        assert all_points.shape == (len_eval * len_eval, 2)
        all_results = loaded_model(all_points[:, 0:1], all_points[:, 1:2]).detach().squeeze()
        assert all_results.shape == torch.Size([len_eval * len_eval])

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    t = int(float(t) * len_eval)

    ax.plot(x_eval.cpu(),
            all_results[torch.tensor([i + t for i in range(0, len_eval * len_eval, len_eval)]).to(device)].cpu(),
            'b-')
    ax.set_title('t = ' + str(float(t_eval[t])))
    ax.set_xlabel('x')
    ax.set_ylabel('u(t, x)')

    plt.show()


def part_graph_Schrodinger(model_path, t):
    loaded_model = torch.load(model_path, map_location=torch.device('cpu')).to(device)

    len_eval = 1000

    x_min = -5.0
    x_max = 5.0
    t_min = 0.0
    t_max = torch.pi / 2.0

    with torch.no_grad():
        t_eval = t_min + (torch.arange(1, len_eval + 1).to(device) / len_eval - (0.5 / len_eval)) * (t_max - t_min)
        x_eval = x_min + (torch.arange(1, len_eval + 1).to(device) / len_eval - (0.5 / len_eval)) * (x_max - x_min)
        # Cartesian product
        all_points = torch.cartesian_prod(t_eval, x_eval).to(device)  # (t, x)
        # torch.cartesian_prod: The behavior is similar to python’s itertools.product
        assert all_points.shape == (len_eval * len_eval, 2)
        real_part, imaginary_part = loaded_model(all_points[:, 0:1], all_points[:, 1:2])
        all_results = torch.sqrt(real_part.squeeze() ** 2 + imaginary_part.squeeze() ** 2)
        assert all_results.shape == torch.Size([len_eval * len_eval])

    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    t = int(float(t) * len_eval)

    ax.set_ylim(0, 5)
    ax.plot(x_eval.cpu(), all_results[len_eval * t:len_eval * (t + 1)].cpu(), 'b-')
    ax.set_title('t = ' + str(float(t_eval[t])))
    ax.set_xlabel('x')
    ax.set_ylabel('|h(t, x)|')

    plt.show()


def print_result(model_path, x, t):
    loaded_model = torch.load(model_path, map_location=device)

    # Convert input values to tensors and add an extra dimension
    x_tensor = torch.tensor(float(x)).unsqueeze(0).unsqueeze(0)
    t_tensor = torch.tensor(float(t)).unsqueeze(0).unsqueeze(0)

    # Get the result from the loaded model
    result = loaded_model(x_tensor, t_tensor).detach().squeeze().item()
    messagebox.showinfo("Calculation Result", f"Result: {result}")


def get_selection():
    model = combo.get()
    print(model)
    model_path = ""
    if model == "Burger Equation":
        model_path = "Burger_PINN_model.pt"
    elif model == "Diffusion Equation":
        model_path = "Diffusion_PINN_model.pt"
    elif model == "Allen Cahn Equation":
        model_path = "allan_chan_equation_model.pt"
    elif model == "Fisher KPP Equation":
        model_path = "Fisher_KPP_equation_model.pt"
    elif model == "Schrodinger Equation":
        model_path = "PINN_model.pt"

    x_value = entry_x.get()
    t_value = entry_t.get()
    print("x:", x_value)
    print("t:", t_value)
    if x_value == "" and t_value == "":
        if model == "Burger Equation":
            draw_Burger(model_path)
        elif model == "Diffusion Equation":
            draw_Diffusion(model_path)
        elif model == "Allen Cahn Equation":
            draw_allan(model_path)
        elif model == "Fisher KPP Equation":
            draw_fisher(model_path)
        elif model == "Schrodinger Equation":
            draw_Schrodinger(model_path)
    elif x_value == "" and t_value != "":
        print("show part graph")
        if model == "Schrodinger Equation":
            part_graph_Schrodinger(model_path, t_value)
        else:
            part_graph(model_path, t_value)
    else:
        print_result(model_path, x_value, t_value)


root = tk.Tk()
root.title("Equation")

# Create dropdown menu
options = ["Burger Equation", "Diffusion Equation", "Allen Cahn Equation", "Fisher KPP Equation", "Poisson Equation", "Schrodinger Equation"]
combo = ttk.Combobox(root, values=options)
combo.grid(row=0, column=0, columnspan=2, pady=5)

# Create labels and entry boxes for x and t
label_x = tk.Label(root, text="x:")
label_x.grid(row=1, column=0)

entry_x = tk.Entry(root)
entry_x.grid(row=1, column=1)

label_t = tk.Label(root, text="t:")
label_t.grid(row=2, column=0)

entry_t = tk.Entry(root)
entry_t.grid(row=2, column=1)

# Create button to get user selections
btn = tk.Button(root, text="Get Selection", command=get_selection)
btn.grid(row=3, column=0, columnspan=2, pady=10)

root.mainloop()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
