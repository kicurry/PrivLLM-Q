import numpy as np
from torch import nn
import torch
import matplotlib.pyplot as plt

def test_bolt_acc():
    X = torch.linspace(-2.7, 2.7, 1000)
    layer = nn.SiLU()
    gelu = nn.GELU()
    ## draw the silu function picture, X=input, Y=silu(X)
    Y = gelu(X)
    plt.plot(X.detach().numpy(), Y.detach().numpy(), label="silu")
    # plt.savefig("silu.png")

    abs_X = torch.abs(X)
    a = 0.020848611754127593
    b = -0.18352506127082727
    c = 0.5410550166368381
    d = -0.03798164612714154
    e = 0.001620808531841547
    y_smooth = a * (abs_X ** 4) + b * (abs_X ** 3) + c * (abs_X ** 2) + d * abs_X + e+0.5*X
    print("mse of gelu:", np.mean((Y.detach().numpy() - y_smooth.detach().numpy())**2))

    # Y = gelu(X)
    plt.plot(X.detach().numpy(), y_smooth.detach().numpy(), label="poly")
    plt.legend()
    plt.savefig("gelu.png")
    plt.close()

def test_silu_acc():
    point = 4.6
    X = np.linspace(0, point, 1000)
    layer = nn.SiLU()
    ## draw the silu function picture, X=input, Y=silu(X)
    Y = layer(torch.tensor(X)).detach().numpy()
    
    tmp_Y = Y-0.5*X
    coefficients = np.polyfit(X, tmp_Y, 4)
    print("coefficients:", coefficients)
    a,b,c,d,e = coefficients
    
    X = np.linspace(-15, 15, 1000)
    Y = layer(torch.tensor(X)).detach().numpy()
    plt.plot(X, Y, label="silu")
    abs_X = np.abs(X)

    y_smooth = a * (abs_X ** 4) + b * (abs_X ** 3) + c * (abs_X ** 2) + d * abs_X + e + 0.5*X
    y_smooth[X>point] = X[X>point]
    y_smooth[X<-point] = 0
    print("mse of silu:", np.mean((Y - y_smooth)**2))

    plt.plot(X, y_smooth, label="poly")
    plt.legend()
    plt.savefig("silu.png")
    plt.close()
    
if __name__ == "__main__":
    test_silu_acc()
    # test_bolt_acc()
