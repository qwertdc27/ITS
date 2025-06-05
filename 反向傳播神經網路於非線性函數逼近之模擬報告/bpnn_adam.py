
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. 目標函數定義 ---
def target_function(x, y):
    return 5 * np.sin(np.pi * x**2) * np.sin(2 * np.pi * y) + 1

# --- 2. 理論正規化與反正規化 ---
def normalize_theoretical(data, d_min=-4, d_max=6, a=0.2, b=0.8):
    return (data - d_min) / (d_max - d_min) * (b - a) + a

def denormalize_theoretical(norm_data, d_min=-4, d_max=6, a=0.2, b=0.8):
    return (norm_data - a) / (b - a) * (d_max - d_min) + d_min

# --- 3. 激活函數與導數 ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# --- 4. 產生資料 ---
np.random.seed(0)
N_train = 300
N_test = 100
x = np.random.uniform(-0.8, 0.7, N_train + N_test).reshape(-1, 1)
y = np.random.uniform(-0.8, 0.7, N_train + N_test).reshape(-1, 1)
z = target_function(x, y)
z_norm = normalize_theoretical(z)

X = np.hstack([x, y])
Z = z_norm.reshape(-1, 1)

X_train, X_test = X[:N_train], X[N_train:]
Z_train, Z_test = Z[:N_train], Z[N_train:]

# --- 5. BPNN 結構參數 ---
input_size = 2
hidden_size = 10
output_size = 1
lr = 0.01
epochs = 5000
beta1, beta2, eps = 0.9, 0.999, 1e-8

# --- 6. 權重初始化與 Adam 參數 ---
W1 = np.random.uniform(-0.5, 0.5, size=(hidden_size, input_size))
b1 = np.zeros((hidden_size, 1))
W2 = np.random.uniform(-0.5, 0.5, size=(output_size, hidden_size))
b2 = np.zeros((output_size, 1))

mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)
mb1 = np.zeros_like(b1); vb1 = np.zeros_like(b1)
mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2)
mb2 = np.zeros_like(b2); vb2 = np.zeros_like(b2)

E_list = []

# --- 7. 訓練迴圈（Batch Mode + Adam + 學習率衰減）---
for epoch in range(1, epochs + 1):
    lr_t = lr * (0.95 ** (epoch // 1000))

    X_input = X_train.T
    Z_target = Z_train.T

    net1 = W1 @ X_input + b1
    out1 = sigmoid(net1)
    net2 = W2 @ out1 + b2
    out2 = sigmoid(net2)

    e = Z_target - out2
    delta2 = e * sigmoid_deriv(net2)
    delta1 = (W2.T @ delta2) * sigmoid_deriv(net1)

    dW2 = delta2 @ out1.T / N_train
    db2 = np.sum(delta2, axis=1, keepdims=True) / N_train
    dW1 = delta1 @ X_input.T / N_train
    db1 = np.sum(delta1, axis=1, keepdims=True) / N_train

    # Adam 更新
    mW2 = beta1 * mW2 + (1 - beta1) * dW2
    vW2 = beta2 * vW2 + (1 - beta2) * (dW2 ** 2)
    mW2_hat = mW2 / (1 - beta1 ** epoch)
    vW2_hat = vW2 / (1 - beta2 ** epoch)
    W2 += lr_t * mW2_hat / (np.sqrt(vW2_hat) + eps)

    mb2 = beta1 * mb2 + (1 - beta1) * db2
    vb2 = beta2 * vb2 + (1 - beta2) * (db2 ** 2)
    mb2_hat = mb2 / (1 - beta1 ** epoch)
    vb2_hat = vb2 / (1 - beta2 ** epoch)
    b2 += lr_t * mb2_hat / (np.sqrt(vb2_hat) + eps)

    mW1 = beta1 * mW1 + (1 - beta1) * dW1
    vW1 = beta2 * vW1 + (1 - beta2) * (dW1 ** 2)
    mW1_hat = mW1 / (1 - beta1 ** epoch)
    vW1_hat = vW1 / (1 - beta2 ** epoch)
    W1 += lr_t * mW1_hat / (np.sqrt(vW1_hat) + eps)

    mb1 = beta1 * mb1 + (1 - beta1) * db1
    vb1 = beta2 * vb1 + (1 - beta2) * (db1 ** 2)
    mb1_hat = mb1 / (1 - beta1 ** epoch)
    vb1_hat = vb1 / (1 - beta2 ** epoch)
    b1 += lr_t * mb1_hat / (np.sqrt(vb1_hat) + eps)

    E = 0.5 * np.mean(e**2)
    E_list.append(E)

# --- 8. 預測函數 ---
def predict(X_input):
    X_input = X_input.T
    net1 = W1 @ X_input + b1
    out1 = sigmoid(net1)
    net2 = W2 @ out1 + b2
    out2 = sigmoid(net2)
    return out2.T

Z_pred_train = predict(X_train)
Z_pred_test = predict(X_test)

Z_pred_train_real = denormalize_theoretical(Z_pred_train)
Z_pred_test_real = denormalize_theoretical(Z_pred_test)
Z_train_real = denormalize_theoretical(Z_train)
Z_test_real = denormalize_theoretical(Z_test)

E_train = 0.5 * np.mean((Z_train_real - Z_pred_train_real)**2)
E_test = 0.5 * np.mean((Z_test_real - Z_pred_test_real)**2)

print("E_train:", E_train)
print("E_test:", E_test)

# --- 9. 圖形：收斂曲線與 3D 曲面圖 ---
plt.figure(figsize=(8, 5))
plt.plot(E_list)
plt.xlabel("Epoch (n)")
plt.ylabel("Average Error E")
plt.title("Error vs Epoch")
plt.grid(True)
plt.tight_layout()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

grid_x, grid_y = np.meshgrid(np.linspace(-0.8, 0.7, 100), np.linspace(-0.8, 0.7, 100))
grid_x_flat = grid_x.ravel().reshape(-1, 1)
grid_y_flat = grid_y.ravel().reshape(-1, 1)
grid_input = np.hstack([grid_x_flat, grid_y_flat])

grid_z_true = target_function(grid_x_flat, grid_y_flat).reshape(100, 100)
grid_z_pred_norm = predict(grid_input)
grid_z_pred = denormalize_theoretical(grid_z_pred_norm).reshape(100, 100)

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(grid_x, grid_y, grid_z_true, cmap='viridis', edgecolor='none')
ax1.set_title("Target Function")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("F(x, y)")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(grid_x, grid_y, grid_z_pred, cmap='plasma', edgecolor='none')
ax2.set_title("BPNN Prediction")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("F_pred(x, y)")

plt.tight_layout()
plt.show()
