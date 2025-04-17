import numpy as np
import matplotlib.pyplot as plt

# パラメータ設定
Du = 0.1     # 細菌の拡散係数
Dv = 0.05    # 化学物質の拡散係数
chi = 5.0    # 走性係数
a = 1.0
b = 0.5
c = 0.5

# シミュレーション設定
nx, ny = 100, 100   # 格子数
dx = dy = 1.0       # 空間刻み
dt = 0.01           # 時間刻み
steps = 1000        # 時間ステップ数

# 初期条件
u = np.random.rand(nx, ny) * 0.1
v = np.zeros((nx, ny))
v[nx//2-5:nx//2+5, ny//2-5:ny//2+5] = 1.0  # 中央に誘引物質

def laplacian(Z):
    return (np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z) / dx**2

# 時間発展
for step in range(steps):
    Lu = laplacian(u)
    Lv = laplacian(v)

    grad_v_x = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dx)
    grad_v_y = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dy)

    chemotaxis_x = (np.roll(u * grad_v_x, -1, axis=0) - np.roll(u * grad_v_x, 1, axis=0)) / (2 * dx)
    chemotaxis_y = (np.roll(u * grad_v_y, -1, axis=1) - np.roll(u * grad_v_y, 1, axis=1)) / (2 * dy)
    chemotaxis = chi * (chemotaxis_x + chemotaxis_y)

    # 更新式
    u += dt * (Du * Lu - chemotaxis + u * (1 - u - a * v))
    v += dt * (Dv * Lv + b * v * (u - c))

    # 可視化（任意）
    if step % 100 == 0:
        plt.imshow(u, cmap='viridis')
        plt.title(f"Step {step}")
        plt.colorbar(label='u (bacteria density)')
        plt.pause(0.01)
        plt.clf()

plt.show()
