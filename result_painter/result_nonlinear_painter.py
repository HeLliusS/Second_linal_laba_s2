import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Папка с результатами (та же, куда пишет nonlinear_demo)
results_dir = "./cmake-build-debug/nonlinear_results"

# ---- 1. XOR данные ----
xor_path = os.path.join(results_dir, "xor_points.csv")
if os.path.exists(xor_path):
    xor_df = pd.read_csv(xor_path)
    plt.figure(figsize=(8, 6))
    plt.scatter(xor_df[xor_df.label==0]['x1'], xor_df[xor_df.label==0]['x2'],
                label='Class 0', alpha=0.7, edgecolors='k')
    plt.scatter(xor_df[xor_df.label==1]['x1'], xor_df[xor_df.label==1]['x2'],
                label='Class 1', alpha=0.7, edgecolors='k')
    plt.title("XOR-like data (non-linearly separable)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "xor_plot.png"))
    plt.show()
else:
    print("XOR points file not found. Run nonlinear_demo first.")

# ---- 2. Circle данные + разделяющая прямая ----
circle_points_path = os.path.join(results_dir, "circle_points.csv")
circle_line_path = os.path.join(results_dir, "circle_line.csv")
if os.path.exists(circle_points_path) and os.path.exists(circle_line_path):
    circ_df = pd.read_csv(circle_points_path)
    line_df = pd.read_csv(circle_line_path)
    plt.figure(figsize=(8, 6))
    plt.scatter(circ_df[circ_df.label==0]['x1'], circ_df[circ_df.label==0]['x2'],
                label='Outside circle', alpha=0.7, edgecolors='k')
    plt.scatter(circ_df[circ_df.label==1]['x1'], circ_df[circ_df.label==1]['x2'],
                label='Inside circle', alpha=0.7, edgecolors='k')
    # Рисуем разделяющую прямую (бесполезную)
    plt.plot(line_df['x1'], line_df['x2'], 'r-', linewidth=2, label='Linear decision boundary')
    # Для наглядности нарисуем истинную границу – окружность радиуса 1.5 (если использовали радиус 1.5)
    # (можно закомментировать, если не нужно)
    theta = np.linspace(0, 2*np.pi, 100)
    radius = 1.5
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)
    plt.plot(circle_x, circle_y, 'g--', label='True boundary (circle)')
    plt.title("Circle data – linear boundary fails")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(results_dir, "circle_plot.png"))
    plt.show()
else:
    print("Circle data files not found. Run nonlinear_demo first.")
