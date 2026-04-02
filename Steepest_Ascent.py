import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ── Define the objective function and its gradient ──────────
def f(x, y):
    return -(x**2 + y**2 - 4*x - 6*y + 4)
    # = -(x-2)² - (y-3)² + 9 → Maximum at (2, 3) with f = 9

def gradient(x, y):
    """∂f/∂x and ∂f/∂y"""
    dfdx = -(2*x - 4)     # = -2(x-2)
    dfdy = -(2*y - 6)     # = -2(y-3)
    return np.array([dfdx, dfdy])

# ── Steepest Ascent Algorithm ────────────────────────────────
def steepest_ascent(f, gradient, x0, y0, alpha=0.1, tol=1e-6, max_iter=200):
    """
    Maximize f(x,y) starting from (x0, y0).
    alpha  : step size (learning rate)
    tol    : stop when gradient norm < tol
    Returns: path of (x, y) points, final (x*, y*), f(x*, y*)
    """
    x, y = x0, y0
    path = [(x, y, f(x, y))]

    print(f"{'Iter':>5} {'x':>12} {'y':>12} {'f(x,y)':>12} {'|grad|':>12}")
    print("-" * 60)

    for i in range(1, max_iter + 1):
        grad = gradient(x, y)
        grad_norm = np.linalg.norm(grad)

        if i <= 20 or i % 20 == 0:
            print(f"{i:>5} {x:>12.8f} {y:>12.8f} {f(x,y):>12.8f} {grad_norm:>12.10f}")

        if grad_norm < tol:
            print(f"\n  Converged at iteration {i}  (|grad| = {grad_norm:.2e})")
            break

        x = x + alpha * grad[0]
        y = y + alpha * grad[1]
        path.append((x, y, f(x, y)))

    return path, x, y, f(x, y)

# ── Run ──────────────────────────────────────────────────────
x0, y0 = -1.0, -1.0        # Starting point (far from maximum)
alpha   = 0.15              # Step size

print("=" * 60)
print("  STEEPEST ASCENT METHOD")
print(f"  Function    : f(x,y) = -(x²+y²-4x-6y+4)")
print(f"  Start point : ({x0}, {y0})")
print(f"  Step size α : {alpha}")
print("=" * 60)

path, x_star, y_star, f_star = steepest_ascent(f, gradient, x0, y0, alpha=alpha)

print("-" * 60)
print(f"\n  ✔ Maximum found at   (x*, y*) = ({x_star:.8f}, {y_star:.8f})")
print(f"  ✔ Maximum value     f(x*, y*) = {f_star:.8f}")
print(f"  ✔ True maximum at   (x*, y*) = (2.0, 3.0),  f = 9.0")
print(f"  ✔ Total steps: {len(path) - 1}")

# ── Extract path arrays ──────────────────────────────────────
px = [p[0] for p in path]
py = [p[1] for p in path]
pf = [p[2] for p in path]

# ── Plot 1: Contour map with path ────────────────────────────
x_g = np.linspace(-3, 6, 300)
y_g = np.linspace(-3, 8, 300)
X, Y = np.meshgrid(x_g, y_g)
Z = f(X, Y)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Contour + path
cp = axes[0].contourf(X, Y, Z, levels=30, cmap='RdYlGn')
plt.colorbar(cp, ax=axes[0])
axes[0].contour(X, Y, Z, levels=30, colors='gray', linewidths=0.4, alpha=0.5)
axes[0].plot(px, py, 'b-o', markersize=3, linewidth=1.5, label='Ascent path', zorder=5)
axes[0].scatter(px[0],  py[0],  color='blue',  s=100, zorder=6, label=f'Start ({x0},{y0})')
axes[0].scatter(x_star, y_star, color='red',   s=150, zorder=6,
                marker='*', label=f'Max ({x_star:.2f},{y_star:.2f})')
axes[0].set_title("Steepest Ascent — Contour Map", fontweight='bold')
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

# f(x,y) vs iteration
axes[1].plot(range(len(pf)), pf, 'g-', linewidth=2)
axes[1].axhline(y=9.0, color='red', linestyle='--', linewidth=1.2, label='True max = 9')
axes[1].set_title("f(x,y) Value vs. Iteration", fontweight='bold')
axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("f(x, y)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle('Steepest Ascent Method', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
