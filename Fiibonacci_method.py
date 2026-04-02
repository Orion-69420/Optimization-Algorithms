import numpy as np
import matplotlib.pyplot as plt

# ── Define the objective function ───────────────────────────
def f(x):
    return (x - 3)**2 + 2    # Minimum at x = 3

# ── Generate Fibonacci numbers ───────────────────────────────
def fibonacci(n):
    fib = [1, 1]
    for _ in range(2, n + 1):
        fib.append(fib[-1] + fib[-2])
    return fib

# ── Fibonacci Search Method ──────────────────────────────────
def fibonacci_search(f, a, b, n=15):
    """
    Minimize f(x) on [a, b] using n Fibonacci iterations.
    Returns: x_min, f_min, history of (a, b) pairs
    """
    fib = fibonacci(n + 1)
    L0 = b - a
    history = [(a, b)]

    x1 = a + (fib[n - 2] / fib[n]) * L0
    x2 = a + (fib[n - 1] / fib[n]) * L0
    f1, f2 = f(x1), f(x2)

    print(f"{'Step':>5} {'a':>10} {'b':>10} {'x1':>10} {'x2':>10} {'f(x1)':>10} {'f(x2)':>10} {'Length':>12}")
    print("-" * 88)

    for k in range(1, n - 1):
        Fk  = fib[n - k]
        Fk1 = fib[n - k - 1]
        Fk2 = fib[n - k - 2] if (n - k - 2) >= 0 else 1

        print(f"{k:>5} {a:>10.6f} {b:>10.6f} {x1:>10.6f} {x2:>10.6f} {f1:>10.6f} {f2:>10.6f} {abs(b-a):>12.8f}")

        if f1 > f2:
            a  = x1
            x1 = x2;  f1 = f2
            x2 = a + (fib[n - k - 1] / fib[n - k]) * (b - a)
            f2 = f(x2)
        else:
            b  = x2
            x2 = x1;  f2 = f1
            x1 = a + (fib[n - k - 2] / fib[n - k]) * (b - a)
            f1 = f(x1)

        history.append((a, b))

    x_min = (a + b) / 2
    return x_min, f(x_min), history, fib[:n+1]

# ── Run ──────────────────────────────────────────────────────
a, b = 0.0, 8.0
n_steps = 15
print("=" * 88)
print("  FIBONACCI SEARCH METHOD")
print(f"  Function : f(x) = (x - 3)² + 2")
print(f"  Interval : [{a}, {b}]")
print(f"  Steps    : n = {n_steps}")
print("=" * 88)

x_min, f_min, history, fib = fibonacci_search(f, a, b, n=n_steps)

print("-" * 88)
print(f"\n  ✔ Minimum found at   x = {x_min:.8f}")
print(f"  ✔ Minimum value   f(x) = {f_min:.8f}")
print(f"  ✔ True minimum at    x = 3.0,  f(3) = 2.0")
print(f"  ✔ Total iterations: {len(history) - 1}")

print(f"\n  Fibonacci numbers used: {fib}")

# ── Plot 1: f(x) with minimum ────────────────────────────────
x_vals = np.linspace(a, b, 400)
y_vals = f(x_vals)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = (x−3)² + 2')
axes[0].axvline(x=x_min, color='red', linestyle='--', linewidth=1.5, label=f'Min at x={x_min:.4f}')
axes[0].scatter([x_min], [f_min], color='red', zorder=5, s=100)

# shade each successive interval lightly
for i, (ai, bi) in enumerate(history):
    axes[0].axvspan(ai, bi, alpha=0.03, color='green')

final_a, final_b = history[-1]
axes[0].axvspan(final_a, final_b, alpha=0.25, color='green', label='Final interval')

axes[0].set_title("f(x) — Fibonacci Search", fontweight='bold')
axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Plot 2: Interval length reduction
lengths = [abs(bi - ai) for ai, bi in history]
axes[1].plot(range(len(lengths)), lengths, 'o-', color='darkorange', linewidth=2)
axes[1].set_title("Interval Length vs. Iteration", fontweight='bold')
axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Interval Length")
axes[1].grid(True, alpha=0.3)

plt.suptitle('Fibonacci Search Method', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
