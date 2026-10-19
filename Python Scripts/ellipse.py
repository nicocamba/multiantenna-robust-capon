
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# 1. PARAMETERS TO EDIT
# ==================================================

C = np.array([
    [4.0, 1.5],
    [1.5, 2.0]
])

a_bar = np.array([0.0, 0.0])  # Ellipse center
epsilons = [0.5, 2.0]         # Two epsilon values

# ==================================================
# 2. EIGENVALUES AND EIGENVECTORS
# ==================================================

if not np.allclose(C, C.T):
    raise ValueError("C must be symmetric.")

eigenvalues, eigenvectors = np.linalg.eigh(C)

if np.any(eigenvalues <= 0):
    raise ValueError("C must be positive definite.")

# Sort in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Matrix C:")
print(C)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors (columns):")
print(eigenvectors)

# ==================================================
# 3. BUILD THE ELLIPSES
# ==================================================

theta = np.linspace(0, 2*np.pi, 500)
circle = np.array([np.cos(theta), np.sin(theta)])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, eps in zip(axes, epsilons):

    # Semiaxis lengths
    radii = np.sqrt(eps * eigenvalues)

    # Boundary parameterization:
    # a = a_bar + V diag(radii) [cos(theta), sin(theta)]
    ellipse = (
        a_bar[:, None]
        + eigenvectors @ (radii[:, None] * circle)
    )

    ax.plot(ellipse[0], ellipse[1], linewidth=2)
    ax.fill(ellipse[0], ellipse[1], alpha=0.15)

    # Center
    ax.scatter(*a_bar, color="black", marker="x",
               s=70, label="Center")

    # Principal axes: eigenvector directions
    for i in range(2):
        v = eigenvectors[:, i]
        r = radii[i]

        ax.plot(
            [a_bar[0] - r*v[0], a_bar[0] + r*v[0]],
            [a_bar[1] - r*v[1], a_bar[1] + r*v[1]],
            "--",
            label=f"Eigenvector {i+1}, lambda={eigenvalues[i]:.2f}"
        )

    ax.set_title(f"Uncertainty region: epsilon={eps}")
    ax.set_xlabel("a1")
    ax.set_ylabel("a2")
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8)

    print(f"\nepsilon = {eps}")
    print(f"Major semiaxis = {radii[0]:.4f}")
    print(f"Minor semiaxis = {radii[1]:.4f}")

plt.tight_layout()
plt.show()
