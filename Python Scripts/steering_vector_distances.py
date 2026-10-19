import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================

M = 10
d = 0.5  # Antenna spacing in wavelengths

theta_nominal = 0.0
thetas = np.round(np.arange(0.0, 10.0 + 0.05, 0.1), 1)

# ============================================================
# STEERING VECTOR
# ============================================================

def steering_vector(theta_deg):
    m = np.arange(M)
    theta = np.deg2rad(theta_deg)

    return np.exp(
        -1j * 2 * np.pi * d * m * np.sin(theta)
    ).reshape(M, 1)


# Nominal steering vector
a0 = steering_vector(theta_nominal)

# ============================================================
# COMPUTE SQUARED DISTANCE
# epsilon_min(theta) = ||a(theta) - a(0)||^2
# ============================================================

epsilon_min = []

for theta in thetas:
    a_theta = steering_vector(theta)

    difference = a_theta - a0

    epsilon = np.linalg.norm(difference)**2

    epsilon_min.append(epsilon)

epsilon_min = np.array(epsilon_min)

# ============================================================
# PRINT RESULTS
# ============================================================

print(f"{'Theta (deg)':>12} | {'Epsilon minimum':>18}")
print("-" * 34)

for theta, eps in zip(thetas, epsilon_min):
    print(f"{theta:12.1f} | {eps:18.6f}")

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(thetas, epsilon_min)

plt.xlabel("Actual SOI angle (degrees)")
plt.ylabel(r"$\epsilon_{\min} = \|a(\theta)-a(0^\circ)\|^2$")
plt.title("Minimum uncertainty radius required vs angular mismatch")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# VALUES OF INTEREST
# ============================================================

for theta_target in [2.0, 5.0, 10.0]:
    idx = np.argmin(np.abs(thetas - theta_target))

    print(
        f"For a mismatch of {thetas[idx]:.1f} degrees: "
        f"epsilon_min = {epsilon_min[idx]:.6f}"
    )