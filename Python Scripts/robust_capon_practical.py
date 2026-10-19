
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

timings = {}

def timed_start():
    return time.perf_counter()

def timed_end(name, t0):
    timings[name] = timings.get(name, 0.0) + time.perf_counter() - t0


# ============================================================
# 1. Common parameters
# ============================================================

M = 8
d = 0.5

theta0 = 25
theta_int = 30

P_signal = 1.0
epsilon = 1.0
C = np.eye(M)

cases = [
    {"name": "Case 1: High interference/noise",
     "P_int": 10.0, "P_noise": 1.0},
    {"name": "Case 2: Low interference/noise",
     "P_int": 0.0005, "P_noise": 0.0001},
]


# ============================================================
# 2. Steering vectors
# ============================================================

def steering_vector(theta_deg):
    m = np.arange(M)
    theta = np.deg2rad(theta_deg)
    return np.exp(
        -1j * 2 * np.pi * d * m * np.sin(theta)
    ).reshape(M, 1)


a0 = steering_vector(theta0)
a_int = steering_vector(theta_int)


# ============================================================
# 3. Run RCB and SCB for one scenario
# ============================================================

def run_case(P_int, P_noise, case_name):

    print(f"\n{'=' * 65}")
    print(case_name)
    print(f"P_signal = {P_signal}, P_int = {P_int}, "
          f"P_noise = {P_noise}")
    print("=" * 65)

    # Theoretical covariance matrices
    t0 = timed_start()
    R_signal = P_signal * (a0 @ a0.conj().T)
    R_int = P_int * (a_int @ a_int.conj().T)
    R_noise = P_noise * np.eye(M)
    R = R_signal + R_int + R_noise
    timed_end("Covariance construction", t0)

    # Spherical uncertainty transformation
    D = np.sqrt(epsilon) * np.eye(M)
    a0_tilde = D @ a0
    R_tilde = D @ R @ D.conj().T

    # Eigendecomposition
    t0 = timed_start()
    eigenvalues, U = np.linalg.eigh(R_tilde)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    U = U[:, idx]

    z = U.conj().T @ a0_tilde
    gamma_max = eigenvalues[0]
    gamma_min = eigenvalues[-1]
    norm_a0 = np.linalg.norm(a0_tilde)
    timed_end("Eigendecomposition", t0)

    # Bounds for lambda
    t0 = timed_start()
    lambda_lower = (
        norm_a0 / np.sqrt(epsilon) - 1
    ) / gamma_max

    lambda_upper_25 = (
        norm_a0 / np.sqrt(epsilon) - 1
    ) / gamma_min

    lambda_upper_26 = np.sqrt(
        np.sum(np.abs(z.flatten())**2 / eigenvalues**2)
        / epsilon
    )

    lambda_upper = min(lambda_upper_25, lambda_upper_26)

    def f_lambda(lam):
        return np.sum(
            np.abs(z.flatten())**2
            / (1 + lam * eigenvalues)**2
        ) - epsilon

    lambda_star = brentq(
        f_lambda, lambda_lower, lambda_upper
    )
    timed_end("Lambda bounds and root solver", t0)

    # Optimal steering vector, Eq. (29)
    t0 = timed_start()
    a_tilde_opt = a0_tilde - np.linalg.solve(
        np.eye(M) + lambda_star * R_tilde,
        a0_tilde
    )
    a_opt = np.linalg.solve(D, a_tilde_opt)
    timed_end("Optimal steering vector", t0)

    # Power estimate, Eq. (30)
    sigma0_squared_hat = 1.0 / np.real(
        (a_opt.conj().T @ np.linalg.solve(R, a_opt))[0, 0]
    )

    # Steering vector comparison
    relative_error_a = (
        np.linalg.norm(a_opt - a0) / np.linalg.norm(a0)
    )

    delta_a = a_opt - a0
    uncertainty_value = np.real(
        (delta_a.conj().T
         @ np.linalg.solve(C, delta_a))[0, 0]
    )

    # Conventional Capon weights
    R_inv_a0 = np.linalg.solve(R, a0)
    w_scb = R_inv_a0 / (a0.conj().T @ R_inv_a0)[0, 0]

    # Robust Capon weights
    R_inv_aopt = np.linalg.solve(R, a_opt)
    w_rcb = R_inv_aopt / (
        a_opt.conj().T @ R_inv_aopt
    )[0, 0]

    # Angular responses
    angles = np.linspace(-90, 90, 1001)
    A_scan = np.column_stack([
        steering_vector(angle).flatten()
        for angle in angles
    ])

    response_scb = (w_scb.conj().T @ A_scan).flatten()
    response_rcb = (w_rcb.conj().T @ A_scan).flatten()

    pattern_scb_db = 20 * np.log10(
        np.maximum(
            np.abs(response_scb) / np.max(np.abs(response_scb)),
            1e-12
        )
    )
    pattern_rcb_db = 20 * np.log10(
        np.maximum(
            np.abs(response_rcb) / np.max(np.abs(response_rcb)),
            1e-12
        )
    )

    # Output power contributions
    def output_power(w, covariance):
        return np.real(
            (w.conj().T @ covariance @ w)[0, 0]
        )

    powers = {
        "SCB": [
            output_power(w_scb, R_signal),
            output_power(w_scb, R_int),
            output_power(w_scb, R_noise),
        ],
        "RCB": [
            output_power(w_rcb, R_signal),
            output_power(w_rcb, R_int),
            output_power(w_rcb, R_noise),
        ],
    }

    # Print results
    print(f"\nLambda*:                  {lambda_star:.8f}")
    print(f"Estimated signal power:   {sigma0_squared_hat:.6f}")
    print(f"Relative steering error:  {relative_error_a:.6f}")
    print(f"Uncertainty value:        {uncertainty_value:.6f}")
    print(f"Constraint satisfied:     "
          f"{uncertainty_value <= epsilon + 1e-8}")

    print("\nOutput powers:")
    print("                 SCB          RCB")
    for i, label in enumerate(["Desired", "Interference", "Noise"]):
        print(f"{label:14s} {powers['SCB'][i]:.6f}    "
              f"{powers['RCB'][i]:.6f}")

    print(f"{'Total':14s} "
          f"{sum(powers['SCB']):.6f}    "
          f"{sum(powers['RCB']):.6f}")

    return {
        "name": case_name,
        "lambda": lambda_star,
        "a_opt": a_opt,
        "relative_error": relative_error_a,
        "uncertainty": uncertainty_value,
        "sigma_hat": sigma0_squared_hat,
        "angles": angles,
        "pattern_scb_db": pattern_scb_db,
        "pattern_rcb_db": pattern_rcb_db,
        "powers": powers,
    }


# ============================================================
# 4. Run both scenarios
# ============================================================

results = [
    run_case(case["P_int"], case["P_noise"], case["name"])
    for case in cases
]


# ============================================================
# 5. Comparison of the two scenarios
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

for ax, result in zip(axes, results):
    ax.plot(
        result["angles"], result["pattern_scb_db"],
        label="SCB"
    )
    ax.plot(
        result["angles"], result["pattern_rcb_db"],
        "--", label="RCB"
    )
    ax.axvline(theta0, color="green", linestyle=":",
               label="Desired signal (25°)")
    ax.axvline(theta_int, color="red", linestyle=":",
               label="Interference (40°)")
    ax.set_title(result["name"])
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylim(-60, 5)
    ax.grid(True)
    ax.legend()

axes[0].set_ylabel("Normalized response (dB)")
fig.suptitle("SCB vs. RCB for both interference/noise scenarios")
plt.tight_layout()
plt.show()


# ============================================================
# 6. Summary of lambda and steering vector results
# ============================================================

print("\n--- Scenario summary ---")
print(f"{'Scenario':35s} {'Lambda*':>12s} "
      f"{'Steering error':>16s} {'Power estimate':>16s}")

for result in results:
    print(f"{result['name']:35s} "
          f"{result['lambda']:12.6f} "
          f"{result['relative_error']:16.6f} "
          f"{result['sigma_hat']:16.6f}")


# ============================================================
# 7. Execution times
# ============================================================

print("\n--- Accumulated execution times ---")
for process, elapsed in timings.items():
    print(f"{process:32s}: {elapsed * 1e3:.4f} ms")

print(f"{'TOTAL measured time':32s}: "
      f"{sum(timings.values()) * 1e3:.4f} ms")







