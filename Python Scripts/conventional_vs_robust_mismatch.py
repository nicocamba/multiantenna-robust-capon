import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ============================================================
# 1. PARAMETERS: REAL SOI AT 2 DEG, NOMINAL MODEL AT 0 DEG
# ============================================================

M = 10
d = 0.5

theta_s = 0.0                 # Nominal SOI direction supplied to SCB/RCB
theta_s_true = 2.0            # Actual SOI direction used to generate data
theta_i1 = 60.0
theta_i2 = 80.0

Ps_dB = 10.0
Pi_dB = 20.0
Pn_dB = 0.0

Ps = 10**(Ps_dB / 10)
Pi = 10**(Pi_dB / 10)
Pn = 10**(Pn_dB / 10)

epsilon_values = [5]

N_pattern = [10, 100, 8000]
rng = np.random.default_rng(42)


# ============================================================
# 2. STEERING VECTOR
# Always return a column vector of shape (M, 1)
# ============================================================

def steering_vector(theta_deg):
    m = np.arange(M)
    theta = np.deg2rad(theta_deg)

    return np.exp(
        -1j * 2 * np.pi * d * m * np.sin(theta)
    ).reshape(M, 1)


a0 = steering_vector(theta_s)              # Nominal steering vector given to SCB/RCB
a_s = steering_vector(theta_s_true)        # Actual steering vector used in R and snapshots
a_i1 = steering_vector(theta_i1)
a_i2 = steering_vector(theta_i2)


# ============================================================
# 3. THEORETICAL COVARIANCE (uses the REAL SOI steering vector at 2 deg)
# ============================================================

R_theory = (
    Ps * (a_s @ a_s.conj().T)
    + Pi * (a_i1 @ a_i1.conj().T)
    + Pi * (a_i2 @ a_i2.conj().T)
    + Pn * np.eye(M, dtype=complex)
)

R_theory = (R_theory + R_theory.conj().T) / 2


# ============================================================
# 4. RANDOM SNAPSHOTS (generated with the REAL SOI at 2 deg)
# ============================================================

def complex_gaussian(shape, power):
    return np.sqrt(power / 2) * (
        rng.standard_normal(shape)
        + 1j * rng.standard_normal(shape)
    )


def generate_snapshots(N):
    s = complex_gaussian((N,), Ps)
    i1 = complex_gaussian((N,), Pi)
    i2 = complex_gaussian((N,), Pi)
    noise = complex_gaussian((M, N), Pn)

    X = (
        a_s @ s.reshape(1, N)
        + a_i1 @ i1.reshape(1, N)
        + a_i2 @ i2.reshape(1, N)
        + noise
    )

    return X


def sample_covariance(X):
    R = X @ X.conj().T / X.shape[1]
    return (R + R.conj().T) / 2


# ============================================================
# 5. STANDARD CAPON BEAMFORMING (SCB)
# ============================================================

def scb(R, a0):
    a0 = np.asarray(a0).reshape(M, 1)

    u = np.linalg.solve(R, a0)

    denominator = float(
        np.real((a0.conj().T @ u)[0, 0])
    )

    if denominator <= 0:
        raise ValueError("SCB: denominator is not positive.")

    w = u / denominator
    P_hat = 1.0 / denominator

    return w, P_hat


# ============================================================
# 6. ROBUST CAPON BEAMFORMING (RCB)
# ============================================================

def rcb(R, a0, epsilon, debug=False):
    """
    Robust Capon Beamforming (Li, Stoica & Wang, 2003).

    Implementation based on equations (24), (27)-(30), and (32).

    Returns
    -------
    w       : RCB weight vector, shape (M,)
    P_hat   : estimated SOI power
    a_hat   : estimated steering vector, shape (M,)
    lam     : Lagrange multiplier
    """
    R = np.asarray(R, dtype=complex)
    a0 = np.asarray(a0, dtype=complex).reshape(-1)

    M = a0.size

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    if R.shape != (M, M):
        raise ValueError("R must have shape (M, M).")

    # --------------------------------------------------
    # 1. Eigendecomposition of R
    # --------------------------------------------------
    R = (R + R.conj().T) / 2

    # eigh returns eigenvalues in ascending order.
    gamma, U = np.linalg.eigh(R)

    # Sort eigenvalues in descending order and reorder U accordingly.
    idx = np.argsort(gamma)[::-1]
    gamma = gamma[idx]
    U = U[:, idx]

    # Check that R is positive definite.
    if np.any(gamma <= 0):
        raise ValueError(
            "R must be positive definite for this implementation."
        )

    # Coordinates of the nominal steering vector.
    z = U.conj().T @ a0

    # --------------------------------------------------
    # 2. Lambda bounds: equation (27)
    # --------------------------------------------------
    sqrt_epsilon = np.sqrt(epsilon)
    norm_a0 = np.linalg.norm(a0)

    if norm_a0 <= sqrt_epsilon:
        raise ValueError(
            "The conditions for the lambda bounds in equation (27) are not met: "
            "||a0|| must be greater than sqrt(epsilon)."
        )

    lambda_min = (
        (norm_a0 - sqrt_epsilon)
        / (gamma[0] * sqrt_epsilon)
    )

    lambda_max_1 = np.sqrt(
        np.sum(np.abs(z)**2 / gamma**2) / epsilon
    )

    lambda_max_2 = (
        (norm_a0 - sqrt_epsilon)
        / (gamma[-1] * sqrt_epsilon)
    )

    lambda_max = min(lambda_max_1, lambda_max_2)

    if not (
        np.isfinite(lambda_min)
        and np.isfinite(lambda_max)
        and 0 <= lambda_min < lambda_max
    ):
        raise RuntimeError(
            "Invalid lambda interval for equation (27)."
        )

    # --------------------------------------------------
    # 3. Solve equation (24) within the bounds from equation (27)
    # --------------------------------------------------
    def equation(lam):
        return (
            np.sum(
                np.abs(z)**2
                / (1.0 + lam * gamma)**2
            )
            - epsilon
        )

    f_min = equation(lambda_min)
    f_max = equation(lambda_max)

    if f_min * f_max > 0:
        raise RuntimeError(
            "Equation (24) does not change sign within "
            "the bounds from equation (27). Check that "
            "R, a0, and epsilon are defined as in the paper."
        )

    lam = brentq(
        equation,
        lambda_min,
        lambda_max,
        xtol=1e-15,
        rtol=1e-14
    )

    # --------------------------------------------------
    # 4. Estimated steering vector: equation (29)
    # --------------------------------------------------
    a_hat = a0 - U @ (z / (1.0 + lam * gamma))

    # --------------------------------------------------
    # 5. Intermediate power: equation (30)
    # --------------------------------------------------
    denominator = np.sum(
        gamma * np.abs(z)**2
        / (
            lam**(-2)
            + 2.0 * lam**(-1) * gamma
            + gamma**2
        )
    )

    if denominator <= 0 or not np.isfinite(denominator):
        raise RuntimeError(
            "Invalid denominator when estimating RCB power."
        )

    P_raw = 1.0 / denominator

    # --------------------------------------------------
    # 6. Scale correction: equation (28)
    # --------------------------------------------------
    correction = np.linalg.norm(a_hat)**2 / M
    P_hat = P_raw * correction

    # --------------------------------------------------
    # 7. RCB weights: equation (32)
    # --------------------------------------------------
    R_inv_a = np.linalg.solve(R, a_hat)
    denom_w = np.vdot(a_hat, R_inv_a)

    if abs(denom_w) < 1e-15:
        raise RuntimeError(
            "Denominator is effectively zero when calculating the weights."
        )

    w = R_inv_a / denom_w
    w = np.asarray(w).reshape(M)

    # --------------------------------------------------
    # 8. Debug output
    # --------------------------------------------------
    if debug:
        distance_sq = np.linalg.norm(a_hat - a0)**2

        print("\n========== DEBUG RCB ==========")
        print(f"M                         = {M}")
        print(f"epsilon                   = {epsilon:.8g}")
        print(f"lambda_min (Eq. 27)       = {lambda_min:.8e}")
        print(f"lambda_max (Eq. 27)       = {lambda_max:.8e}")
        print(f"lambda                    = {lam:.8e}")
        print(f"Eq. (24) residual         = {equation(lam):.8e}")
        print(f"||a_hat - a0||^2          = {distance_sq:.8e}")
        print(f"||a_hat||^2               = {np.linalg.norm(a_hat)**2:.8e}")
        print(f"Correction                = {correction:.8e}")
        print(f"Denominator Eq. (30)      = {denominator:.8e}")
        print(f"P_raw                     = {P_raw:.8e}")
        print(f"P_hat                     = {P_hat:.8e}")
        print(f"shape(w)                  = {w.shape}")
        print("================================\n")

    return w, P_hat, a_hat, lam


# ============================================================
# 7. BEAMPATTERN
# Explicit dimensions: A is (M, number_of_angles)
# ============================================================

angles = np.linspace(-90, 90, 1801)

A = np.stack(
    [steering_vector(theta).ravel() for theta in angles],
    axis=1
)

assert A.shape == (M, len(angles)), (
    f"Unexpected A dimensions: {A.shape}"
)


def beampattern_db(w):
    # Force w to have shape (M,), avoiding accidental broadcasting
    w = np.asarray(w).reshape(M)

    # Result must have exactly one value per angle
    response = np.abs(w.conj() @ A)**2

    assert response.shape == (len(angles),), (
        f"Unexpected response dimensions: {response.shape}"
    )

    maximum = np.max(response)

    if maximum <= 0:
        raise ValueError("Beampattern has a non-positive maximum.")

    response = response / maximum

    return 10 * np.log10(
        np.maximum(response, 1e-16)
    )


# ============================================================
# 8. BEAMPATTERNS ONLY (no power plots / no power Monte Carlo)
# Theoretical covariance and N = 10, 100, 8000
# ============================================================

for epsilon in epsilon_values:

    cases = [("Theoretical covariance R", R_theory)]

    # One realization for each snapshot count
    for N in N_pattern:
        X = generate_snapshots(N)
        R_hat = sample_covariance(X)
        cases.append((f"Sample covariance, N={N}", R_hat))

    fig, axes = plt.subplots(
        1, 4,
        figsize=(19, 4.8),
        sharey=True
    )

    for ax, (title, R) in zip(axes, cases):

        w_scb, _ = scb(R, a0)
        w_rcb, _, _, _ = rcb(R, a0, epsilon)

        # Both functions return one value per angle
        B_scb = beampattern_db(w_scb)
        B_rcb = beampattern_db(w_rcb)

        ax.plot(angles, B_scb, label="SCB")
        ax.plot(angles, B_rcb, "--", label="RCB")

        ax.axvline(
            theta_s_true,
            color="black",
            linestyle=":",
            label="Actual SOI"
        )
        ax.axvline(
            theta_s,
            color="black",
            linestyle="--",
            label="Nominal SOI"
        )

        ax.axvline(
            theta_i1,
            color="gray",
            linestyle=":",
            label="Interference"
        )

        ax.axvline(
            theta_i2,
            color="gray",
            linestyle="--",
            label="Interference"
        )

        ax.set_title(title)
        ax.set_xlabel("Angle (degrees)")
        ax.set_xlim(-90, 90)
        ax.set_ylim(-120, 3)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Normalized response (dB)")
    axes[0].legend(fontsize=8)

    fig.suptitle(
        f"SCB vs RCB beampatterns: SOI mismatch, nominal={theta_s}°, actual={theta_s_true}°, epsilon={epsilon}"
    )

    plt.tight_layout()
    plt.show()


print("\nFinished.")



# ============================================================
# 9. COMPARISON OF DIFFERENT MISMATCHES AND EPSILON VALUES
# Beampatterns only: no power plots
# ============================================================

# Each scenario specifies:
#   - Actual SOI direction
#   - Epsilon used by RCB
#
# The nominal direction remains theta_s = 0 degrees.

scenarios = [
    {
        "title": "Mismatch = 2°, epsilon = 1",
        "theta_true": 2.0,
        "epsilon": 1.0,
    },
    {
        "title": "Mismatch = 5°, epsilon = 1",
        "theta_true": 5.0,
        "epsilon": 1.0,
    },
    {
        "title": "Mismatch = 5°, epsilon = 8",
        "theta_true": 5.0,
        "epsilon": 8.0,
    },
]

# Keep the nominal steering vector fixed at 0 degrees.
a0 = steering_vector(theta_s)

# Steering vectors for the two interferers.
a_i1 = steering_vector(theta_i1)
a_i2 = steering_vector(theta_i2)

# Generate the plots for each scenario.
for scenario in scenarios:

    theta_true = scenario["theta_true"]
    epsilon_test = scenario["epsilon"]

    print(
        f"\nScenario: actual SOI = {theta_true} degrees, "
        f"nominal SOI = {theta_s} degrees, "
        f"epsilon = {epsilon_test}"
    )

    # Actual steering vector of the SOI.
    a_s_test = steering_vector(theta_true)

    # --------------------------------------------------------
    # Theoretical covariance for this scenario
    # --------------------------------------------------------

    R_theory_test = (
        Ps * (a_s_test @ a_s_test.conj().T)
        + Pi * (a_i1 @ a_i1.conj().T)
        + Pi * (a_i2 @ a_i2.conj().T)
        + Pn * np.eye(M, dtype=complex)
    )

    R_theory_test = (
        R_theory_test + R_theory_test.conj().T
    ) / 2

    # --------------------------------------------------------
    # Generate snapshots using the ACTUAL SOI direction.
    # The algorithms will still receive a0, corresponding
    # to the NOMINAL direction.
    # --------------------------------------------------------

    cases_test = [
        ("Theoretical covariance", R_theory_test)
    ]

    for N in N_pattern:

        # Generate the SOI at theta_true, not at theta_s.
        s = complex_gaussian((N,), Ps)
        i1 = complex_gaussian((N,), Pi)
        i2 = complex_gaussian((N,), Pi)
        noise = complex_gaussian((M, N), Pn)

        X = (
            a_s_test @ s.reshape(1, N)
            + a_i1 @ i1.reshape(1, N)
            + a_i2 @ i2.reshape(1, N)
            + noise
        )

        R_hat_test = sample_covariance(X)

        cases_test.append(
            (f"Sample covariance, N={N}", R_hat_test)
        )

    # --------------------------------------------------------
    # Plot SCB and RCB beampatterns
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(cases_test),
        figsize=(19, 4.8),
        sharey=True
    )

    for ax, (case_title, R_test) in zip(axes, cases_test):

        # Both algorithms receive the NOMINAL steering vector.
        w_scb, _ = scb(R_test, a0)

        w_rcb, _, _, _ = rcb(
            R_test,
            a0,
            epsilon_test
        )

        B_scb = beampattern_db(w_scb)
        B_rcb = beampattern_db(w_rcb)

        ax.plot(
            angles,
            B_scb,
            label="SCB"
        )

        ax.plot(
            angles,
            B_rcb,
            "--",
            label="RCB"
        )

        # Actual direction of the desired signal.
        ax.axvline(
            theta_true,
            color="black",
            linestyle=":",
            label="Actual SOI"
        )

        # Direction assumed by both algorithms.
        ax.axvline(
            theta_s,
            color="black",
            linestyle="--",
            label="Nominal SOI"
        )

        # Interference directions.
        ax.axvline(
            theta_i1,
            color="gray",
            linestyle=":",
            label="Interference"
        )

        ax.axvline(
            theta_i2,
            color="gray",
            linestyle="--",
            label="Interference"
        )

        ax.set_title(case_title)
        ax.set_xlabel("Angle (degrees)")
        ax.set_xlim(-10, 15)
        ax.set_ylim(-120, 3)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Normalized response (dB)")
    axes[0].legend(fontsize=8)

    fig.suptitle(
        f"SCB vs RCB: actual SOI = {theta_true}°, "
        f"nominal SOI = {theta_s}°, "
        f"epsilon = {epsilon_test}"
    )

    plt.tight_layout()
    plt.show()

print("\nMismatch and epsilon comparison finished.")
