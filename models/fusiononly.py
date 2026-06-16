import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np

t_initial, t_final = 0, 10
S0, I_L0, I_H0, T0, R0, C0 = 1, 0, 0, 0.00, 0.00, 0.00

alpha = 0.4 # infection rate
lamba = 0.8 # rate of moving out of recovery
p = 0.95 # probability of recovery
rho = 0.1 # relapse rate into low pain
q = 0.4 # probability of pain being low pain
tau_L = 0.3 # treatment rate for low pain
tau_H = 0.7 # treatment rate for high pain

Y0 = [S0, I_L0, I_H0, T0, R0, C0]
t_span = (t_initial , t_final)


def sir(t, y, alpha, lamba, p, rho, q, tau_L, tau_H):
  S, I_L, I_H, T, R, C = y
  return [
    -alpha * S,
    q * alpha * S + rho * R - tau_L * I_L,
    (1-q) * alpha * S - tau_H * I_H,
    tau_L * I_L + tau_H * I_H - lamba * T,
    p * lamba * T - rho * R,
    (1-p) * lamba * T
  ]

sol = solve_ivp(sir, t_span, Y0, args=(alpha, lamba, p, rho, q, tau_L, tau_H), dense_output=True)
t = np.linspace(t_initial, t_final, 100)
y = sol.sol(t)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(t, y.T, label=["$S(t)$", "$I_L(t)$", "$I_H(t)$", "$R(t)$", "$T(t)$", "$C(t)$"])
ax.legend(shadow=True)
ax.set_xlim(left=-0.01)
ax.set_ylim(0, 1)
ax.set_xlabel("$t$ (days)")
ax.set_ylabel("Population")
ax.set_title("SIIRT-C")