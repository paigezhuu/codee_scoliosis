import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import array

t_initial, t_final = 0, 50
S0f, S0t, I_L0, I_H0, T0, R0, C0 = 0.94, 0.06, 0, 0, 0.00, 0.00, 0.00

fusion_var = {
  "alpha_f": 0.05,
  "lamba_f": 0.5,
  "p_f": 0.9,
  "rho_f": 0.04,
  "q_f": 0.5,
  "tau_Lf": 0.6,
  "tau_Hf": 0.7
}

tethering_var = {
  "alpha_t": 0.01,
  "lamba_t": 0.8,
  "p_t": 0.95,
  "rho_t": 0.1,
  "q_t": 0.6,
  "tau_Lt": 0.3,
  "tau_Ht": 0.1,
  "beta_t": 0.4
}

Y0_combined = [S0f, I_L0, I_H0, T0, R0, C0, S0t, I_L0, I_H0, T0, R0, C0]
t_span = (t_initial , t_final)


def sir_two_cohorts(t, y, fusion_var, tethering_var):
  S_f, I_Lf, I_Hf, T_f, R_f, C_f = y[:6]
  S_t, I_Lt, I_Ht, T_t, R_t, C_t = y[6:]

  alpha_f = fusion_var["alpha_f"]
  lamba_f = fusion_var["lamba_f"]
  p_f = fusion_var["p_f"]
  rho_f = fusion_var["rho_f"]
  q_f = fusion_var["q_f"]
  tau_Lf = fusion_var["tau_Lf"]
  tau_Hf = fusion_var["tau_Hf"]

  alpha_t = tethering_var["alpha_t"]
  lamba_t = tethering_var["lamba_t"]
  p_t = tethering_var["p_t"]
  rho_t = tethering_var["rho_t"]
  q_t = tethering_var["q_t"]
  tau_Lt = tethering_var["tau_Lt"]
  tau_Ht = tethering_var["tau_Ht"]
  beta_t = tethering_var["beta_t"]


  # Fusion cohort
  dydt_f = [
    -alpha_f * S_f + beta_t * C_t,
    q_f * alpha_f * S_f + rho_f * R_f - tau_Lf * I_Lf,
    (1-q_f) * alpha_f * S_f - tau_Hf * I_Hf,
    tau_Lf * I_Lf + tau_Hf * I_Hf - lamba_f * T_f,
    p_f * lamba_f * T_f - rho_f * R_f,
    (1-p_f) * lamba_f * T_f
  ]

  # Tethering cohort
  dydt_t = [
    -alpha_t * S_t,
    q_t * alpha_t * S_t + rho_t * R_t - tau_Lt * I_Lt,
    (1-q_t) * alpha_t * S_t - tau_Ht * I_Ht,
    tau_Lt * I_Lt + tau_Ht * I_Ht - lamba_t * T_t,
    p_t * lamba_t * T_t - rho_t * R_t,
    (1-p_t) * lamba_t * T_t - beta_t * C_t
  ]

  return dydt_f + dydt_t

sol = solve_ivp(sir_two_cohorts, t_span, Y0_combined, args=(fusion_var, tethering_var), dense_output=True)
t = np.linspace(t_initial, t_final, 500)
y = sol.sol(t)

y_fusion = y[:6]
y_tethering = y[6:]


# Fusion and Tethering side-by-side
fig, (ax_f, ax_t) = plt.subplots(1, 2, figsize=(20, 6))

# Fusion
ax_f.plot(t, y_fusion.T, label=["$S_f(t)$", "$I^L_{f}(t)$", "$I^H_{f}(t)$", "$T_f(t)$", "$R_f(t)$", "$C_f(t)$"])
ax_f.legend(shadow=True)
ax_f.set_xlim(left=-0.01)
ax_f.set_ylim(0,.805)
ax_f.set_xlabel("$t$ (months)", fontsize = 14)
ax_f.set_ylabel("Population Proportion", fontsize = 14)
ax_f.set_title("$SI^LI^HRT$-$C$ Model - Fusion Cohort", fontsize = 16)


# Tethering
ax_t.plot(t, y_tethering.T, label=["$S_t(t)$", "$I^L_{t}(t)$", "$I^H_{t}(t)$", "$T_t(t)$", "$R_t(t)$", "$C_t(t)$"])
ax_t.legend(shadow=True)
ax_t.set_xlim(left=-0.01)
ax_t.set_ylim(0,.0605)
ax_t.set_xlabel("$t$ (months)", fontsize = 14)
ax_t.set_ylabel("Population Proportion", fontsize = 14)
ax_t.set_title("$SI^LI^HRT$-$C$ Model - Tethering Cohort", fontsize = 16)

plt.tight_layout()
plt.show()