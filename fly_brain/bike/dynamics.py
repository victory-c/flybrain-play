"""B copies of the Tarmac, integrated together in torch (so they can sit on the GPU next to the brain).

State per bike (8): x, y, psi, v, phi, delta, phi_dot, delta_dot
  x, y      rear contact point on the road (m), x forward, y to the right
  psi       heading (rad, right positive)
  v         forward speed (m/s)
  phi       lean (rad, right positive)        delta      steer (rad, right positive)

Lateral dynamics are the linearised Whipple equations (bike/tarmac.py), so a bike is "fallen" once
|phi| passes phi_max (30 deg) rather than actually hitting the road. Longitudinal: pedal power,
aerodynamic drag, rolling resistance, grade and a brake. A side gust is an Ornstein-Uhlenbeck roll
torque; it is what the rider has to fight below the self-stable speed.
"""
import math

import torch

from .tarmac import G, LONGITUDINAL, tarmac_sl9, whipple_matrices

STATE = ["x", "y", "psi", "v", "phi", "delta", "phi_dot", "delta_dot"]


class Peloton:
    def __init__(self, n, params=None, device="cpu", dt=1e-3, phi_max_deg=30.0, delta_max=1.0, grade=0.0, seed=0):
        p = params or tarmac_sl9()
        m = whipple_matrices(p)
        self.p, self.mats, self.n, self.dt = p, m, n, dt
        self.dev = torch.device(device)
        f = lambda a: torch.tensor(a, dtype=torch.float32, device=self.dev)
        self.Minv, self.C1, self.K0, self.K2 = f(torch.linalg.inv(torch.tensor(m["M"])).numpy()), f(m["C1"]), f(m["K0"]), f(m["K2"])
        self.mT, self.w, self.c, self.cl = m["mT"], m["w"], m["c"], math.cos(m["lam"])
        self.phi_max, self.delta_max, self.grade = math.radians(phi_max_deg), delta_max, grade
        self.long = dict(LONGITUDINAL)
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.state = torch.zeros(n, 8, device=self.dev)
        self.done = torch.zeros(n, dtype=torch.bool, device=self.dev)
        self.gust = torch.zeros(n, device=self.dev)
        self.psi_dot = torch.zeros(n, device=self.dev)
        self.gust_nm, self.gust_tau = 0.0, 1.0

    def reset(self, v0, phi0_deg=2.0, gust_nm=0.0, gust_tau=1.0):
        s = torch.zeros(self.n, 8, device=self.dev)
        s[:, 3] = v0
        s[:, 4] = torch.randn(self.n, generator=self.gen, device=self.dev) * math.radians(phi0_deg)
        s[:, 6] = torch.randn(self.n, generator=self.gen, device=self.dev) * math.radians(phi0_deg)
        self.state = s
        self.done.zero_()
        self.gust.zero_()
        self.psi_dot.zero_()
        self.gust_nm, self.gust_tau = gust_nm, gust_tau
        return self.state

    @torch.no_grad()
    def step(self, steer_torque, power, brake, n_sub=10):
        """Advance n_sub steps of dt. steer_torque (Nm), power (W), brake (m/s^2 of deceleration): (n,) tensors."""
        dt, L = self.dt, self.long
        s = self.state
        x, y, psi, v, q, qd = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4:6], s[:, 6:8]
        for _ in range(n_sub):
            if self.gust_nm > 0:
                noise = torch.randn(self.n, generator=self.gen, device=self.dev)
                self.gust = self.gust + (-self.gust / self.gust_tau) * dt + self.gust_nm * math.sqrt(2 * dt / self.gust_tau) * noise
            f = torch.stack([self.gust, steer_torque], 1)
            rhs = f - v[:, None] * (qd @ self.C1.T) - (G * (q @ self.K0.T) + (v * v)[:, None] * (q @ self.K2.T))
            qdd = rhs @ self.Minv.T
            qd = qd + qdd * dt
            q = q + qd * dt
            # handlebar stops
            over = q[:, 1].abs() > self.delta_max
            q = torch.stack([q[:, 0], q[:, 1].clamp(-self.delta_max, self.delta_max)], 1)
            qd = torch.stack([qd[:, 0], torch.where(over & (qd[:, 1] * q[:, 1] > 0), torch.zeros_like(qd[:, 1]), qd[:, 1])], 1)
            # longitudinal
            f_ped = power * L["drivetrain"] / v.clamp(min=0.5)
            f_res = 0.5 * L["rho"] * L["CdA"] * v * v + L["Crr"] * self.mT * G + self.mT * G * math.sin(self.grade)
            v = (v + ((f_ped - f_res) / self.mT - brake) * dt).clamp(min=0.0)
            psi_dot = (v * q[:, 1] + self.c * qd[:, 1]) * self.cl / self.w
            psi = psi + psi_dot * dt
            x = x + v * torch.cos(psi) * dt
            y = y + v * torch.sin(psi) * dt
        new = torch.cat([x[:, None], y[:, None], psi[:, None], v[:, None], q, qd], 1)
        keep = self.done[:, None]
        self.state = torch.where(keep, s, new)
        self.psi_dot = torch.where(self.done, self.psi_dot, psi_dot)
        fell = (self.state[:, 4].abs() > self.phi_max) & ~self.done
        self.done |= fell
        return fell

    def pd_oracle(self, kp=40.0, kd=8.0, ky=0.25, kpsi=1.0, lane_y=0.0):
        """A conventional rider: lean toward the lane centre, steer into the lean. Returns steer torque (Nm)."""
        s = self.state
        phi_des = -(ky * (s[:, 1] - lane_y) + kpsi * s[:, 2]).clamp(-0.2, 0.2)
        return kp * (s[:, 4] - phi_des) + kd * s[:, 6]
