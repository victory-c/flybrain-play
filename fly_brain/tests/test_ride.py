"""Closed-loop pieces on a toy brain and the bare bike (no connectome needed; CPU is fine)."""
import math

import numpy as np
import torch

from bike.dynamics import Peloton
from brain.loop import BrainLoop
from tests.test_sim import toy_brain


def main():
    # 1. the bike: self-stable at 6 m/s, falls at 3 m/s, and a PD rider holds it at 3 m/s in gusts
    torch.manual_seed(0)
    bikes = Peloton(4, seed=1)
    bikes.reset(6.0, phi0_deg=3.0)
    for _ in range(500):
        bikes.step(torch.zeros(4), torch.zeros(4), torch.zeros(4), n_sub=10)
    lean = bikes.state[:, 4].abs().max().item()
    print(f"6 m/s, hands off, 5 s: max |lean| {math.degrees(lean):.2f} deg, fallen {int(bikes.done.sum())}/4")
    assert not bikes.done.any() and lean < math.radians(1.0)

    bikes.reset(3.0, phi0_deg=3.0)
    for _ in range(500):
        bikes.step(torch.zeros(4), torch.zeros(4), torch.zeros(4), n_sub=10)
    print(f"3 m/s, hands off, 5 s: fallen {int(bikes.done.sum())}/4")
    assert bikes.done.all()

    bikes.reset(3.0, phi0_deg=3.0, gust_nm=15.0)
    for _ in range(1000):
        bikes.step(bikes.pd_oracle(), torch.full((4,), 60.0), torch.zeros(4), n_sub=10)
    s = bikes.state
    print(f"3 m/s, PD rider, gusts 15 Nm, 10 s: fallen {int(bikes.done.sum())}/4, |y| {s[:, 1].abs().max():.2f} m, "
          f"v {s[:, 3].mean():.2f} m/s, x {s[:, 0].mean():.1f} m")
    assert not bikes.done.any() and s[:, 1].abs().max() < 1.0

    # 2. the loop: per-trial rates, persistence across chunks, agreement with simulate()'s toy result
    b = toy_brain([(0, 1, 100.0), (0, 2, -100.0)], 3)
    loop = BrainLoop(b, n_run=4, stim_idx=[0], readout_idx=[0, 1, 2], seed=0)
    loop.set_rates(np.array([[100.0], [100.0], [0.0], [30.0]]))
    total = torch.zeros(4, 3, dtype=torch.int32)
    for _ in range(100):
        counts, pop = loop.run(10.0)
        total += counts
    hz = total.float().numpy()
    print("toy loop 1 s, rates Hz per trial:\n", hz)
    assert 90 < hz[0, 0] <= 101 and 90 < hz[1, 0] <= 101, hz
    assert hz[2].sum() == 0
    assert 20 < hz[3, 0] < 40
    assert hz[0, 1] > 0 and hz[0, 2] == 0
    assert loop.t == 10000
    print("all ride checks passed")


if __name__ == "__main__":
    main()
