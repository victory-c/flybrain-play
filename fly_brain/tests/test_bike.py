"""Whipple matrices against Meijaard et al. 2007, and the Tarmac SL9's self-stable speed range."""
import numpy as np

from bike.tarmac import BENCHMARK_EXPECTED, benchmark, eigen_speeds, tarmac_sl9, whipple_matrices


def main():
    m = whipple_matrices(benchmark())
    for k in ("M", "C1", "K0", "K2"):
        err = np.abs(m[k] - np.array(BENCHMARK_EXPECTED[k])).max()
        print(f"{k}: max |error| {err:.2e}")
        assert err < 1e-8, (k, m[k])
    wv, cv = eigen_speeds(m)
    print(f"benchmark weave {wv:.3f} (paper {BENCHMARK_EXPECTED['weave_speed']:.3f}), capsize {cv:.3f}"
          f" (paper {BENCHMARK_EXPECTED['capsize_speed']:.3f}) m/s")
    assert abs(wv - BENCHMARK_EXPECTED["weave_speed"]) < 2e-3
    assert abs(cv - BENCHMARK_EXPECTED["capsize_speed"]) < 2e-3

    t = whipple_matrices(tarmac_sl9())
    wv, cv = eigen_speeds(t)
    print(f"Tarmac SL9 + 70 kg rider: {t['mT']:.1f} kg, trail {t['c'] * 1000:.1f} mm, "
          f"self-stable {wv:.2f}..{cv:.2f} m/s = {wv * 3.6:.1f}..{cv * 3.6:.1f} km/h")
    assert 0.05 < t["c"] < 0.06
    assert wv is not None and 3.0 < wv < 8.0
    print("all bike checks passed")


if __name__ == "__main__":
    main()
