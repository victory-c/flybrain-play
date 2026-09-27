"""The bicycle: a Specialized S-Works Tarmac SL9 (56 cm) plus rider as a Whipple-Carvallo model.

The linearised equations of motion (Meijaard, Papadopoulos, Ruina & Schwab, Proc. R. Soc. A 2007)
for lean phi and steer delta at forward speed v are

    M q'' + v C1 q' + (g K0 + v^2 K2) q = [T_phi, T_delta],   q = [phi, delta]

The four 2x2 matrices follow from 25 physical parameters through the formulas in that paper's
section 5. `benchmark()` is the paper's bicycle; tests/test_bike.py checks our matrices against
the published values to 1e-8.

Tarmac SL9 numbers come from Specialized's 56 cm geometry chart (wheelbase, head angle, fork
offset, BB drop), the wheels from the Roval Rapide CLX III with 28 mm tyres, and the masses
from a 6.9 kg complete bike. The rider (70 kg by default) sits in a hoods position. Everything
about the rider's mass distribution is an ASSUMPTION, scaled from the benchmark rider; the
Tarmac is treated as rigid and its tyres as knife edges, like every Whipple bicycle.

Sign conventions: x forward, z down. phi > 0 leans to the right, delta > 0 steers right.
"""
import math

import numpy as np

G = 9.81


def whipple_matrices(p):
    """p: dict of physical parameters (see benchmark()). Returns dict(M, C1, K0, K2, plus derived)."""
    w, c, lam = p["w"], p["c"], p["lam"]
    rR, mR, IRxx, IRyy = p["rR"], p["mR"], p["IRxx"], p["IRyy"]
    xB, zB, mB, IBxx, IBxz, IBzz = p["xB"], p["zB"], p["mB"], p["IBxx"], p["IBxz"], p["IBzz"]
    xH, zH, mH, IHxx, IHxz, IHzz = p["xH"], p["zH"], p["mH"], p["IHxx"], p["IHxz"], p["IHzz"]
    rF, mF, IFxx, IFyy = p["rF"], p["mF"], p["IFxx"], p["IFyy"]
    sl, cl = math.sin(lam), math.cos(lam)

    mT = mR + mB + mH + mF
    xT = (xB * mB + xH * mH + w * mF) / mT
    zT = (-rR * mR + zB * mB + zH * mH - rF * mF) / mT
    ITxx = IRxx + IBxx + IHxx + IFxx + mR * rR ** 2 + mB * zB ** 2 + mH * zH ** 2 + mF * rF ** 2
    ITxz = IBxz + IHxz - mB * xB * zB - mH * xH * zH + mF * w * rF
    IRzz, IFzz = IRxx, IFxx
    ITzz = IRzz + IBzz + IHzz + IFzz + mB * xB ** 2 + mH * xH ** 2 + mF * w ** 2

    mA = mH + mF
    xA = (xH * mH + w * mF) / mA
    zA = (zH * mH - rF * mF) / mA
    IAxx = IHxx + IFxx + mH * (zH - zA) ** 2 + mF * (rF + zA) ** 2
    IAxz = IHxz - mH * (xH - xA) * (zH - zA) + mF * (w - xA) * (rF + zA)
    IAzz = IHzz + IFzz + mH * (xH - xA) ** 2 + mF * (w - xA) ** 2
    uA = (xA - w - c) * cl - zA * sl
    IAll = mA * uA ** 2 + IAxx * sl ** 2 + 2 * IAxz * sl * cl + IAzz * cl ** 2
    IAlx = -mA * uA * zA + IAxx * sl + IAxz * cl
    IAlz = mA * uA * xA + IAxz * sl + IAzz * cl

    mu = c / w * cl
    SR, SF = IRyy / rR, IFyy / rF
    ST = SR + SF
    SA = mA * uA + mu * mT * xT

    M = np.array([[ITxx, IAlx + mu * ITxz],
                  [IAlx + mu * ITxz, IAll + 2 * mu * IAlz + mu ** 2 * ITzz]])
    K0 = np.array([[mT * zT, -SA],
                   [-SA, -SA * sl]])
    K2 = np.array([[0.0, (ST - mT * zT) / w * cl],
                   [0.0, (SA + SF * sl) / w * cl]])
    C1 = np.array([[0.0, mu * ST + SF * cl + ITxz / w * cl - mu * mT * zT],
                   [-(mu * ST + SF * cl), IAlz / w * cl + mu * (SA + ITzz / w * cl)]])
    return {"M": M, "C1": C1, "K0": K0, "K2": K2, "mT": mT, "xT": xT, "zT": zT, "w": w, "c": c, "lam": lam}


def benchmark():
    """The Meijaard et al. 2007 benchmark bicycle (Table 1)."""
    return dict(w=1.02, c=0.08, lam=math.pi / 10,
                rR=0.3, mR=2.0, IRxx=0.0603, IRyy=0.12,
                xB=0.3, zB=-0.9, mB=85.0, IBxx=9.2, IBxz=2.4, IBzz=2.8,
                xH=0.9, zH=-0.7, mH=4.0, IHxx=0.05892, IHxz=-0.00756, IHzz=0.00708,
                rF=0.35, mF=3.0, IFxx=0.1405, IFyy=0.28)


BENCHMARK_EXPECTED = {  # Meijaard et al. 2007, section 5.3 (M, C1, K0, K2) and Table 2 (speeds)
    "M": [[80.81722, 2.31941332208709], [2.31941332208709, 0.29784188199686]],
    "C1": [[0.0, 33.86641391492494], [-0.85035641456978, 1.68540397397560]],
    "K0": [[-80.95, -2.59951685249872], [-2.59951685249872, -0.80329488458618]],
    "K2": [[0.0, 76.59734589573222], [0.0, 2.65431523794604]],
    "weave_speed": 4.29238253634111,
    "capsize_speed": 6.02426201538837,
}


def tarmac_sl9(rider_mass=70.0):
    """S-Works Tarmac SL9, 56 cm, Roval Rapide CLX III 700x28, rider on the hoods.

    Geometry (Specialized chart, 56 cm): wheelbase 981 mm, head angle 73.5 deg, fork offset 44 mm,
    BB drop 72 mm. Trail follows from those: (rF sin(lam) - offset) / cos(lam) = 55 mm.
    Masses: frame module + seatpost + drivetrain 3.0 kg, fork + bars + stem 1.2 kg, wheels 1.2 / 1.5 kg
    with tyres, rotors and cassette; bike 6.9 kg. The rider's hands and forearms (1.3 kg) turn with
    the bars. Inertias are the benchmark rider's scaled by mass.
    """
    head_deg, offset, rF = 73.5, 0.044, 0.336
    lam = math.radians(90.0 - head_deg)
    c = (rF * math.sin(lam) - offset) / math.cos(lam)
    kB = (rider_mass + 3.0) / 85.0
    kH = 2.5 / 4.0
    return dict(w=0.981, c=c, lam=lam,
                rR=rF, mR=1.5, IRxx=0.072, IRyy=0.144,
                xB=0.38, zB=-0.97, mB=rider_mass + 3.0, IBxx=9.2 * kB, IBxz=2.4 * kB, IBzz=2.8 * kB,
                xH=0.87, zH=-0.72, mH=2.5, IHxx=0.05892 * kH, IHxz=-0.00756 * kH, IHzz=0.00708 * kH,
                rF=rF, mF=1.2, IFxx=0.058, IFyy=0.115)


# Longitudinal model of the same bike. CdA on the hoods, Crr of 28 mm race tyres on tarmac.
LONGITUDINAL = {"CdA": 0.30, "Crr": 0.0035, "rho": 1.225, "drivetrain": 0.975,
                "brake_max": 6.0}  # m/s^2


def state_matrix(mats, v, g=G):
    """A(v) of x' = A x for x = [phi, delta, phi', delta']."""
    M, C1, K0, K2 = mats["M"], mats["C1"], mats["K0"], mats["K2"]
    Minv = np.linalg.inv(M)
    A = np.zeros((4, 4))
    A[:2, 2:] = np.eye(2)
    A[2:, :2] = -Minv @ (g * K0 + v ** 2 * K2)
    A[2:, 2:] = -Minv @ (v * C1)
    return A


def eigen_speeds(mats, v_max=15.0, dv=0.001):
    """Weave and capsize speeds: the self-stable range where every eigenvalue has Re < 0."""
    vs = np.arange(0.0, v_max, dv)
    stable = np.array([np.linalg.eigvals(state_matrix(mats, v)).real.max() < 0 for v in vs])
    if not stable.any():
        return None, None
    i0 = int(np.argmax(stable))
    after = np.where(~stable[i0:])[0]
    i1 = i0 + int(after[0]) if len(after) else None
    return float(vs[i0]), (float(vs[i1]) if i1 is not None else None)


if __name__ == "__main__":
    for name, p in [("benchmark", benchmark()), ("Tarmac SL9 + 70 kg", tarmac_sl9())]:
        m = whipple_matrices(p)
        wv, cv = eigen_speeds(m)
        print(f"{name}: mass {m['mT']:.1f} kg, CoM x {m['xT']:.3f} z {m['zT']:.3f}, trail {m['c'] * 1000:.0f} mm,"
              f" self-stable {wv:.2f} .. {cv:.2f} m/s ({wv * 3.6:.1f} .. {cv * 3.6:.1f} km/h)")
