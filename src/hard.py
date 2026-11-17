"""HARD problem: best smooth route (cubic B-spline, SLSQP, terrain risk).

    min_p  J = alpha * L(p) + beta * S(p)
    L(p) = int_0^1 ||p'(t)|| dt                       [m]
    S(p) = int_0^1 R(p(t)) ||p'(t)|| dt               [m]   (R is dimensionless)
    s.t.   d_min <= d(p(t)) <= d_max                  (seabed depth proxy)
           s(p(t)) <= s_max                           (terrain slope)
           |kappa(t)| <= kappa_max                    (route curvature)
           kappa = (x'y'' - y'x'') / (x'^2 + y'^2)^{3/2}
           control points inside the grid box

p(t) is a clamped cubic B-spline with control points A, K interior points and B.
Decision variables: the (x, y) of the K interior control points (unit square
coordinates).  Objective and constraints are sampled at M points.

Two SLSQP strategies are run and recorded (see `solve_hard`):
  1. "direct": the pointwise constraints are handed to SLSQP as 4 inequality
     vectors (4*M rows).  This is the literal formulation.
  2. "penalty": the constraints enter a QUADRATIC EXTERIOR PENALTY function
     (mu * ||max(g, 0)||^2), minimised by SLSQP over the control-point bounds
     only, with a continuation on mu (1e2 -> 1e4 -> 1e6).  For a finite mu the
     penalised minimiser is only approximately feasible; a feasible point of the
     penalised subproblem is therefore NOT a KKT point of the constrained
     problem and not an optimum of it.  Every penalised result is checked
     against the true constraints and against the nominal limits at a much
     denser sample set before it is reported.

Both use scipy.optimize.minimize(method="SLSQP").  Either way the reported
optimum is LOCAL and a candidate: the problem is non-convex, no global
optimality is claimed, and the solver's exit status only says the local
optimality conditions of the (penalised) subproblem were satisfied.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import BSpline, RectBivariateSpline
from scipy.optimize import minimize

from .features import fill_nan_nearest

DEGREE = 3


# --------------------------------------------------------------------------
# smooth field interpolators + required checks (spec Sec. 6)
# --------------------------------------------------------------------------
class SmoothFields:
    """Spline interpolators of z, d, s, R on the metric grid ("cubic" or "linear").

    Masked (land / NoData) cells are filled with the NEAREST VALID OCEAN value
    before the interpolator is built (nearest-neighbour fill, no interpolation
    across the mask), so land elevations cannot leak into coastal cells.  The
    unfilled arrays are retained: feasibility is always judged on the ORIGINAL
    data, and `distance_to_mask` gives the metric clearance of any point to the
    nearest masked cell so that interpolation artefacts near land/NoData can be
    detected rather than assumed away.

    `method="cubic"` is a tensor-product cubic spline (kx = ky = 3);
    `method="linear"` is bilinear (kx = ky = 1) and is used as the
    overshoot-free reference in the interpolation comparison.
    """

    def __init__(self, x, y, z, d, s, R, mask, k: int | None = None, method: str = "cubic"):
        self.x, self.y, self.mask = x, y, mask
        self.z_raw, self.d_raw, self.s_raw, self.R_raw = z, d, s, R
        self.method = method
        self.k = int(k if k is not None else (3 if method == "cubic" else 1))
        self.fields = {}
        self.raw_valid = {}
        for name, arr in (("z", z), ("d", d), ("s", s), ("R", R)):
            arr = np.asarray(arr, float)
            # mask-aware: land and NoData are removed BEFORE the nearest-value
            # fill, so the interpolant is built purely from ocean samples and can
            # never carry a land elevation (or a negative depth) into the
            # solution.  The unfilled arrays are kept for validation.
            valid = (~np.asarray(mask, bool)) & np.isfinite(arr)
            self.raw_valid[name] = np.where(valid, arr, np.nan)
            self.fields[name] = fill_nan_nearest(self.raw_valid[name])
        self._spl = {n: RectBivariateSpline(y, x, v, kx=self.k, ky=self.k, s=0)
                     for n, v in self.fields.items()}
        # metric distance (m) from every node to the nearest masked cell
        from scipy.ndimage import distance_transform_edt
        dx_m = float(x[1] - x[0])
        dy_m = float(y[1] - y[0])
        self._dist_mask = distance_transform_edt(~mask, sampling=(dy_m, dx_m))

    def __call__(self, field: str, X, Y):
        return self._spl[field].ev(Y, X)

    def grad(self, field: str, X, Y):
        f = self._spl[field]
        return f.ev(Y, X, dx=1), f.ev(Y, X, dy=1)

    def distance_to_mask(self, X, Y) -> np.ndarray:
        """Metric clearance to the nearest masked cell (m), bilinear interpolation."""
        spl = RectBivariateSpline(self.y, self.x, self._dist_mask, kx=1, ky=1, s=0)
        return spl.ev(Y, X)

    def inside_mask(self, X, Y) -> np.ndarray:
        """True where the enclosing node cell is masked (land / NoData)."""
        j = np.clip(np.round((np.asarray(X) - self.x[0]) / (self.x[1] - self.x[0])).astype(int),
                    0, len(self.x) - 1)
        i = np.clip(np.round((np.asarray(Y) - self.y[0]) / (self.y[1] - self.y[0])).astype(int),
                    0, len(self.y) - 1)
        return self.mask[i, j]

    # -- required interpolator checks -------------------------------------
    def check(self, n_probe: int = 20000, seed: int = 0) -> dict:
        """Overshoot / leakage diagnostics against the source grid.

        Probes are split into INTERIOR cells (all four surrounding nodes carry
        valid ocean data) and BOUNDARY cells (at least one neighbour is land or
        NoData).  An interpolant can only be expected to stay inside the corner
        bracket in the interior; next to the mask the nearest-value fill means
        the bracket is not defined, so those probes are reported separately
        instead of being averaged into a single misleading number.
        """
        rng = np.random.default_rng(seed)
        ny, nx = self.mask.shape
        i = rng.integers(1, ny - 2, n_probe)
        j = rng.integers(1, nx - 2, n_probe)
        fi, fj = rng.random(n_probe), rng.random(n_probe)
        X = self.x[j] + fj * (self.x[1] - self.x[0])
        Y = self.y[i] + fi * (self.y[1] - self.y[0])
        out = {"method": self.method, "spline_order": self.k, "n_probe": int(n_probe)}
        for name in ("z", "d", "s", "R"):
            raw = self.raw_valid[name]
            interp = self(name, X, Y)
            corners = np.stack([raw[i + a, j + b] for a in (0, 1) for b in (0, 1)])
            finite = np.isfinite(corners)
            n_valid = finite.sum(axis=0)
            interior = n_valid == 4
            any_valid = n_valid > 0
            lo = np.where(finite, corners, np.inf).min(axis=0)
            hi = np.where(finite, corners, -np.inf).max(axis=0)
            lo[~any_valid] = np.nan
            hi[~any_valid] = np.nan
            overshoot = np.maximum(np.maximum(interp - hi, lo - interp), 0.0)

            def stats(sel):
                v = overshoot[sel & np.isfinite(overshoot)]
                if v.size == 0:
                    return {"n": 0, "max": float("nan"), "p99": float("nan"),
                            "mean": float("nan")}
                return {"n": int(v.size), "max": float(v.max()),
                        "p99": float(np.percentile(v, 99)), "mean": float(v.mean())}

            span = np.nanmax(raw) - np.nanmin(raw)
            si = stats(interior)
            sb = stats(any_valid & ~interior)
            out[name] = {
                "interior_max_overshoot": si["max"],
                "interior_p99_overshoot": si["p99"],
                "interior_mean_overshoot": si["mean"],
                "interior_n_probe": si["n"],
                "boundary_max_overshoot": sb["max"],
                "boundary_p99_overshoot": sb["p99"],
                "boundary_n_probe": sb["n"],
                # kept for backwards compatibility with earlier result files
                "max_abs_overshoot": si["max"],
                "p99_abs_overshoot": si["p99"],
                "relative_max_overshoot": (si["max"] / span) if span else float("nan"),
                "n_probe": si["n"] + sb["n"],
            }
        d_interp = self("d", X, Y)
        # a cubic interpolant can undershoot between nodes; a negative depth would
        # imply land, so the same probes are run through a linear (monotone on the
        # cell) depth interpolant as a reference.  Neither result is "fixed" by
        # clipping: a negative depth is INFEASIBLE, and the constraints reject it
        # because d_min > 0.
        lin_d = RectBivariateSpline(self.y, self.x, self.fields["d"], kx=1, ky=1, s=0)
        d_lin = lin_d.ev(Y, X)
        out["depth_leakage"] = {
            "min_interp_depth_m": float(np.min(d_interp)),
            "n_probe_implying_land": int(np.sum(d_interp < 0)),
            "n_probe_on_masked_cell": int(np.sum(self.inside_mask(X, Y))),
            "linear_min_interp_depth_m": float(np.min(d_lin)),
            "linear_n_probe_implying_land": int(np.sum(d_lin < 0)),
            "policy": ("an interpolated depth <= 0 means the interpolant implied land; it is "
                       "treated as infeasible (the constraint d >= d_min with d_min > 0 "
                       "rejects it), never clipped to a small positive value"),
            "note": ("depth is interpolated from ocean samples only; the two columns show "
                     "the cubic and the linear depth interpolants on identical probes."),
        }
        return out


# --------------------------------------------------------------------------
# B-spline path parameterisation (precomputed basis)
# --------------------------------------------------------------------------
BC = "clamped"


class Spline:
    """Clamped cubic B-spline: A, K interior control points, B (both ends fixed).

    `bc="clamped"` is the only boundary condition implemented: the knot vector
    repeats (k+1) times at 0 and at 1 and the first/last control points are
    pinned to A and B, so the curve interpolates the two fixed endpoints with a
    prescribed tangent direction.  `config.yaml::hard.spline_bc` is checked
    against this, so the configuration cannot advertise an unimplemented choice.
    """

    def __init__(self, A, B, K: int, extent, origin, M: int = 200, bc: str = BC):
        if bc != BC:
            raise ValueError(f"spline_bc={bc!r} is not implemented; only {BC!r} is")
        self.bc = bc
        self.A, self.B, self.K = np.asarray(A, float), np.asarray(B, float), K
        self.extent, self.origin = np.asarray(extent, float), np.asarray(origin, float)
        n = K + 2
        k = DEGREE
        n_int = n - k
        self.n = n
        self.knots = np.concatenate([[0.0] * (k + 1),
                                     np.arange(1, n_int) / n_int, [1.0] * (k + 1)])
        self.t_knots = np.linspace(0.0, 1.0, n)
        self.set_M(M)

    def set_M(self, M: int):
        """Precompute basis matrices at the M sample points (t is fixed)."""
        self.M = M
        self.t = np.linspace(0.0, 1.0, M)
        eye = np.eye(self.n)
        b = BSpline(self.knots, eye, DEGREE)
        self.N0 = b(self.t)                                   # (M, n)
        self.N1 = b.derivative(1)(self.t)
        self.N2 = b.derivative(2)(self.t)

    # -- coordinates ---------------------------------------------------
    def control_points(self, v):
        K = self.K
        u = np.concatenate([[self.A[0]], self.origin[0] + v[:K] * self.extent[0],
                            [self.B[0]]])
        w = np.concatenate([[self.A[1]], self.origin[1] + v[K:] * self.extent[1],
                            [self.B[1]]])
        return np.column_stack([u, w])

    def evaluate(self, v, order: int = 0):
        P = self.control_points(v)
        X, Y = self.N0 @ P[:, 0], self.N0 @ P[:, 1]
        if order == 0:
            return X, Y
        dX, dY = self.N1 @ P[:, 0], self.N1 @ P[:, 1]
        if order == 1:
            return X, Y, dX, dY
        return X, Y, dX, dY, self.N2 @ P[:, 0], self.N2 @ P[:, 1]

    # -- starts ---------------------------------------------------------
    def straight(self):
        f = self.t_knots[1:-1]
        P = self.A[None, :] + f[:, None] * (self.B - self.A)[None, :]
        return self._to_v(P)

    def from_polyline(self, xy, n_fit: int = 400):
        """Least-squares B-spline fit of a polyline (well-behaved warm start).

        Sampling the polyline at the knots is not enough: a B-spline does not
        pass through its interior control points, so the curve would cut corners
        across steep terrain.  Fitting the control points by least squares keeps
        the starting curve close to the (feasible) Dijkstra route.
        """
        seg = np.hypot(*np.diff(xy, axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        s /= s[-1]
        tt = np.linspace(0.0, 1.0, n_fit)
        X = np.interp(tt, s, xy[:, 0])
        Y = np.interp(tt, s, xy[:, 1])
        Nm = BSpline(self.knots, np.eye(self.n), DEGREE)(tt)
        free = slice(1, self.n - 1)
        cx = np.linalg.lstsq(Nm[:, free], X - Nm[:, 0] * self.A[0] - Nm[:, -1] * self.B[0],
                             rcond=None)[0]
        cy = np.linalg.lstsq(Nm[:, free], Y - Nm[:, 0] * self.A[1] - Nm[:, -1] * self.B[1],
                             rcond=None)[0]
        return self._to_v(np.column_stack([cx, cy]))

    def _to_v(self, P):
        u = (P[:, 0] - self.origin[0]) / self.extent[0]
        w = (P[:, 1] - self.origin[1]) / self.extent[1]
        return np.clip(np.concatenate([u, w]), 0.0, 1.0)


def route_curvature(X, Y, dX, dY, ddX, ddY):
    den = np.maximum((dX ** 2 + dY ** 2) ** 1.5, 1e-9)
    return (dX * ddY - dY * ddX) / den


# --------------------------------------------------------------------------
# problem wrapper
# --------------------------------------------------------------------------
@dataclass
class HardResult:
    v: np.ndarray
    xy: np.ndarray
    J: float
    L: float
    S: float
    status: int
    message: str
    nit: int
    runtime_s: float
    start_label: str
    solver: str
    max_kappa: float
    max_slope: float
    depth_range: tuple[float, float]
    max_violation: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


class HardProblem:
    def __init__(self, fields: SmoothFields, spline: Spline, d_min, d_max, s_max,
                 kappa_max, alpha=1.0, beta=1.0, M=200, depth_scale=100.0,
                 margin: float = 0.0, n_constraint: int | None = None):
        self.f, self.spl = fields, spline
        # nominal limits are what the report and the validation use; the solver
        # enforces a slightly tighter set (margin) which absorbs the residual
        # that pointwise collocation leaves *between* its own samples.
        self.s_max_nominal, self.kappa_max_nominal = s_max, kappa_max
        self.margin = float(margin)
        self.d_min, self.d_max = d_min, d_max
        self.s_max = s_max * (1.0 - self.margin)
        self.kappa_max = kappa_max * (1.0 - self.margin)
        self.M_collocation = int(n_constraint or M)
        self.alpha, self.beta = alpha, beta
        self.depth_scale = depth_scale
        self.n_var = 2 * spline.K
        self.bounds = [(0.0, 1.0)] * self.n_var
        self.set_M(int(n_constraint or M))

    def set_M(self, M: int):
        self.M = M
        self.spl.set_M(M)

    # ---- path quantities ---------------------------------------------
    def state(self, v):
        X, Y, dX, dY, ddX, ddY = self.spl.evaluate(v, order=2)
        speed = np.hypot(dX, dY)
        d = self.f("d", X, Y)
        s = self.f("s", X, Y)
        R = self.f("R", X, Y)
        kap = route_curvature(X, Y, dX, dY, ddX, ddY)
        return X, Y, speed, d, s, R, kap

    def terms(self, v, M: int | None = None):
        """(L, S, J, speed, fields...) at M samples; M defaults to self.M."""
        if M is None or M == self.M:
            X, Y, speed, d, s, R, kap = self.state(v)
            t = self.spl.t
        else:
            sp = self.spl
            saved = sp.M
            sp.set_M(M)
            X, Y, speed, d, s, R, kap = self.state(v)
            t = sp.t
            sp.set_M(saved)
        L = float(np.trapezoid(speed, t))
        S = float(np.trapezoid(R * speed, t))
        return L, S, (self.alpha * L + self.beta * S) / 1000.0, dict(
            X=X, Y=Y, speed=speed, d=d, s=s, R=R, kappa=kap, t=t)

    # ---- objective ----------------------------------------------------
    def objective(self, v) -> float:                     # J in km
        return self.terms(v)[2]

    def constraints(self, v) -> np.ndarray:
        """All four constraint vectors stacked (normalised, <=0 is feasible)."""
        _, _, _, d, s, _, kap = self.state(v)
        return np.concatenate([
            (self.d_min - d) / self.depth_scale,
            (d - self.d_max) / self.depth_scale,
            (s - self.s_max) / self.s_max,
            (np.abs(kap) - self.kappa_max) / self.kappa_max,
        ])

    def c_depth_lo(self, v):
        return (self.d_min - self.state(v)[3]) / self.depth_scale

    def c_depth_hi(self, v):
        return (self.state(v)[3] - self.d_max) / self.depth_scale

    def c_slope(self, v):
        return (self.state(v)[4] - self.s_max) / self.s_max

    def c_kappa(self, v):
        return (np.abs(self.state(v)[6]) - self.kappa_max) / self.kappa_max

    def violation_nominal(self, v) -> dict:
        """Violation measured against the NOMINAL limits (what is reported).

        The solver enforces `s_max * (1 - margin)` and `kappa_max * (1 - margin)`
        so that the validated route stays inside the nominal limits between
        collocation samples.  Feasibility of a candidate is therefore decided
        here, against the nominal limits, not against the tightened ones.
        """
        _, _, _, d, s, _, kap = self.state(v)
        return {
            "depth_lo": float(max(0.0, np.max((self.d_min - d) / self.depth_scale))),
            "depth_hi": float(max(0.0, np.max((d - self.d_max) / self.depth_scale))),
            "slope": float(max(0.0, np.max((s - self.s_max_nominal) / self.s_max_nominal))),
            "kappa": float(max(0.0, np.max((np.abs(kap) - self.kappa_max_nominal)
                                           / self.kappa_max_nominal))),
        }

    def violation(self, v) -> dict:
        g = self.constraints(v)
        n = self.M
        return {
            "depth_lo": float(max(0.0, g[0:n].max())),
            "depth_hi": float(max(0.0, g[n:2 * n].max())),
            "slope": float(max(0.0, g[2 * n:3 * n].max())),
            "kappa": float(max(0.0, g[3 * n:].max())),
        }

    # ---- SLSQP runs -----------------------------------------------------
    def _result(self, res, v0, label, solver, runtime):
        L, S, J, _ = self.terms(res.x)
        X, Y, speed, d, s, R, kap = self.state(res.x)
        return HardResult(
            v=res.x, xy=np.column_stack([X, Y]), J=J, L=L, S=S,
            status=int(res.status), message=str(res.message), nit=int(getattr(res, "nit", 0)),
            runtime_s=runtime, start_label=label, solver=solver,
            max_kappa=float(np.max(np.abs(kap))), max_slope=float(np.max(s)),
            depth_range=(float(d.min()), float(d.max())),
            max_violation=self.violation(res.x),
            meta={"nfev": int(getattr(res, "nfev", 0)), "M": self.M,
                  "moved": float(np.linalg.norm(res.x - v0)),
                  "max_violation_nominal": self.violation_nominal(res.x),
                  "feasible_nominal": bool(max(self.violation_nominal(res.x).values()) <= 1e-3)},
        )

    def scale_vector(self) -> np.ndarray:
        """Natural magnitude of each constraint row (4 blocks of M).

        depth: 100 m of water; slope: the slope limit itself; curvature: the
        curvature limit itself.  A violation that reaches the limit is then O(1)
        and the penalty can compete with an objective measured in metres.
        """
        rep = lambda v: np.full(self.M, float(v))
        return np.concatenate([rep(self.depth_scale), rep(self.depth_scale),
                               rep(self.s_max), rep(self.kappa_max)])

    def scaled_groups(self):
        w = 1.0 / self.scale_vector()
        m = self.M
        cuts = [slice(0, m), slice(m, 2 * m), slice(2 * m, 3 * m), slice(3 * m, 4 * m)]
        return [(f, w[c]) for f, c in zip((self.c_depth_lo, self.c_depth_hi,
                                           self.c_slope, self.c_kappa), cuts)]

    def solve_direct(self, v0, label, ftol=1e-8, maxiter=300, scaled=False):
        """Literal formulation: 4*M pointwise constraints handed to SLSQP.

        scaled=True keeps the same feasible set but divides each constraint
        block by its natural magnitude, which materially improves SLSQP's
        behaviour when one block is in 1/m and another in 1e-4 /m units.
        """
        if scaled:
            cons = [{"type": "ineq", "fun": (lambda x, f=f, w=w: f(x) * w)}
                    for f, w in self.scaled_groups()]
        else:
            cons = [{"type": "ineq", "fun": self.c_depth_lo},
                    {"type": "ineq", "fun": self.c_depth_hi},
                    {"type": "ineq", "fun": self.c_slope},
                    {"type": "ineq", "fun": self.c_kappa}]
        t0 = time.perf_counter()
        res = minimize(self.objective, np.asarray(v0, float), method="SLSQP",
                       bounds=self.bounds, constraints=cons,
                       options={"ftol": ftol, "maxiter": maxiter, "disp": False})
        return self._result(res, v0, label, "slsqp_direct", time.perf_counter() - t0)

    def solve_penalty(self, v0, label, mus=(1e2, 1e4, 1e6), ftol=1e-9, maxiter=300,
                      scaled=False):
        """Quadratic exterior-penalty continuation, SLSQP over the bounds only.

        phi(x) = alpha*L(x) + beta*S(x) + mu * sum_i max(g_i(x), 0)^2, with
        mu = 1e2 -> 1e4 -> 1e6.  For a finite mu the minimiser is only
        approximately feasible; reaching feasibility does not make it a KKT
        point of the constrained problem, and the constrained multipliers are
        never formed here.

        scaled=False (default) uses the constraint values in their native units,
        exactly as the formulation states them; scaled=True divides each block by
        its natural magnitude.  Both were tested: in native units the mu ladder
        reaches a feasible point from the grid warm start for every beta in the
        sweep, whereas the scaled variant stalls in infeasible local minima when
        the slope block dominates (recorded in the report).
        """
        v = np.asarray(v0, float).copy()
        t0 = time.perf_counter()
        hist, status, message, nit, nfev = [], 0, "", 0, 0
        w = 1.0 / self.scale_vector() if scaled else np.ones(4 * self.M)
        for mu in mus:
            def phi(x, mu=mu):
                g = np.maximum(self.constraints(x), 0.0) * w
                return self.objective(x) + mu * float(np.sum(g ** 2))
            res = minimize(phi, v, method="SLSQP", bounds=self.bounds,
                           options={"ftol": ftol, "maxiter": maxiter, "disp": False})
            v = res.x
            status, message, nit = res.status, res.message, res.nit
            nfev += getattr(res, "nfev", 0)
            hist.append({"mu": mu, "J_km": float(self.objective(v)),
                         "max_violation": float(max(self.violation(v).values())),
                         "status": int(res.status), "nit": int(res.nit),
                         "nfev": int(getattr(res, "nfev", 0))})
        out = self._result(res, v0, label, "slsqp_penalty", time.perf_counter() - t0)
        out.meta["continuation"] = hist
        out.meta["nfev"] = nfev
        out.meta["mus"] = list(mus)
        return out


# --------------------------------------------------------------------------
# multi-start driver
# --------------------------------------------------------------------------
def smooth_polyline(xy, window_m: float = 250.0, n: int = 2000):
    """Moving-average smoothing of a polyline, returned at uniform arc length.

    Used to build a *curvature-feasible* warm start: the raw 8-neighbour grid
    route has corner curvature ~1e-2 1/m, 15x the limit, which is a bad place for
    a local solver to begin.
    """
    xy = np.asarray(xy, float)
    if len(xy) < 3:
        return xy
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    t = np.linspace(0.0, s[-1], n)
    X = np.interp(t, s, xy[:, 0])
    Y = np.interp(t, s, xy[:, 1])
    k = max(3, int(round(window_m / (s[-1] / n))) | 1)
    ker = np.ones(k) / k
    pad = k // 2
    X = np.convolve(np.pad(X, pad, mode="edge"), ker, mode="valid")
    Y = np.convolve(np.pad(Y, pad, mode="edge"), ker, mode="valid")
    X[0], Y[0] = xy[0]
    X[-1], Y[-1] = xy[-1]
    return np.column_stack([X, Y])


def make_starts(prob: HardProblem, easy_xy, cfg: dict, n_random: int | None = None):
    sc = cfg["hard"]["slsqp"]
    starts = []
    if easy_xy is not None and len(easy_xy) > 2:
        # two warm starts from the same grid route: the exact least-squares fit
        # of the (zig-zag) node polyline, and a 250 m smoothed version of it.
        # Both are kept: empirically the zig-zag fit escapes the penalty's
        # infeasible local minima more reliably, the smoothed one gives a better
        # conditioned start for the curvature constraint.
        starts.append(("warm_start_dijkstra", prob.spl.from_polyline(easy_xy)))
        starts.append(("warm_start_dijkstra_smoothed",
                       prob.spl.from_polyline(smooth_polyline(easy_xy))))
    base = prob.spl.straight()
    starts.append(("straight_line", base.copy()))
    n_random = int(sc["n_random_starts"] if n_random is None else n_random)
    rng = np.random.default_rng(sc["random_seed"])
    K = prob.spl.K
    su = float(sc["perturb_sigma_cells"]) * 450.0 / prob.spl.extent[0]
    sv = float(sc["perturb_sigma_cells"]) * 450.0 / prob.spl.extent[1]
    for i in range(n_random):
        v = base.copy()
        v[:K] = np.clip(v[:K] + rng.normal(0.0, su, size=K), 0.02, 0.98)
        v[K:] = np.clip(v[K:] + rng.normal(0.0, sv, size=K), 0.02, 0.98)
        starts.append((f"random_{i:02d}", v))
    return starts


def independent_curvature(spline: "Spline", v, n: int = 2001) -> dict:
    """Recompute route curvature from scratch, without the optimiser's matrices.

    The control points are rebuilt into a fresh `scipy.interpolate.BSpline`
    object on the same knot vector, differentiated by scipy, and the curvature
    is formed with the formula in the specification:
        kappa = (x'y'' - y'x'') / (x'^2 + y'^2)^{3/2}
    The result is compared with the value the optimiser evaluated from its own
    precomputed basis matrices, so a mistake in that basis (or in the sign
    convention) cannot go unnoticed.
    """
    P = np.asarray(spline.control_points(v), float)
    bs = BSpline(spline.knots, P, DEGREE, extrapolate=False)
    t = np.linspace(0.0, 1.0, n)
    d1 = np.asarray(bs.derivative(1)(t))
    d2 = np.asarray(bs.derivative(2)(t))
    x1, y1 = d1[:, 0], d1[:, 1]
    x2, y2 = d2[:, 0], d2[:, 1]
    kappa = (x1 * y2 - y1 * x2) / np.maximum((x1 ** 2 + y1 ** 2) ** 1.5, 1e-12)
    # the optimiser's own evaluation, on its own sample grid
    X, Y, dX, dY, ddX, ddY = spline.evaluate(v, order=2)
    k_opt = route_curvature(X, Y, dX, dY, ddX, ddY)
    speed = np.hypot(x1, y1)
    i_max = int(np.argmax(np.abs(kappa)))
    return {
        "kappa_max": float(np.max(np.abs(kappa))),
        "kappa_min_radius_m": float(1.0 / np.max(np.abs(kappa))) if np.max(np.abs(kappa)) > 0 else float("inf"),
        "kappa_max_location_frac": float(t[i_max]),
        "kappa_signed_at_max": float(kappa[i_max]),
        "optimiser_kappa_max": float(np.max(np.abs(k_opt))),
        "agreement_vs_optimiser": float(abs(np.max(np.abs(kappa)) - np.max(np.abs(k_opt)))),
        "n_samples": int(n),
        "speed_min_m_per_unit_t": float(speed.min()),
        "endpoint_speed_m_per_unit_t": [float(speed[0]), float(speed[-1])],
        "note": ("sample-independent recomputation on a rebuilt BSpline; the "
                 "parameter speed is not uniform, so the maximum is located by "
                 "sampling t uniformly, not by arc length"),
    }


def account_runs(runs: list, starts: list) -> dict:
    """Honest multi-start accounting: executions vs distinct initial curves.

    `solve_hard` performs one SLSQP execution per start for the direct strategy
    and then re-runs a *subset* of those starts with the penalty continuation,
    so the number of solver executions is larger than the number of independent
    initial curves and must not be reported as "N independent starts".
    """
    direct = [r for r in runs if r.solver == "slsqp_direct"]
    penalty = [r for r in runs if r.solver == "slsqp_penalty"]
    ok = [r for r in runs if np.isfinite(r.J)]

    # Feasibility is judged against the NOMINAL limits, never the margined ones:
    # the 2 % margin is an enforcement device, not a relaxation of the problem.
    def _worst(r):
        v = r.meta.get("max_violation_nominal")
        return max((v if v is not None else r.max_violation).values())

    feas = [r for r in ok if _worst(r) <= 1e-3]
    infeas = [r for r in ok if _worst(r) > 1e-3]
    conv = [r for r in ok if r.status == 0]
    Jf = np.array([r.J for r in feas])
    Ji = np.array([r.J for r in infeas])
    return {
        "n_distinct_initial_curves": len(starts),
        "start_labels": [lab for lab, _ in starts],
        "n_solver_executions": len(runs),
        "n_direct_executions": len(direct),
        "n_penalty_executions": len(penalty),
        "penalty_start_labels": sorted({r.start_label for r in penalty}),
        "n_converged_status0": len(conv),
        "n_feasible": len(feas),
        "n_infeasible": len(infeas),
        "feasible_start_labels": sorted({r.start_label for r in feas}),
        "feasible_solvers": sorted({r.solver for r in feas}),
        "J_feasible_min": float(Jf.min()) if len(Jf) else float("nan"),
        "J_feasible_max": float(Jf.max()) if len(Jf) else float("nan"),
        "J_feasible_spread": float(Jf.max() - Jf.min()) if len(Jf) > 1 else 0.0,
        "n_infeasible_with_J": len(infeas),
        "J_infeasible_range": [float(Ji.min()), float(Ji.max())] if len(Ji) else [],
        "infeasible_used_for_optimum": False,
        "note": ("Distinct initial curves and solver executions are different "
                 "quantities: the direct strategy runs once per start, and the "
                 "penalty continuation re-runs a subset of those same starts. "
                 "Only feasible results enter the reported objective spread; "
                 "infeasible candidates are listed but never used as an optimum. "
                 "No global optimality is claimed."),
    }


def nominal_worst(r: HardResult) -> float:
    """Worst NOMINAL constraint violation of a run (0 = feasible)."""
    v = r.meta.get("max_violation_nominal")
    if v is None:
        v = r.max_violation
    return float(max(v.values())) if v else float("inf")


def is_feasible(r: HardResult, tol: float = 1e-3) -> bool:
    return bool(np.isfinite(r.J) and nominal_worst(r) <= tol)


def feasible_score(r: HardResult, tol: float = 1e-3):
    """Sort key: feasible candidates first, then by objective."""
    return (not is_feasible(r, tol), r.J if np.isfinite(r.J) else np.inf)


def solve_hard(prob: HardProblem, starts, strategy: str = "auto",
               tol: float = 1e-3, penalty_starts: int = 3):
    """Run SLSQP from every start.  Returns (best_result, all_runs).

    strategy = "direct"  : literal constrained SLSQP for every start.
    strategy = "penalty" : penalty continuation for every start.
    strategy = "auto"    : direct for every start, then penalty continuation for
                           the `penalty_starts` most promising starts (warm start,
                           straight line, best direct run); the adopted optimum is
                           the best FEASIBLE result of either strategy.
    """
    runs: list[HardResult] = []
    if strategy in ("direct", "auto"):
        for label, v0 in starts:
            try:
                runs.append(prob.solve_direct(v0, label))
            except Exception as exc:
                runs.append(HardResult(np.asarray(v0, float), np.empty((0, 2)), np.nan,
                                       np.nan, np.nan, -1, f"exception: {exc}", 0, 0.0,
                                       label, "slsqp_direct", 0.0, 0.0, (np.nan, np.nan)))
    if strategy in ("penalty", "auto"):
        cand = list(starts)
        if strategy == "auto" and runs:
            # Penalty continuation is applied to (i) every canonical warm start
            # and (ii) the best direct candidates.  The direct stage can fail on
            # every start (it often does), so the warm starts are always included
            # -- otherwise a bad direct stage would silently remove the only
            # starts that reach feasibility.
            canonical = [lab for lab in ("warm_start_dijkstra",
                                         "warm_start_dijkstra_smoothed",
                                         "straight_line")]
            pool = {lab: v for lab, v in starts if lab in canonical}
            good = sorted([r for r in runs if np.isfinite(r.J)],
                          key=lambda r: feasible_score(r))
            for r in good:
                pool.setdefault(r.start_label, r.v)
            cand = list(pool.items())[:max(penalty_starts, len(canonical))]
        for label, v0 in cand:
            try:
                runs.append(prob.solve_penalty(v0, label))
            except Exception as exc:
                runs.append(HardResult(np.asarray(v0, float), np.empty((0, 2)), np.nan,
                                       np.nan, np.nan, -1, f"exception: {exc}", 0, 0.0,
                                       label, "slsqp_penalty", 0.0, 0.0, (np.nan, np.nan)))
    pool = [r for r in runs if np.isfinite(r.J)]
    if not pool:
        raise RuntimeError("every SLSQP start failed")
    best = min(pool, key=lambda r: feasible_score(r, tol))
    return best, runs
