"""Scalar F5 inner-domain implementation for the opt-in V44 study35 adapter.

Ported from the frozen reviewed scalar constructor; all rejection guards retained.
No simulator, Torch, transport or hardware calls. Geometry supplied by bound metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct

URDF_SHA = "cfae29e48a29d5d63db99a9f4443f05c0074d7100c890b227553d35a3aa6b474"
LEGS = ("front_left", "rear_left", "front_right", "rear_right")
JOINTS = tuple(leg+"_"+suffix for suffix in ("hip_abduction", "hip_flexion", "knee") for leg in LEGS)
MARGIN = 4.e-6  # rad: interior construction margin, never extra joint authority.
BRANCH_EPS = 1.e-7
MAX_X = .0175
TILT = math.radians(12.)


class EmptyDomain(ValueError):
    """No certified inner domain; never substitute a different lift or target."""


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def clamp(value, lower, upper):
    if lower > upper:
        raise EmptyDomain("Reversed interval")
    return min(upper, max(lower, value))


def acos(value):
    # Only transcendental roundoff is tolerated; physical reach is not clamped.
    if not math.isfinite(value) or abs(value) > 1.+8.e-15:
        raise EmptyDomain("Analytic inverse outside reachable height interval")
    return math.acos(clamp(value, -1., 1.))


def mm(a, b):
    return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def rz(angle):
    c,s = math.cos(angle),math.sin(angle)
    return ((c,-s,0.),(s,c,0.),(0.,0.,1.))


def rpy_rotation(values):
    r,p,y = values
    cr,sr,cp,sp,cy,sy = math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return ((cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr),
            (sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr),(-sp,cp*sr,cp*cr))


@dataclass(frozen=True)
class Domain:
    R: float
    origin: float
    branch: float
    lift: float
    length: float
    C: float
    D: float
    normal_C: float
    normal_D: float
    H: tuple
    K: tuple
    J: tuple
    S: tuple

    def mapped_hip(self, h, lift=None):
        d = self.lift if lift is None else lift
        t = self.branch*(h-self.origin)
        return self.origin+self.branch*acos(math.cos(t)+d/self.R)

    def inverse_hip(self, mapped):
        t = self.branch*(mapped-self.origin)
        return self.origin+self.branch*acos(math.cos(t)-self.lift/self.R)

    def hip_interval(self, total):
        lo = max(self.J[0],total-self.K[1])
        hi = min(self.J[1],total-self.K[0])
        image_lo,image_hi = self.mapped_hip(self.J[0]),self.mapped_hip(self.J[1])
        ml,mh = max(image_lo,total-self.K[1]),min(image_hi,total-self.K[0])
        if ml > mh or lo > hi:
            raise EmptyDomain("No conditional hip/knee interval")
        lo,hi = max(lo,self.inverse_hip(ml)),min(hi,self.inverse_hip(mh))
        if hi-lo <= 6.*MARGIN:
            raise EmptyDomain("Conditional hip interval lacks float32 interior")
        return lo+MARGIN,hi-MARGIN

    def contains(self, h, k):
        total = h+k
        if not self.S[0] <= total <= self.S[1]:
            return False
        lo,hi = self.hip_interval(total)
        return lo <= h <= hi

    def project(self, h, k):
        if not all(math.isfinite(v) and f32(v) == v for v in (h,k)):
            raise ValueError("Projection input must be exact finite recorded float32 targets")
        if self.contains(h,k):
            return (h,k),True  # exact original objects/bits; no round-trip.
        total = clamp(h+k,self.S[0]+MARGIN,self.S[1]-MARGIN)
        lo,hi = self.hip_interval(total)
        chosen = clamp(h,lo+MARGIN,hi-MARGIN)
        pair = (f32(chosen),f32(total-chosen))
        # This is evidence of numerical closure, not a catch/fallback path.
        self.assert_geometry(*pair)
        if not self.contains(*pair):
            raise ValueError("Float32 projection did not remain inside its certified inner domain")
        return pair,False

    def assert_geometry(self, h, k, lift=None):
        d = self.lift if lift is None else lift
        if not 0. <= d <= self.lift:
            raise ValueError("Only the certified amplitude interval is allowed")
        mapped = self.mapped_hip(h,d)
        mk = h+k-mapped
        dx = self.C*(math.cos(mapped)-math.cos(h))+self.D*(math.sin(mapped)-math.sin(h))
        nz = self.normal_C*math.cos(h+k)+self.normal_D*math.sin(h+k)
        if not (self.H[0]-MARGIN <= h <= self.H[1]+MARGIN and self.H[0]-MARGIN <= mapped <= self.H[1]+MARGIN
            and self.K[0]-MARGIN <= k <= self.K[1]+MARGIN and self.K[0]-MARGIN <= mk <= self.K[1]+MARGIN
            and k*mk > 0. and nz >= math.cos(TILT)-1.e-12 and abs(dx) <= MAX_X+1.e-12):
            raise ValueError("Projected result violates original bounds/tilt/X/sign")
        return {"mapped_hip_knee_rad":[mapped,mk], "signed_X_change_m":dx,
                "pitch_sum_error_rad":mapped+mk-h-k, "tilt_rad":acos(nz),
                "joint_increment_rad":mapped-h}


def build_domain(frame, abd, hip_center, knee_center, hip_bounds, knee_bounds, lift):
    """Analytic sufficient F for fixed context and every amplitude in[0,lift].

    Nonemptiness over *varying* reference/abduction contexts is a separate
    obligation. The companion calculator never promotes a finite lattice to
    that continuum certificate.
    """
    if not all(math.isfinite(v) for v in (abd,hip_center,knee_center,lift)) or not 0. <= lift <= .005:
        raise ValueError("Nonfinite context or unsupported lift")
    ar,hr,sign,length = frame
    P = mm(mm(ar,rz(abd)),hr)
    A,B = length*P[2][0],sign*length*P[2][1]
    R,phi = math.hypot(A,B),math.atan2(B,A)
    offset = math.atan2(math.sin(hip_center-phi),math.cos(hip_center-phi))
    if R <= 1.e-8 or abs(math.sin(offset)) <= BRANCH_EPS:
        raise EmptyDomain("Reference center has no pinned monotone branch")
    branch = 1. if offset > 0. else -1.
    origin = hip_center-offset
    H = (max(hip_bounds[0],hip_center-.15)+MARGIN,min(hip_bounds[1],hip_center+.15)-MARGIN)
    K = (max(knee_bounds[0],knee_center-.20)+MARGIN,min(knee_bounds[1],knee_center+.20)-MARGIN)
    K = (max(K[0],BRANCH_EPS),K[1]) if knee_center > 0. else (K[0],min(K[1],-BRANCH_EPS))
    delta = MAX_X/length-4.*MARGIN
    if H[0] >= H[1] or K[1]-K[0] < delta+8.*MARGIN:
        raise EmptyDomain("Original hip/knee box lacks certified angular width")
    tl,th = sorted((branch*(H[0]-origin),branch*(H[1]-origin)))
    tl,th = max(tl,BRANCH_EPS),min(th,math.pi-BRANCH_EPS)
    z = lift/R
    if tl >= th:
        raise EmptyDomain("Original hip box misses selected branch")
    # Both original t and raised t' must remain in the same hip interval.
    tl = max(tl,acos(math.cos(tl)-z))
    if z > 1.-math.cos(delta):
        ratio = z/(2.*math.sin(delta/2.))
        if ratio > 1.:
            raise EmptyDomain("Lift cannot satisfy conservative X certificate")
        angle = math.asin(ratio)
        tl,th = max(tl,delta/2.+angle),min(th,delta/2.+math.pi-angle)
    if th-tl <= 8.*MARGIN:
        raise EmptyDomain("No positive-width pre/post hip interval")
    J = tuple(sorted((origin+branch*tl,origin+branch*th)))
    nC,nD = -P[2][0],-sign*P[2][1]
    normal_radius = math.hypot(nC,nD)
    if normal_radius < math.cos(TILT-4.*MARGIN):
        raise EmptyDomain("Abduction leaves no <=12deg sole orientation")
    center_sum = hip_center+knee_center
    pitch_center = math.atan2(nD,nC)
    pitch_center += 2.*math.pi*round((center_sum-pitch_center)/(2.*math.pi))
    half = acos(math.cos(TILT-4.*MARGIN)/normal_radius)
    proto = Domain(R,origin,branch,lift,length,length*P[0][0],sign*length*P[0][1],nC,nD,H,K,J,(0.,0.))
    slo = max(K[0]+max(J[0],proto.mapped_hip(J[0])),pitch_center-half)+8.*MARGIN
    shi = min(K[1]+min(J[1],proto.mapped_hip(J[1])),pitch_center+half)-8.*MARGIN
    if shi-slo <= 2.*MARGIN:
        raise EmptyDomain("Tilt and coupled hip/knee sum interval do not intersect")
    result = Domain(R,origin,branch,lift,length,proto.C,proto.D,nC,nD,H,K,J,(slo,shi))
    # Finite endpoint arithmetic, not a search or action-dependent retry.
    for total in (slo,shi):
        result.hip_interval(total)
    return result


def project_pair(domain, h, k, common_lift):
    if common_lift <= 1.e-12:
        return (h,k),True  # Explicit inactive stance/park bypass; no tilt projection.
    if domain is None or domain.lift != common_lift:
        raise ValueError("Positive common envelope requires its exact domain")
    return domain.project(h,k)
