"""Bounded offline geometry study of forward+lateral pre-swing body transfer.

New references only. No simulation, policy, hardware, mass/asset or safety-cap
change. Passing this geometry study is not demonstrated support stability.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path

import generate_shifted_common_floor_crawl_reference as shifted

g = shifted.geometry
crawl = shifted.crawl


@dataclass(frozen=True)
class PoseFamily:
    forward_transfer_m: float = .020
    lateral_transfer_m: float = .030
    lowest_edge_depth_m: float = .325
    front_stance_fore_m: float = .100
    rear_stance_aft_m: float = .085
    duty: float = .85
    stride_m: float = .055
    lift_m: float = .020


def forward_shift(phase: float, amplitude: float, duty: float):
    """X transfer only for rear swings; return during the next all-support gap."""
    phase %= .5
    width = duty - .75
    if phase < width:
        u, sign, value = phase/width, 1., 0.
    elif phase < .25:
        return amplitude, 0.
    elif phase < .25+width:
        u, sign, value = (phase-.25)/width, -1., amplitude
    else:
        return 0., 0.
    return value+sign*amplitude*g.smoothstep(u), sign*amplitude/width*30*u*u*(1-u)*(1-u)


def leg_trajectory(phase: float, front: float, family: PoseFamily):
    x, clearance, stance, envelope = shifted.trajectory(phase,front,family.duty,family.stride_m,family.lift_m)
    nominal = family.front_stance_fore_m if front > 0 else family.rear_stance_aft_m
    return x+front*(nominal-g.STANCE_FORWARD_M),clearance,stance,envelope


def solve_pose(x, clearance, body_y, side, front, depth):
    target = shifted.HIP_DROP_M+depth-clearance
    if front*x <= 0:
        raise ValueError("opposed_IK_branch_crossing")
    radius2 = (g.PROXIMAL_M+g.TREAD_CONTACT_M)**2-x*x
    if radius2 <= 0:
        raise ValueError("sagittal_reach")
    def pose(d):
        h,k=g.inverse_sagittal(x,d,front)
        return shifted.abduction_for_center(d,body_y,side),h,k
    low,high=.150,math.sqrt(radius2)-1e-10
    if shifted.smooth_corner_depth(*pose(low)) >= target or shifted.smooth_corner_depth(*pose(high)) <= target:
        raise ValueError("common_floor_reach_bracket")
    for _ in range(50):
        mid=(low+high)/2
        if shifted.smooth_corner_depth(*pose(mid)) > target:
            high=mid
        else:
            low=mid
    answer=pose((low+high)/2)
    if abs(shifted.smooth_corner_depth(*answer)-target)>1e-9:
        raise ValueError("floor_solve_precision")
    return answer


def family_parameters_valid(f):
    intervals={
        "forward_transfer_m":(.010,.025),"lateral_transfer_m":(.020,.030),
        "lowest_edge_depth_m":(.320,.330),"front_stance_fore_m":(.080,.110),
        "rear_stance_aft_m":(.075,.095),
    }
    for key,(low,high) in intervals.items():
        value=getattr(f,key)
        if not math.isfinite(value) or not low-1e-12<=value<=high+1e-12:
            raise ValueError(f"{key} outside explicitly scoped search family")
    if not (math.isclose(f.duty,.85) and math.isclose(f.stride_m,.055) and math.isclose(f.lift_m,.020)):
        raise ValueError("Duty .85, stride55mm and lift20mm remain fixed in this study")


def rows_and_metrics(family, count, frequency, *, reject_limits=True):
    family_parameters_valid(family)
    rows=[]
    metrics={"max_stance_sole_tilt_deg":0.,"max_corner_gap_m":0.,"max_support_depth_spread_m":0.,
             "minimum_core_99pct_corner_height_m":math.inf,"maximum_core_99pct_corner_height_m":-math.inf,
             "max_tread_center_XY_error_m":0.}
    for i in range(count):
        phase=i/count
        bx,_=forward_shift(phase,family.forward_transfer_m,family.duty)
        by=shifted.body_lateral_shift(phase,family.lateral_transfer_m,family.duty)
        row=[0.]*12;support_depths=[]
        for j,(front,side,offset) in enumerate(zip(shifted.FRONTS,shifted.SIDES,shifted.OFFSETS)):
            nominal_x,clearance,stance,envelope=leg_trajectory((phase+offset)%1,front,family)
            x=nominal_x-bx
            a,h,k=solve_pose(x,clearance,by,side,front,family.lowest_edge_depth_m)
            if any(abs(angle)>math.radians(limit) for angle,limit in zip((a,h,k),(23.75,57.,85.5))):
                raise ValueError("95pct_joint_soft_bound")
            tilt=math.degrees(math.acos(max(-1.,min(1.,math.cos(h+k)*math.cos(a)))))
            if stance:
                metrics["max_stance_sole_tilt_deg"]=max(metrics["max_stance_sole_tilt_deg"],tilt)
                if reject_limits and tilt>12.:
                    raise ValueError("12deg_stance_sole_tilt")
            corners=shifted.corner_positions(a,h,k,side)
            lowest_depth=-min(c[2] for c in corners)
            height=shifted.HIP_DROP_M+family.lowest_edge_depth_m-lowest_depth
            gap=height-clearance
            if not -1e-8<=gap<=shifted.SMOOTH_GAP_BOUND_M+1e-8:
                raise ValueError("exact_3D_corner_gap")
            metrics["max_corner_gap_m"]=max(metrics["max_corner_gap_m"],gap)
            if stance:
                support_depths.append(lowest_depth)
            elif envelope>=.99:
                metrics["minimum_core_99pct_corner_height_m"]=min(metrics["minimum_core_99pct_corner_height_m"],height)
                metrics["maximum_core_99pct_corner_height_m"]=max(metrics["maximum_core_99pct_corner_height_m"],height)
            center=[sum(c[axis] for c in corners[4:])/4 for axis in (0,1)]
            expected=[side*shifted.HIP_FORWARD_OFFSET_M+x,side*shifted.HIP_OUTWARD_M-by]
            metrics["max_tread_center_XY_error_m"]=max(metrics["max_tread_center_XY_error_m"],math.dist(center,expected))
            row[j],row[4+j],row[8+j]=a,h,k
        metrics["max_support_depth_spread_m"]=max(metrics["max_support_depth_spread_m"],max(support_depths)-min(support_depths))
        rows.append(row)
    rates=[max(abs(rows[(i+1)%count][j]-rows[i][j]) for i in range(count))*count*frequency*180/math.pi for j in range(12)]
    if reject_limits and any(rate>limit for rate,limit in zip(rates,[60.]*4+[90.]*8)):
        raise ValueError("60abd_90sag_target_rate")
    metrics.update(samples_per_cycle=count,frequency_hz=frequency,
                   peak_joint_target_velocity_deg_s=rates,
                   nominal_60Hz_packet_step_upper_bound_deg=[r/60 for r in rates],
                   joint_ranges_deg=[[math.degrees(min(r[j] for r in rows)),math.degrees(max(r[j] for r in rows))] for j in range(12)])
    return rows,metrics


def write_new(path,record):
    if path.exists():
        raise ValueError("Never overwrite a source reference or report")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(record,stream,indent=2,allow_nan=False);stream.write("\n")


def search(path):
    if path.exists():
        raise ValueError("Use a NEW search output")
    accepted=[];rejections={};attempts=0
    for x,y,down,front,rear in itertools.product((.010,.015,.020,.025),(.030,.020),(.330,.3275,.325,.3225,.320),(.080,.085,.090,.095,.100,.105,.110),(.075,.080,.085,.090,.095)):
        family=PoseFamily(x,y,down,front,rear)
        attempts+=1
        try:
            _,metrics=rows_and_metrics(family,128,.10)
        except ValueError as exc:
            reason=str(exc);rejections[reason]=rejections.get(reason,0)+1
            continue
        accepted.append({"family":asdict(family),"coarse_metrics":metrics})
    accepted.sort(key=lambda r:(-r["family"]["forward_transfer_m"],
                               abs(r["family"]["lowest_edge_depth_m"]-.330)+abs(r["family"]["front_stance_fore_m"]-.080)+abs(r["family"]["rear_stance_aft_m"]-.080)))
    report={"qualification":"COARSE_ANALYTICAL_SCREEN_ONLY_NOT_DYNAMIC_OR_CONTINUOUS_PROOF",
            "generator_sha256":crawl.digest(Path(__file__)),"attempts":attempts,
            "coarse_samples_per_cycle":128,"frequency_hz":.10,"rejection_counts":rejections,
            "accepted":accepted,"accepted_count":len(accepted),
            "limitations":["Dense full-cycle bounds must be recomputed before reference export.",
                           "No mass FK or dynamic COM support claim is made by this joint/sole/reach screen."]}
    write_new(path,report)
    print(f"ANALYTICAL screen: {len(accepted)}/{attempts} coarse candidates accepted; {path}")
    for result in accepted[:8]:
        print(json.dumps({"family":result["family"],"peak_rate":max(result["coarse_metrics"]["peak_joint_target_velocity_deg_s"]),"sole":result["coarse_metrics"]["max_stance_sole_tilt_deg"]}))


def generate(output_dir,family,count,mass_provenance_trace=None):
    paths=[output_dir/name for name in ("low.json","high.json")]
    if any(path.exists() for path in paths):
        raise ValueError("Choose a NEW output directory")
    rows,export_metrics=rows_and_metrics(family,count,.10)
    _,dense_metrics=rows_and_metrics(family,max(4096,count),.10)
    mass_context=None
    if mass_provenance_trace is not None:
        with mass_provenance_trace.open("r",encoding="utf-8") as stream:
            metadata=json.loads(stream.readline())
        if metadata.get("record_type")!="metadata":
            raise ValueError("Mass context input must be a diagnostic trace with metadata header")
        domain=metadata["static_domain_parameters"]
        mass_context={
            "qualification":"Mass-context provenance only; references do not modify mass. Use the matching isolated environment dry mass override.",
            "trace_path":str(mass_provenance_trace),"trace_sha256":crawl.digest(mass_provenance_trace),
            "trace_id":metadata["trace_id"],
            "total_robot_mass_kg":domain["total_robot_mass_kg"],
            "base_mass_kg":domain["base_mass_kg"],"base_center_of_mass_m":domain["base_center_of_mass_m"],
            "dry_robot_mass_scale":metadata["environment_ablation"]["effective_values"].get("dry_robot_mass_scale"),
        }
    records=[]
    for frequency in (.09,.10):
        speed=family.stride_m*frequency/family.duty
        bx=[forward_shift(i/count,family.forward_transfer_m,family.duty) for i in range(count)]
        by=[shifted.body_lateral_shift(i/count,family.lateral_transfer_m,family.duty) for i in range(count)]
        dy=[shifted.body_shift_derivative(i/count,family.lateral_transfer_m,family.duty) for i in range(count)]
        records.append({
            "schema":"drobot.phase-joint-reference.v1","sample_count":count,"joint_order":crawl.ACTION_ORDER,
            "joint_position_rad":rows,"forward_speed_m_s":speed,
            "body_lateral_shift_m":by,"body_lateral_shift_derivative_m_per_cycle":dy,
            "body_forward_shift_m":[value[0] for value in bx],
            "body_forward_shift_derivative_m_per_cycle":[value[1] for value in bx],
            "generation":{
                "qualification":"UNQUALIFIED_COMBINED_TRANSFER_CRAWL_ABLATION",
                "method":"combined-transfer-common-floor-full-3d.v1","family":asdict(family),
                "generator_sha256":crawl.digest(Path(__file__)),"geometry_helper_sha256":crawl.digest(Path(shifted.__file__)),
                "canonical_urdf_path":shifted.CANONICAL_URDF.relative_to(shifted.REPOSITORY_ROOT).as_posix(),
                "canonical_urdf_sha256":crawl.digest(shifted.CANONICAL_URDF),
                "runtime_kinematics_sha256":crawl.digest(shifted.RUNTIME_KINEMATICS),
                "reference_mass_context":mass_context,
                "frequency_hz":frequency,"gait_period_s":1/frequency,"speed_m_s":speed,
                "duty_factor":family.duty,"phase_offsets_fl_rl_fr_rr":shifted.OFFSETS,
                "common_lowest_edge_depth_m":family.lowest_edge_depth_m,
                "stride_m":family.stride_m,"sole_corner_clearance_m":family.lift_m,
                "clearance_envelope":"quintic-symmetric.v1","clearance_transition_fraction":.5,
                "contact_schedule":crawl.schedule(family.duty,shifted.OFFSETS),
                "lateral_reference_contract":{
                    "schema":"drobot.phase-body-lateral-reference.v1",
                    "positive_direction":"body translation along episode-left (+Y at initial heading), NOT foot offset",
                    "derivative_units":"metres per unit global cycle phase; multiply by actual phase rate for m/s",
                    "phase_alignment":"Same phase i/sample_count as joint rows; low/high vectors identical",
                    "phase_zero_shift_m":-family.lateral_transfer_m,"cycle_mean_shift_m":0.},
                "forward_reference_contract":{
                    "schema":"drobot.phase-body-forward-reference.v1",
                    "positive_direction":"body translation along episode-forward (+X at initial heading), NOT foot offset",
                    "derivative_units":"metres per unit global cycle phase; multiply by actual phase rate for m/s",
                    "phase_alignment":"Same phase i/sample_count as joint rows; low/high vectors identical",
                    "phase_zero_shift_m":0.,"cycle_mean_shift_m":family.forward_transfer_m/2,
                    "maximum_abs_derivative_m_per_cycle":family.forward_transfer_m/(family.duty-.75)*1.875,
                    "positive_plateaus_for":["rear_right","rear_left"],
                    "zero_plateaus_for":["front_right","front_left"],
                    "transition_phase_intervals":[[0,.10],[.25,.35],[.5,.60],[.75,.85]],
                    "startup_stop":"Periodic posture transfer only. Scoring of periodic forward displacement/velocity and stop recentering requires explicit future env support; these arrays do not change current reward code."},
                "exported_table_metrics_at_0p10Hz":export_metrics,
                "dense_analytical_metrics_at_0p10Hz":dense_metrics,
                "target_velocity_limit_deg_s_by_action":[60.]*4+[90.]*8,"packet_step_limit_deg":2.,
                "maximum_stance_sole_tilt_deg":12.,
                "forward_push_derivation":"During support x_body=stance+stroke_phase-body_X and y_body=nominal_y-body_Y; intended body velocity combines +stroke*cadence/duty with the explicit transfer derivatives. Pressure-center rolling/slip is not modelled by center matching.",
                "limitations":["Analytical reference only, not loaded contact, whole-COM balance or dynamic validation.",
                               "No assets, masses, payload, servo gains or physical limits changed.",
                               "Body assumed level; full 3D rotated structural/tread corners are solved.",
                               "A stance geometry hypothesis is not a measured calibration.",
                               "Periodic bounds do not establish startup/stop or held-packet recovery rates.",
                               "Future phase-aware forward reward support is not implemented by this generator."]}})
    for path,record in zip(paths,records):
        write_new(path,record)
        print(f"Wrote {path}; ANALYTICAL_UNQUALIFIED; peak rate {max(dense_metrics['peak_joint_target_velocity_deg_s']):.3f}deg/s at .10Hz")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    output=parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--search-output",type=Path)
    output.add_argument("--output-dir",type=Path)
    parser.add_argument("--forward-transfer-m",type=float,default=.020)
    parser.add_argument("--lateral-transfer-m",type=float,default=.030)
    parser.add_argument("--lowest-edge-depth-m",type=float,default=.325)
    parser.add_argument("--front-stance-fore-m",type=float,default=.100)
    parser.add_argument("--rear-stance-aft-m",type=float,default=.085)
    parser.add_argument("--samples",type=int,default=1024)
    parser.add_argument("--mass-provenance-trace",type=Path)
    args=parser.parse_args()
    try:
        if args.search_output:
            search(args.search_output)
        else:
            if args.samples<512:
                raise ValueError("Use >=512 exported rows")
            generate(args.output_dir,PoseFamily(args.forward_transfer_m,args.lateral_transfer_m,args.lowest_edge_depth_m,args.front_stance_fore_m,args.rear_stance_aft_m),args.samples,args.mass_provenance_trace)
    except ValueError as exc:
        parser.error(str(exc))


if __name__=="__main__":
    main()
