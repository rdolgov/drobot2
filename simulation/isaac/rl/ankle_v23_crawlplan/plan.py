"""Pure geometric foot/body reference calculation; no simulator dependencies.

The explicit fixed stair map is a planning prior. It is not a camera-derived
map. Trajectories are motor targets only, never simulated-state writes.
"""
import math

NOMINAL = ((.08305, .198696689, 0.), (-.08695, .198696689, 0.),
           (.08695, -.198696689, 0.), (-.08305, -.198696689, 0.))
HIP_X = (.05805, -.06195, .06195, -.05805)
ORDER = (0, 2, 3, 1)


def smooth(t):
    t = min(1., max(0., t))
    return t*t*(3.-2.*t)


def make_plan(phases, dt=1/60, phase_s=9., start_x=.2, rise=.02, tread=.25, edge=.34):
    feet = [list(foot) for foot in NOMINAL]
    body_x = 0.
    rows, bases, stance, qrefs = [], [], [], []
    starts = []
    for phase in range(phases):
        cycle, order_id = divmod(phase, 4)
        leg = ORDER[order_id]
        old = [foot[:] for foot in feet]
        target = feet[leg][:]
        target[0] += .170 if cycle == 0 and leg in (1, 3) else .125
        target[2] = rise*min(3, max(0, math.floor((start_x+target[0]-edge)/tread)+1))
        cycle_start = 0. if cycle == 0 else .148+.125*(cycle-1)
        pre = cycle_start+(0., .035, .055, .115 if cycle == 0 else .100)[order_id]
        final = cycle_start+(.148 if cycle == 0 else .125) if order_id == 3 else pre
        starts.append({"phase": phase, "leg": leg, "foot_start": old[leg], "foot_goal": target,
                       "base_start_x": body_x, "base_pre_x": pre, "base_end_x": final})
        for tick in range(round(phase_s/dt)):
            t = tick*dt
            current = [foot[:] for foot in old]
            current[leg][0] += (target[0]-old[leg][0])*smooth((t-3.)/2.)
            peak = max(old[leg][2], target[2])+.025
            current[leg][2] += (peak-old[leg][2])*smooth((t-1.5)/1.5)
            current[leg][2] += (target[2]-peak)*smooth((t-5.)/1.5)
            bx = body_x+(pre-body_x)*smooth(t/1.5)+(final-pre)*smooth((t-7.)/2.)
            # Body-height reference follows support levels, not swing apex.
            plane_sum = sum(foot[2] for foot in old)+(target[2]-old[leg][2])*smooth((t-5.)/1.5)
            bz = .388908389+plane_sum/4
            # Keep 2.8 mm reserve from the exact 319.793 mm two-link reach.
            for index in range(4):
                forward = current[index][0]-bx-HIP_X[index]
                bz = min(bz, current[index][2]+.088908389+math.sqrt(max(.001, .317**2-forward**2)))
            qs = [0.]*16
            for index in range(4):
                forward = current[index][0]-bx-HIP_X[index]
                down = bz-current[index][2]-.088908389
                cosine = max(-.999, min(.999, (forward**2+down**2-2*.159896689**2)/(2*.159896689**2)))
                knee = (-1 if index in (0, 2) else 1)*math.acos(cosine)
                hip = math.atan2(forward, down)-math.atan2(math.sin(knee), 1+math.cos(knee))
                # Swing feet may tilt when the ankle reaches its real soft limit.
                ankle = max(-math.radians(33.25), min(math.radians(33.25), -hip-knee))
                qs[4+index], qs[8+index], qs[12+index] = hip, knee, ankle
            rows.append(current)
            bases.append([bx, 0., bz])
            stance.append([not (index == leg and 1.5<t<6.5) for index in range(4)])
            qrefs.append(qs)
        feet[leg] = target
        body_x = final
    # Exact final reference, held for four seconds.
    for _ in range(round(4./dt)+2):
        rows.append([foot[:] for foot in feet])
        bases.append(bases[-1][:])
        stance.append([True]*4)
        qrefs.append(qrefs[-1][:])
    return {"feet": rows, "base": bases, "stance": stance, "joints": qrefs,
            "goal_feet": feet, "moved_legs": sorted(set(ORDER[:min(4, phases)])), "phases": starts}
