"""Output-frame -> log-frame list for the stability comparison shot (shared by step 6 and step 7, 120 Hz logs).

Speed (log seconds per video second) eases between: 1x during the hold, 0.25x while the push acts,
0.5x during the fall, 1x afterwards. Writes build/video/fall2_frames.json.
"""
import json
import numpy as np

A = json.load(open('build/video/fall6_annot.json'))
fps_log, n = A['fps'], len(A['frames'])
t_rel, t_push = A['hold_t'], A['push_t']


def speed(t):
    pts = [(0.0, 1.0), (t_rel - 0.25, 1.0), (t_rel, 0.25), (t_rel + t_push, 0.25), (t_rel + t_push + 0.25, 0.5),
           (t_rel + 1.4, 0.5), (t_rel + 1.8, 1.0), (99, 1.0)]
    xs, ys = zip(*pts)
    return float(np.interp(t, xs, ys))


t, out = 0.55, []                         # start a little into the hold
while t < (n - 1) / fps_log:
    out.append(int(round(t * fps_log)))
    t += speed(t) / 30.0
json.dump(dict(log_frames=out, hold_end_frames=30), open('build/video/fall2_frames.json', 'w'))
print(len(out), 'frames,', round(len(out) / 30, 1), 's (+1 s hold)')
