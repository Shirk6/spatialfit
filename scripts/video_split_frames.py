"""List exactly the video frames the synced A|B split screen uses, so only those get rendered.

Writes build/video/tiles.json: per side the sorted unique video-frame indices, the matching log indices, and
the log name. Render each side with:
  blender -b --factory-startup -P render/blender_render2.py -- build/render_<cfg> build/<log>.npz build/video/tile_<side>/ \
          --engine cycles --res 1280x720 --samples 16 --frames <log indices, comma separated>
(PNG i of that render = i-th entry of the list).
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "render"))
import compose_final as C

logs = {"A": ("A_seed0_bias", "A"), "B": ("B_seed0", "B")}
huds = {k: json.load(open(C.BUILD / f"{v[0]}_hud.json"))["hud"] for k, v in logs.items()}
idxs = {k: C.video_index(huds[k]) for k in logs}
rem = {"A": C.remap_side(huds["A"], idxs["A"], 2.2, ("jam",)), "B": C.remap_side(huds["B"], idxs["B"], 2.2, ())}
out = {}
for side in logs:
    used = sorted({v for vs in rem[side].values() for v in vs})
    out[side] = dict(log=logs[side][0], cfg=logs[side][1], video_frames=used, log_frames=[idxs[side][v] for v in used])
    print(side, len(used), "frames")
json.dump(out, open(C.BUILD / "video" / "tiles.json", "w"))
