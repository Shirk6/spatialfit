"""Encode a PNG frame sequence to H.264 mp4 with Blender's sequencer (no ffmpeg needed on the worker).

usage: blender -b --factory-startup -P render/encode_frames.py -- FRAMES_DIR OUT.mp4 [fps]
"""
import bpy, os, sys
argv = sys.argv[sys.argv.index("--") + 1:]
fdir, out = argv[0], argv[1]
fps = int(argv[2]) if len(argv) > 2 else 30
files = sorted(f for f in os.listdir(fdir) if f.startswith("frame_") and f.endswith(".png"))
img = bpy.data.images.load(os.path.join(fdir, files[0])); w, h = img.size
sc = bpy.context.scene
ed = sc.sequence_editor_create()
coll = getattr(ed, "strips", None) or ed.sequences
st = coll.new_image(name="frames", filepath=os.path.join(fdir, files[0]), channel=1, frame_start=1)
for f in files[1:]:
    st.elements.append(f)
sc.frame_start, sc.frame_end = 1, len(files)
sc.render.fps = fps
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = w, h, 100
sc.render.image_settings.file_format = "FFMPEG"
sc.render.ffmpeg.format = "MPEG4"; sc.render.ffmpeg.codec = "H264"
sc.render.ffmpeg.constant_rate_factor = "HIGH"; sc.render.ffmpeg.ffmpeg_preset = "GOOD"
sc.render.filepath = out
sc.render.use_file_extension = False
bpy.ops.render.render(animation=True)
print("encoded", len(files), "frames ->", out)
