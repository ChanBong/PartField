r"""Convert embossment test STEP files to OBJ with B-Rep face mapping.

Run in FreeCAD Python console:
    exec(open(r"C:\Users\harsh\projects\forks\PartField\convert_emboss_test.py").read())
"""
import os
import numpy as np

import FreeCAD
import Part

INPUT_FILES = [
    r"C:\Users\harsh\Downloads\can_opener.step",
    r"C:\Users\harsh\Downloads\cmkib4hnd026r5tp7c250u2mo.step",
    r"C:\Users\harsh\Downloads\cmkm85tfc02qo5tp7iy8rn9r1 (1).step",
    r"C:\Users\harsh\Downloads\cmkm85tfc02qo5tp7iy8rn9r1.step",
    r"C:\Users\harsh\Downloads\cmmq6emrr01e3c6p7ymud109e.step",
    r"C:\Users\harsh\Downloads\cmmt1q54d00ql33p8l6uexj5b.step",
    r"C:\Users\harsh\Downloads\ecu-housing-machined-test.stp",
]
OUTPUT_DIR = r"C:\Users\harsh\projects\forks\PartField\data\emboss-test-obj"

TESSELLATION_TOLERANCE = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)

succeeded = 0
failed = 0
errors = []

for i, step_path in enumerate(INPUT_FILES, 1):
    basename = os.path.splitext(os.path.basename(step_path))[0]
    obj_path = os.path.join(OUTPUT_DIR, basename + ".obj")
    face_map_path = os.path.join(OUTPUT_DIR, basename + "_face_map.npy")

    if os.path.exists(obj_path) and os.path.exists(face_map_path):
        print(f"[{i}/{len(INPUT_FILES)}] Skipping (already exists): {basename}")
        succeeded += 1
        continue

    print(f"[{i}/{len(INPUT_FILES)}] Converting: {basename}")

    try:
        shape = Part.Shape()
        shape.read(os.path.abspath(step_path))

        all_verts = []
        all_tris = []
        face_map = []
        vert_offset = 0

        for face_id, face in enumerate(shape.Faces):
            try:
                verts, tris = face.tessellate(TESSELLATION_TOLERANCE)
            except Exception as e:
                print(f"  Warning: skipping B-Rep face {face_id}: {e}")
                continue

            if len(tris) == 0:
                continue

            for v in verts:
                all_verts.append((v.x, v.y, v.z))

            for tri in tris:
                all_tris.append((tri[0] + vert_offset,
                                 tri[1] + vert_offset,
                                 tri[2] + vert_offset))
                face_map.append(face_id)

            vert_offset += len(verts)

        if len(all_tris) == 0:
            raise RuntimeError("Tessellation produced 0 triangles")

        with open(obj_path, 'w') as f:
            for v in all_verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for tri in all_tris:
                f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

        np.save(face_map_path, np.array(face_map, dtype=np.int32))

        n_brep = len(set(face_map))
        print(f"  {len(all_verts)} verts, {len(all_tris)} tris, "
              f"{n_brep} B-Rep faces, {len(shape.Faces)} total B-Rep faces")
        succeeded += 1

    except Exception as e:
        print(f"  ERROR: {e}")
        errors.append((basename, str(e)))
        failed += 1

print(f"\n{'='*50}")
print(f"Done! Succeeded: {succeeded}/{len(INPUT_FILES)}, Failed: {failed}")
if errors:
    print(f"\nFailed files:")
    for name, err in errors:
        print(f"  {name}: {err}")
