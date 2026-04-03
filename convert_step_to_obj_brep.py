r"""Convert STEP/STP files to OBJ with B-Rep face mapping using FreeCAD.

Each B-Rep face is tessellated separately, producing:
  {name}.obj           - the mesh (all triangles merged)
  {name}_face_map.npy  - int32 array mapping each triangle to its B-Rep face ID

This allows the query tool to select entire B-Rep faces (holes, fillets,
planar surfaces) with a single click instead of individual triangles.

Run in FreeCAD Python console:
    exec(open(r"C:\Users\harsh\projects\forks\PartField\convert_step_to_obj_brep.py").read())
"""
import os
import glob
import numpy as np

import FreeCAD
import Part

INPUT_DIR = r"C:\Users\harsh\projects\forks\PartField\data\step-200"
OUTPUT_DIR = r"C:\Users\harsh\projects\forks\PartField\data\step-200-obj-brep"

TESSELLATION_TOLERANCE = 0.1  # linear deflection in mm

os.makedirs(OUTPUT_DIR, exist_ok=True)

step_files = glob.glob(os.path.join(INPUT_DIR, "*.step")) + \
             glob.glob(os.path.join(INPUT_DIR, "*.STEP")) + \
             glob.glob(os.path.join(INPUT_DIR, "*.stp")) + \
             glob.glob(os.path.join(INPUT_DIR, "*.STP"))

# Deduplicate (case-insensitive glob on Windows may return same files)
seen = set()
unique_files = []
for f in step_files:
    normalized = os.path.normpath(f).lower()
    if normalized not in seen:
        seen.add(normalized)
        unique_files.append(f)

print(f"Found {len(unique_files)} STEP files to convert")

succeeded = 0
failed = 0
errors = []

for i, step_path in enumerate(sorted(unique_files), 1):
    basename = os.path.splitext(os.path.basename(step_path))[0]
    obj_path = os.path.join(OUTPUT_DIR, basename + ".obj")
    face_map_path = os.path.join(OUTPUT_DIR, basename + "_face_map.npy")

    if os.path.exists(obj_path) and os.path.exists(face_map_path):
        print(f"[{i}/{len(unique_files)}] Skipping (already exists): {basename}")
        succeeded += 1
        continue

    print(f"[{i}/{len(unique_files)}] Converting: {basename}")

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

        # Write OBJ
        with open(obj_path, 'w') as f:
            for v in all_verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for tri in all_tris:
                f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

        # Save face map
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
print(f"Done! Succeeded: {succeeded}/{len(unique_files)}, Failed: {failed}")
if errors:
    print(f"\nFailed files:")
    for name, err in errors:
        print(f"  {name}: {err}")
