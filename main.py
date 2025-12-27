#Made on  Google collab!
#The external auditory canal segment was analyzed by sampling cross-sectional areas along its principal axis.
#Canal volume was computed by numerical integration of these areas. The canal isthmus was identified as the location of minimum cross-sectional area and used to 
#divide the canal into cartilaginous and bony segments, whose volumes were computed separately (A conservative error bound proportional to the maximum cross-sectional 
#area and sampling step was reported, and convergence was verified by increasing the number of sampled sections)

!pip install trimesh shapely pyembree


import numpy as np
import trimesh
import shapely.geometry as geom
import matplotlib.pyplot as plt
from google.colab import files


uploaded = files.upload()
filename = list(uploaded.keys())[0]

mesh = trimesh.load(filename, process=False)

print("Watertight:", mesh.is_watertight)
print("Faces:", len(mesh.faces))
print("Bounds (mm):", mesh.bounds)


pts = mesh.vertices
center = pts.mean(axis=0)

_, _, Vt = np.linalg.svd(pts - center)
axis = Vt[0]
axis /= np.linalg.norm(axis)

proj = (pts - center) @ axis


n_sections = 150
s_vals = np.linspace(proj.min(), proj.max(), n_sections)

areas = []
s_used = []

for s in s_vals:
    slice_ = mesh.section(
        plane_origin=center + s * axis,
        plane_normal=axis
    )
    if slice_ is None:
        continue

    slice_2d, _ = slice_.to_planar()
    pts_all = []

    for ent in slice_2d.entities:
        pts_all.append(ent.discrete(slice_2d.vertices))

    if not pts_all:
        continue

    poly = geom.Polygon(np.vstack(pts_all))
    if not poly.is_valid or poly.area <= 0:
        continue

    areas.append(poly.area)
    s_used.append(s)

areas = np.array(areas)
s_used = np.array(s_used)


idx_istmus = np.argmin(areas)
s_istmus = s_used[idx_istmus]

print(f"Isthmus at s = {s_istmus:.2f} mm")
print(f"Minimum area = {areas[idx_istmus]:.2f} mm²")


plt.figure(figsize=(6,4))
plt.plot(s_used, areas, '-k')
plt.axvline(s_istmus, color='r', linestyle='--', label='Isthmus')
plt.xlabel("Position along canal (mm)")
plt.ylabel("Cross-sectional area (mm²)")
plt.legend()
plt.title("Auditory canal cross-sectional area")
plt.show()


V_total_mm3 = np.trapz(areas, s_used)
print(f"Total volume = {V_total_mm3:.1f} mm³ ({V_total_mm3/1000:.3f} cm³)")


mask_outer = s_used < s_istmus
mask_inner = s_used >= s_istmus

V_outer = np.trapz(areas[mask_outer], s_used[mask_outer])
V_inner = np.trapz(areas[mask_inner], s_used[mask_inner])

print(f"Outer (cartilaginous) volume = {V_outer/1000:.3f} cm³")
print(f"Inner (bony) volume = {V_inner/1000:.3f} cm³")


res_100 = ear_volume_by_sections(mesh, n_sections=100)["volume_total_mm3"]
res_200 = ear_volume_by_sections(mesh, n_sections=200)["volume_total_mm3"]

err_conv = abs(res_200 - res_100)

print(f"Convergence error ≈ ±{err_conv:.1f} mm³")
print(f"Convergence error ≈ ±{err_conv/1000:.3f} cm³")
