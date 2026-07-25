"""Visualize logged metric_depth in 3D (viser web viewer — works over SSH/VM).

Single frame: backproject one metric_depth image, colored by its hires JPG.
--all: TSDF-fuse every frame into a surface mesh in the VIO world frame using
qvio_extended poses (see tsdf_fuse), and draw the camera trajectory.

Serves a web viewer (default http://localhost:8080); VSCode forwards the port.

Camera->world inverts geometry.py:
    P_imu = P_cam @ R_cam_to_imu.T + T_cam_wrt_imu   (hires extrinsic)
    P_vio = P_imu @ R_imu_to_vio.T + T_imu_wrt_vio   (VIO pose)

uv run python visualize.py --all --stride 3            # fused mesh + trajectory
uv run python visualize.py --all --voxel 0.02 --max-range 5   # finer, near-range
uv run python visualize.py --all --save map.ply        # export mesh, no viewer
uv run python visualize.py --frame 5                   # single-frame cloud

"""
from pathlib import Path
import argparse
import struct
import csv
import time
import numpy as np
import cv2
import viser

from proto.config import load
from proto.rescaler.camera_model import CameraModel
from proto.rescaler.preprocess import Preprocessor
from proto.vio.ext_vio import parse_buffer

_META = struct.Struct("<Iqihhiiihhhh")  # camera_image_metadata_t, 40 bytes
_TOL_NS = 100_000_000                    # frame<->pose match window
_DEFAULT = ("/home/ubuntu/voxl_logs/onboard/rescaler_20260604_180337"
            "/log0000/run/mpa")


def depth_frames(raw_path, w, h):
    """Yield (frame_id, timestamp_ns, depth[h,w])."""
    pkt = 40 + w * h * 4
    buf = np.memmap(raw_path, dtype=np.uint8, mode="r")
    for off in range(0, (len(buf) // pkt) * pkt, pkt):
        m = _META.unpack_from(buf, off)
        depth = buf[off + 40: off + pkt].view("<f4").reshape(h, w)
        yield m[2], m[1], depth


def hires_by_frame_id(hires_dir):
    """Map frame_id -> JPG path (row index i is 000i.jpg)."""
    rows = list(csv.DictReader(open(Path(hires_dir) / "data.csv")))
    return {int(r["frame_id"]): Path(hires_dir) / f"{i:05d}.jpg"
            for i, r in enumerate(rows)}


def load_poses(raw_path):
    """Return (ts[N], R_imu_to_vio[N,3,3], T_imu_wrt_vio[N,3]) for valid poses."""
    recs = parse_buffer(Path(raw_path).read_bytes())["v"]
    ts = recs["timestamp_ns"].astype(np.int64)
    R = recs["R_imu_to_vio"].astype(np.float64)
    T = recs["T_imu_wrt_vio"].astype(np.float64)
    ok = np.isfinite(R).all((1, 2)) & (np.abs(np.linalg.det(R) - 1.0) < 0.02)
    order = np.argsort(ts[ok])
    return ts[ok][order], R[ok][order], T[ok][order]


def hires_model_rgb(pre, jpg):
    """Load a hires JPG and remap it onto the depth grid (uint8, HxWx3)."""
    return pre.prepare(cv2.cvtColor(cv2.imread(str(jpg)), cv2.COLOR_BGR2RGB))


def backproject(depth, K):
    """Depth[h,w] -> P_cam[h*w,3] in the OpenCV camera frame."""
    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64)
    return np.stack([(us - K[0, 2]) / K[0, 0] * z,
                     (vs - K[1, 2]) / K[1, 1] * z, z], -1).reshape(-1, 3)


def write_ply(path, pts, cols):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n")
        np.savetxt(f, np.hstack([pts, cols]), fmt="%.3f %.3f %.3f %d %d %d")


# --------------------------------------------------------------------------- #
# TSDF fusion (Open3D). Integrates every frame's RGBD into one volume so
# overlapping observations average into a surface instead of scattering.
# --------------------------------------------------------------------------- #
def tsdf_fuse(matched, hires, pre, R_ci, T_ci, poses,
              stride, voxel, sdf_trunc, max_range):
    """Return (trimesh.Trimesh, camera_centers[N,3]) fused in the VIO frame."""
    import open3d as o3d
    import trimesh

    ts, R_iv, T_iv = poses
    K = pre.K_model
    intr = o3d.camera.PinholeCameraIntrinsic(
        pre.dst_w, pre.dst_h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    centers = []
    for fid, t, depth in matched[::stride]:
        j = int(np.argmin(np.abs(ts - t)))
        if abs(ts[j] - t) > _TOL_NS:
            continue
        R_c2w = R_iv[j] @ R_ci
        c = R_iv[j] @ T_ci + T_iv[j]                 # camera center in world
        extr = np.eye(4)                             # world->camera
        extr[:3, :3] = R_c2w.T
        extr[:3, 3] = -R_c2w.T @ c
        rgb = np.ascontiguousarray(hires_model_rgb(pre, hires[fid]))
        dep = np.ascontiguousarray(depth.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), o3d.geometry.Image(dep),
            depth_scale=1.0, depth_trunc=max_range, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, extr)
        centers.append(c)

    m = vol.extract_triangle_mesh()
    m.compute_vertex_normals()
    mesh = trimesh.Trimesh(
        vertices=np.asarray(m.vertices), faces=np.asarray(m.triangles),
        vertex_colors=(np.asarray(m.vertex_colors) * 255).astype(np.uint8),
        vertex_normals=np.asarray(m.vertex_normals), process=False)
    return mesh, np.asarray(centers)


def serve(port):
    print(f"viser serving on port {port} — open the forwarded port in your "
          "browser. Ctrl+C to stop.")
    while True:
        time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=_DEFAULT, help="run/mpa log dir")
    ap.add_argument("--frame", type=int, default=0, help="matched-frame index")
    ap.add_argument("--all", action="store_true", help="TSDF-fuse all frames into a mesh")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame (--all)")
    ap.add_argument("--voxel", type=float, default=0.03, help="TSDF voxel m (--all)")
    ap.add_argument("--sdf-trunc", type=float, default=0.12, help="TSDF truncation m (--all)")
    ap.add_argument("--max-range", type=float, default=0.0, help="drop depth beyond m; 0 keeps all")
    ap.add_argument("--point-size", type=float, default=0.02, help="viser point size m")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--save", help="write geometry and exit (single: .ply, --all: mesh)")
    args = ap.parse_args()

    cfg = load()
    cam = CameraModel(cfg.hires.K, cfg.hires.D, cfg.hires.distortion_model)
    W, H = cfg.inference.input_resolution
    pre = Preprocessor(cam, (cfg.hires.width, cfg.hires.height), (W, H),
                       mode=cfg.inference.preprocess, fov=cfg.inference.fov,
                       antialias=cfg.inference.antialias)

    mpa = Path(args.log)
    hires = hires_by_frame_id(mpa / "hires_small_color")
    matched = [(fid, ts, d) for fid, ts, d in
               depth_frames(mpa / "metric_depth" / "data.raw", W, H) if fid in hires]
    if not matched:
        raise SystemExit("no metric_depth frame matches a hires JPG")

    if args.all:
        mesh, centers = tsdf_fuse(
            matched, hires, pre, cfg.extr_hires.R, cfg.extr_hires.T,
            load_poses(mpa / "qvio_extended" / "data.raw"),
            args.stride, args.voxel, args.sdf_trunc, args.max_range or 6.0)
        print(f"TSDF mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} tris")
        if args.save:
            mesh.export(args.save)
            print("wrote", args.save)
            return
        server = viser.ViserServer(port=args.port)
        server.scene.set_up_direction("-z")
        server.scene.add_mesh_trimesh("/tsdf", mesh)
        if len(centers) >= 2:
            server.scene.add_spline_catmull_rom(
                "/trajectory", points=centers.astype(np.float32),
                color=(255, 0, 0), line_width=3.0)
    else:
        fid, _, depth = matched[args.frame]
        cols = hires_model_rgb(pre, hires[fid]).reshape(-1, 3)
        pts = backproject(depth, pre.K_model)
        keep = pts[:, 2] > 0
        if args.max_range > 0:
            keep &= pts[:, 2] <= args.max_range
        pts, cols = pts[keep], cols[keep]
        print(f"frame {args.frame}  frame_id {fid}  {len(pts)} points  "
              f"z {pts[:, 2].min():.2f}..{pts[:, 2].max():.2f} m")
        if args.save:
            write_ply(args.save, pts, cols)
            print("wrote", args.save)
            return
        server = viser.ViserServer(port=args.port)
        server.scene.set_up_direction("-y")
        server.scene.add_point_cloud(
            "/cloud", points=pts.astype(np.float32),
            colors=cols.astype(np.uint8), point_size=args.point_size)

    serve(args.port)


if __name__ == "__main__":
    main()
