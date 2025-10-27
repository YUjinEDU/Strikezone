import numpy as np

from strikezone.config import CameraIntrinsics
from strikezone.depth import back_project_uvz_to_xyz, project_xyz_to_uv


def test_round_trip_projection():
    intrinsics = CameraIntrinsics(fx=800.0, fy=805.0, cx=320.0, cy=240.0)
    rng = np.random.default_rng(42)

    for _ in range(10):
        point = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(5.0, 15.0)])
        uv = project_xyz_to_uv(point, intrinsics)
        reconstructed = back_project_uvz_to_xyz(uv[0], uv[1], point[2], intrinsics)
        np.testing.assert_allclose(reconstructed, point, atol=1e-6)
