from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pyrealsense2 as rs

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs" / "realsense_depth"
WIDTH = 640
HEIGHT = 480
FPS = 30
WARMUP_FRAMES = 30
MIN_DISTANCE_M = 0.1
MAX_DISTANCE_M = 2.0
POINT_SAMPLE_COUNT = 6000


def depth_to_gray(depth_m, min_distance_m=MIN_DISTANCE_M, max_distance_m=MAX_DISTANCE_M):
    valid = (depth_m > min_distance_m) & (depth_m <= max_distance_m)
    clipped = np.clip(depth_m, min_distance_m, max_distance_m)

    gray = 255 - (
        (clipped - min_distance_m)
        / (max_distance_m - min_distance_m)
        * 255
    )
    gray[~valid] = 0

    return gray.astype(np.uint8), valid


def depth_to_point_cloud(depth_m, intrinsics, valid_mask):
    height, width = depth_m.shape
    pixel_x, pixel_y = np.meshgrid(np.arange(width), np.arange(height))

    z = depth_m
    x = (pixel_x - intrinsics.ppx) / intrinsics.fx * z
    y = (pixel_y - intrinsics.ppy) / intrinsics.fy * z

    points = np.dstack((x, y, z)).astype(np.float32)
    return points[valid_mask]


def save_point_cloud_graph(points, output_path):
    if points.size == 0:
        raise ValueError("No valid depth points were captured.")

    step = max(1, len(points) // POINT_SAMPLE_COUNT)
    sampled = points[::step]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        sampled[:, 0],
        sampled[:, 1],
        sampled[:, 2],
        c=sampled[:, 2],
        cmap="viridis",
        s=1,
        alpha=0.75,
    )

    ax.set_title("RealSense Depth Point Cloud")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=20, azim=-65)
    fig.colorbar(scatter, ax=ax, shrink=0.65, label="Depth (m)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return sampled


def set_depth_sensor_options(depth_sensor):
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)

    if depth_sensor.supports(rs.option.enable_auto_exposure):
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)


def capture_realsense_depth():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    profile = pipeline.start(config)
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()
    hole_filling_filter = rs.hole_filling_filter()

    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        set_depth_sensor_options(depth_sensor)
        depth_scale = depth_sensor.get_depth_scale()

        depth_frame = None
        color_frame = None

        for _ in range(WARMUP_FRAMES):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

        if not depth_frame:
            raise RuntimeError("Depth frame was not captured.")

        filtered_depth = spatial_filter.process(depth_frame)
        filtered_depth = temporal_filter.process(filtered_depth)
        filtered_depth = hole_filling_filter.process(filtered_depth)
        filtered_depth = filtered_depth.as_depth_frame()

        depth_raw = np.asanyarray(filtered_depth.get_data())
        depth_m = depth_raw.astype(np.float32) * depth_scale
        depth_gray, valid_mask = depth_to_gray(depth_m)
        depth_colormap = cv2.applyColorMap(depth_gray, cv2.COLORMAP_JET)

        depth_intrinsics = (
            filtered_depth.profile
            .as_video_stream_profile()
            .intrinsics
        )
        points = depth_to_point_cloud(depth_m, depth_intrinsics, valid_mask)

        color_path = OUTPUT_DIR / "realsense_color.jpg"
        depth_gray_path = OUTPUT_DIR / "realsense_depth_gray.jpg"
        depth_colormap_path = OUTPUT_DIR / "realsense_depth_colormap.jpg"
        point_cloud_path = OUTPUT_DIR / "realsense_point_cloud.png"
        depth_m_path = OUTPUT_DIR / "realsense_depth_m.npy"

        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imwrite(str(color_path), color_image)

        cv2.imwrite(str(depth_gray_path), depth_gray)
        cv2.imwrite(str(depth_colormap_path), depth_colormap)
        np.save(depth_m_path, depth_m)
        sampled_points = save_point_cloud_graph(points, point_cloud_path)

        valid_depth = depth_m[valid_mask]

        print(f"depth_scale: {depth_scale}")
        print(f"depth_shape: {depth_m.shape}")
        print(f"valid_depth_pixels: {valid_depth.size}")
        print(f"point_cloud_points: {points.shape[0]}")
        print(f"sampled_points: {sampled_points.shape[0]}")
        if valid_depth.size:
            print(f"depth_min_m: {valid_depth.min():.4f}")
            print(f"depth_max_m: {valid_depth.max():.4f}")
            print(f"depth_mean_m: {valid_depth.mean():.4f}")
        print(f"saved: {color_path}")
        print(f"saved: {depth_gray_path}")
        print(f"saved: {depth_colormap_path}")
        print(f"saved: {point_cloud_path}")
        print(f"saved: {depth_m_path}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    capture_realsense_depth()
