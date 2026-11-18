"""
Sample Road Media Generator
Creates realistic synthetic road images and video feeds with potholes and crack distresses
for offline testing, validation, and dashboard demonstrations.
"""

import os
import cv2
import numpy as np


def generate_asphalt_texture(width: int = 1280, height: int = 720) -> np.ndarray:
    """Generate realistic textured asphalt pavement background."""
    # Base asphalt color (dark slate gray)
    base_color = np.array([55, 58, 62], dtype=np.uint8)
    asphalt = np.full((height, width, 3), base_color, dtype=np.uint8)

    # Add Gaussian pebble noise
    noise = np.random.normal(0, 18, (height, width)).astype(np.int16)
    for c in range(3):
        channel = asphalt[:, :, c].astype(np.int16) + noise
        asphalt[:, :, c] = np.clip(channel, 20, 120).astype(np.uint8)

    # Draw lane road markings (yellow and white road lines)
    # White dashed lane marker on left
    cv2.line(asphalt, (int(width * 0.15), 0), (int(width * 0.05), height), (220, 220, 220), 12)
    # Yellow double center line
    cv2.line(asphalt, (int(width * 0.90), 0), (int(width * 0.98), height), (0, 215, 255), 10)

    return asphalt


def draw_pothole(img: np.ndarray, center: tuple, radius_x: int, radius_y: int) -> None:
    """Draw realistic road pothole with dark depth gradient and irregular erosion boundary."""
    cx, cy = center
    num_pts = 36
    angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
    # Add random perimeter jitter for realistic crater erosion
    radii_x = radius_x * (1 + 0.15 * np.sin(3 * angles) + 0.12 * np.random.randn(num_pts))
    radii_y = radius_y * (1 + 0.15 * np.cos(4 * angles) + 0.12 * np.random.randn(num_pts))

    pts = np.zeros((num_pts, 2), dtype=np.int32)
    for i in range(num_pts):
        pts[i, 0] = int(cx + radii_x[i] * np.cos(angles[i]))
        pts[i, 1] = int(cy + radii_y[i] * np.sin(angles[i]))

    # Outer eroded depression
    cv2.fillPoly(img, [pts], (25, 26, 28))
    # Inner deep crater
    inner_pts = (pts - np.array([cx, cy])) * 0.7 + np.array([cx, cy])
    cv2.fillPoly(img, [inner_pts.astype(np.int32)], (12, 14, 15))
    # Crater rim shadow highlight
    cv2.polylines(img, [pts], True, (15, 15, 18), 3)


def draw_longitudinal_crack(img: np.ndarray, start_x: int, start_y: int, length: int) -> None:
    """Draw jagged vertical/longitudinal crack."""
    cur_x = start_x
    cur_y = start_y
    step = 15
    steps = length // step
    for _ in range(steps):
        next_x = cur_x + np.random.randint(-6, 7)
        next_y = cur_y + step
        cv2.line(img, (cur_x, cur_y), (next_x, next_y), (10, 10, 12), 3)
        # Small branching fractures
        if np.random.rand() > 0.6:
            bx = next_x + np.random.randint(-20, 21)
            by = next_y + np.random.randint(-5, 15)
            cv2.line(img, (next_x, next_y), (bx, by), (15, 15, 20), 2)
        cur_x, cur_y = next_x, next_y


def draw_alligator_crack(img: np.ndarray, x: int, y: int, width: int, height: int) -> None:
    """Draw interconnected polygonal alligator cracking network."""
    # Draw intersecting polygonal fractures
    grid_rows = 5
    grid_cols = 5
    dx = width // grid_cols
    dy = height // grid_rows

    grid = []
    for r in range(grid_rows + 1):
        row = []
        for c in range(grid_cols + 1):
            px = x + c * dx + np.random.randint(-dx // 3, dx // 3 + 1)
            py = y + r * dy + np.random.randint(-dy // 3, dy // 3 + 1)
            row.append((px, py))
        grid.append(row)

    # Draw horizontal and vertical connecting fissures
    for r in range(grid_rows + 1):
        for c in range(grid_cols + 1):
            if c < grid_cols:
                cv2.line(img, grid[r][c], grid[r][c + 1], (14, 14, 16), 2)
            if r < grid_rows:
                cv2.line(img, grid[r][c], grid[r + 1][c], (14, 14, 16), 2)
            if r < grid_rows and c < grid_cols and np.random.rand() > 0.4:
                cv2.line(img, grid[r][c], grid[r + 1][c + 1], (18, 18, 22), 2)


def draw_transverse_crack(img: np.ndarray, start_x: int, start_y: int, length: int) -> None:
    """Draw horizontal/transverse crack running across pavement."""
    cur_x = start_x
    cur_y = start_y
    step = 20
    steps = length // step
    for _ in range(steps):
        next_x = cur_x + step
        next_y = cur_y + np.random.randint(-7, 8)
        cv2.line(img, (cur_x, cur_y), (next_x, next_y), (12, 12, 15), 3)
        cur_x, cur_y = next_x, next_y


def create_sample_image(output_path: str = "sample_road.jpg") -> str:
    """Generate and save a high-resolution synthetic road distress test image."""
    img = generate_asphalt_texture(1280, 720)

    # 1. High Severity Pothole (center-left)
    draw_pothole(img, (450, 480), 95, 65)

    # 2. Medium Severity Alligator Cracking (center-right)
    draw_alligator_crack(img, 720, 360, 240, 160)

    # 3. Longitudinal Crack (mid road)
    draw_longitudinal_crack(img, 280, 220, 300)

    # 4. Transverse Crack (spanning across road)
    draw_transverse_crack(img, 350, 260, 320)

    # Save to disk
    cv2.imwrite(output_path, img)
    print(f"Created sample image: {output_path}")
    return output_path


def create_sample_video(output_path: str = "sample_road.mp4", num_frames: int = 75, fps: int = 24) -> str:
    """Generate and save a synthetic road inspection driving video."""
    width, height = 960, 540
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    base_road = generate_asphalt_texture(width, height + 400)

    # Add distresses at fixed longitudinal roadway positions
    # Pothole (appears mid video)
    draw_pothole(base_road, (width // 2 - 80, 600), 70, 50)
    # Alligator cracking (appears early)
    draw_alligator_crack(base_road, width // 2 + 60, 350, 180, 130)
    # Longitudinal crack
    draw_longitudinal_crack(base_road, 250, 200, 280)
    # Transverse crack
    draw_transverse_crack(base_road, 280, 480, 260)

    # Scroll camera window forward to simulate driving
    max_offset = 350
    for f in range(num_frames):
        # Progressively move viewing window
        y_offset = int((f / num_frames) * max_offset)
        frame = base_road[y_offset : y_offset + height, 0:width].copy()

        # Add camera vibration jitter
        if np.random.rand() > 0.5:
            jitter = np.random.randint(-1, 2)
            M = np.float32([[1, 0, 0], [0, 1, jitter]])
            frame = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_REFLECT)

        out.write(frame)

    out.release()
    print(f"Created sample video: {output_path} ({num_frames} frames @ {fps} FPS)")
    return output_path


if __name__ == "__main__":
    create_sample_image("sample_road.jpg")
    create_sample_video("sample_road.mp4")
