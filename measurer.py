"""
CV Measurement & Severity Mapping Engine Module
Performs Region of Interest (ROI) extraction, asphalt noise reduction,
morphological edge detection, Green's Theorem contour integration,
ground-plane spatial metric calibration, and severity tier mapping.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger("SeverityMeasurer")


class SeverityMeasurer:
    """
    Computer Vision measurement and damage severity classification engine.
    Computes precise surface area using Green's Theorem and maps structural risk.
    """

    # Severity Thresholds (in cm²)
    LOW_SEVERITY_LIMIT: float = 150.0    # Area < 150 cm² -> Low Severity
    HIGH_SEVERITY_LIMIT: float = 500.0   # Area >= 500 cm² -> High Severity (150 <= Area < 500 -> Medium)

    # Color Styling (BGR format for OpenCV, RGB format for PIL/Streamlit)
    COLOR_MAP = {
        "Low Severity": {
            "bgr": (0, 230, 118),       # Bright Green
            "rgb": (0, 230, 118),
            "hex": "#00E676"
        },
        "Medium Severity": {
            "bgr": (0, 165, 255),       # Orange / Amber
            "rgb": (255, 165, 0),
            "hex": "#FF9100"
        },
        "High Severity": {
            "bgr": (0, 0, 255),         # Crimson Red
            "rgb": (255, 23, 68),
            "hex": "#FF1744"
        }
    }

    def __init__(self, spatial_calibration_k: float = 0.15):
        """
        Initialize the Severity Measurer.

        Args:
            spatial_calibration_k: Ground-plane spatial calibration constant 'k' (cm/pixel).
                                   Formula: metric_area_cm2 = pixel_area * (k ** 2)
        """
        self.k = spatial_calibration_k

    def update_calibration_k(self, new_k: float) -> None:
        """Update spatial calibration constant dynamically."""
        if new_k > 0:
            self.k = new_k

    def process_roi(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        blur_kernel: Tuple[int, int] = (5, 5),
        canny_thresh1: int = 50,
        canny_thresh2: int = 150
    ) -> Dict[str, Any]:
        """
        Extract ROI sub-array, execute morphological edge detection, calculate
        contour area via Green's Theorem, and classify severity.

        Args:
            frame: Full original frame (BGR np.ndarray).
            box: Bounding box coordinates (x1, y1, x2, y2).
            blur_kernel: Gaussian kernel size for texture suppression.
            canny_thresh1: Lower threshold for hysteresis in Canny.
            canny_thresh2: Upper threshold for hysteresis in Canny.

        Returns:
            Dictionary containing:
                - 'pixel_area': float
                - 'metric_area_cm2': float
                - 'severity': str ("Low Severity" | "Medium Severity" | "High Severity")
                - 'color_bgr': Tuple[int, int, int]
                - 'color_rgb': Tuple[int, int, int]
                - 'color_hex': str
                - 'contour_count': int
                - 'cropped_roi': np.ndarray (BGR)
                - 'edge_map': np.ndarray (Grayscale)
                - 'annotated_roi': np.ndarray (BGR with drawn contours)
        """
        try:
            x1, y1, x2, y2 = box
            fh, fw = frame.shape[:2]

            # Coordinate bounds clamping
            x1 = max(0, min(x1, fw - 1))
            y1 = max(0, min(y1, fh - 1))
            x2 = max(0, min(x2, fw))
            y2 = max(0, min(y2, fh))

            if (x2 - x1) <= 2 or (y2 - y1) <= 2:
                return self._empty_result((x1, y1, x2, y2))

            # 1. Dynamic sub-array ROI crop
            roi = frame[y1:y2, x1:x2].copy()

            # 2. Grayscale conversion
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 3. Gaussian Blur to suppress high-frequency rough asphalt texture noise
            k_w = blur_kernel[0] if blur_kernel[0] % 2 != 0 else blur_kernel[0] + 1
            k_h = blur_kernel[1] if blur_kernel[1] % 2 != 0 else blur_kernel[1] + 1
            blurred = cv2.GaussianBlur(gray, (k_w, k_h), 0)

            # 4. Morphological gradient / Canny Edge Detection
            edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)

            # Morphological dilation to connect fragmented crack contours
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated_edges = cv2.dilate(edges, morph_kernel, iterations=1)

            # 5. Contour Extraction
            contours, _ = cv2.findContours(
                dilated_edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # 6. Green's Theorem Contour Area Calculation:
            # cv2.contourArea computes line integral: 1/2 * sum(x_i * y_{i+1} - x_{i+1} * y_i)
            # Filter noise contours smaller than 5 pixels
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= 5]

            annotated_roi = roi.copy()
            if valid_contours:
                # Sum the enclosed areas of all valid distress contours in the ROI
                pixel_area = float(sum(cv2.contourArea(cnt) for cnt in valid_contours))
                cv2.drawContours(annotated_roi, valid_contours, -1, (0, 255, 255), 2)
            else:
                # Fallback: estimate area from ROI bounding area scaled by a filling factor
                pixel_area = float((x2 - x1) * (y2 - y1) * 0.35)

            # 7. Apply ground-plane spatial calibration constant:
            # metric_area = pixel_area * (k ** 2)
            metric_area_cm2 = round(pixel_area * (self.k ** 2), 2)

            # 8. Severity Matrix Evaluation
            severity = self.classify_severity(metric_area_cm2)
            colors = self.COLOR_MAP[severity]

            return {
                "pixel_area": round(pixel_area, 2),
                "metric_area_cm2": metric_area_cm2,
                "severity": severity,
                "color_bgr": colors["bgr"],
                "color_rgb": colors["rgb"],
                "color_hex": colors["hex"],
                "contour_count": len(valid_contours),
                "cropped_roi": roi,
                "edge_map": dilated_edges,
                "annotated_roi": annotated_roi
            }

        except Exception as e:
            logger.error(f"Error processing ROI {box}: {e}")
            return self._empty_result(box)

    def classify_severity(self, metric_area_cm2: float) -> str:
        """
        Map metric damage area to severity tier according to evaluation matrix.
          * Area < 150 cm² -> "Low Severity"
          * 150 cm² <= Area < 500 cm² -> "Medium Severity"
          * Area >= 500 cm² -> "High Severity"
        """
        if metric_area_cm2 < self.LOW_SEVERITY_LIMIT:
            return "Low Severity"
        elif metric_area_cm2 < self.HIGH_SEVERITY_LIMIT:
            return "Medium Severity"
        else:
            return "High Severity"

    def annotate_frame(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        measurement: Dict[str, Any],
        draw_hud: bool = True
    ) -> np.ndarray:
        """
        Render styled bounding box, severity badge, class name, and damage telemetry on frame.
        """
        annotated = frame.copy()
        x1, y1, x2, y2 = detection["box"]
        color_bgr = measurement["color_bgr"]
        cls_name = detection["class_name"]
        conf = detection.get("confidence", 0.0)
        severity = measurement["severity"]
        area_cm2 = measurement["metric_area_cm2"]

        # Main bounding box with rounded corner accents
        thickness = 2 if severity == "Low Severity" else (3 if severity == "Medium Severity" else 4)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, thickness)

        # Corner brackets for HUD aesthetic
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        if corner_len > 4:
            # Top-left
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color_bgr, thickness + 1)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color_bgr, thickness + 1)
            # Top-right
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color_bgr, thickness + 1)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color_bgr, thickness + 1)
            # Bottom-left
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color_bgr, thickness + 1)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color_bgr, thickness + 1)
            # Bottom-right
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color_bgr, thickness + 1)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color_bgr, thickness + 1)

        if draw_hud:
            # Banner label above bounding box
            label_text = f"{cls_name} ({conf:.0%}) | {severity} | {area_cm2} cm2"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

            badge_y1 = max(0, y1 - text_h - 10)
            badge_y2 = y1
            badge_x2 = min(frame.shape[1], x1 + text_w + 14)

            # Dark translucent backing badge
            cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), (18, 18, 18), -1)
            # Left color bar
            cv2.rectangle(annotated, (x1, badge_y1), (x1 + 4, badge_y2), color_bgr, -1)
            # Text label
            cv2.putText(
                annotated,
                label_text,
                (x1 + 8, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )

        return annotated

    def _empty_result(self, box: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Return safe fallback result when ROI is invalid or empty."""
        colors = self.COLOR_MAP["Low Severity"]
        return {
            "pixel_area": 0.0,
            "metric_area_cm2": 0.0,
            "severity": "Low Severity",
            "color_bgr": colors["bgr"],
            "color_rgb": colors["rgb"],
            "color_hex": colors["hex"],
            "contour_count": 0,
            "cropped_roi": np.zeros((10, 10, 3), dtype=np.uint8),
            "edge_map": np.zeros((10, 10), dtype=np.uint8),
            "annotated_roi": np.zeros((10, 10, 3), dtype=np.uint8)
        }
