"""Visualization utilities for traffic sign detection and recognition.

Draws bounding boxes, class labels, and similarity scores on images,
producing annotated output similar to typical object detection displays.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class AnnotatedDetection:
    """A detection with recognition results attached.

    Attributes:
        bbox: (x1, y1, x2, y2) bounding box in pixel coordinates.
        detection_confidence: Object detection confidence.
        recognized_class: Class name from gallery matching, or None if unknown.
        similarity_score: Cosine similarity to best gallery match.
        all_scores: Similarity scores for all gallery classes.
    """
    bbox: tuple[int, int, int, int]
    detection_confidence: float
    recognized_class: Optional[str]
    similarity_score: float
    all_scores: dict[str, float]


# Color palette for different classes (RGB)
CLASS_COLORS = [
    (255, 50, 50),    # red
    (50, 200, 50),    # green
    (50, 100, 255),   # blue
    (255, 200, 0),    # yellow
    (200, 50, 255),   # purple
    (255, 128, 0),    # orange
    (0, 220, 220),    # cyan
    (255, 80, 180),   # pink
    (128, 255, 128),  # light green
    (180, 180, 255),  # light blue
]

UNKNOWN_COLOR = (160, 160, 160)  # gray for unknown signs


def _get_color(class_name: Optional[str], class_color_map: dict) -> tuple:
    """Get a consistent color for a class name."""
    if class_name is None:
        return UNKNOWN_COLOR
    if class_name not in class_color_map:
        idx = len(class_color_map) % len(CLASS_COLORS)
        class_color_map[class_name] = CLASS_COLORS[idx]
    return class_color_map[class_name]


def _get_font(size: int = 16) -> ImageFont.FreeTypeFont:
    """Try to load a decent font, fall back to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    detections: list[AnnotatedDetection],
    line_width: int = 3,
    font_size: int = 16,
    show_similarity: bool = True,
    show_unknown: bool = True,
) -> Image.Image:
    """Draw bounding boxes with labels and similarity scores on an image.

    Args:
        image: Original PIL Image (RGB).
        detections: List of AnnotatedDetection results.
        line_width: Bounding box line width.
        font_size: Label font size.
        show_similarity: Whether to show similarity scores.
        show_unknown: Whether to draw boxes for unknown detections.

    Returns:
        Annotated PIL Image.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = _get_font(font_size)

    class_color_map: dict[str, tuple] = {}

    for det in detections:
        if det.recognized_class is None and not show_unknown:
            continue

        x1, y1, x2, y2 = det.bbox
        color = _get_color(det.recognized_class, class_color_map)

        # Draw bounding box
        for i in range(line_width):
            draw.rectangle(
                [x1 - i, y1 - i, x2 + i, y2 + i],
                outline=color,
            )

        # Build label text
        if det.recognized_class is not None:
            label = det.recognized_class.replace("_", " ")
            if show_similarity:
                label += f" {det.similarity_score:.2f}"
        else:
            label = "Unknown"
            if show_similarity:
                label += f" ({det.similarity_score:.2f})"

        # Measure text size
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        padding = 4
        label_bg_y1 = max(0, y1 - text_h - 2 * padding)
        label_bg_y2 = y1

        # If label would go above image, put it inside the box
        if label_bg_y1 == 0 and y1 < text_h + 2 * padding:
            label_bg_y1 = y1
            label_bg_y2 = y1 + text_h + 2 * padding

        # Draw label background
        draw.rectangle(
            [x1, label_bg_y1, x1 + text_w + 2 * padding, label_bg_y2],
            fill=color,
        )

        # Draw label text (white or black for contrast)
        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

        draw.text(
            (x1 + padding, label_bg_y1 + padding),
            label,
            fill=text_color,
            font=font,
        )

    return annotated


def draw_detections_with_panel(
    image: Image.Image,
    detections: list[AnnotatedDetection],
    line_width: int = 3,
    font_size: int = 16,
    panel_width: int = 300,
) -> Image.Image:
    """Draw detections with a side panel showing all scores.

    Creates an annotated image with bounding boxes on the left and a
    detailed score panel on the right showing all class similarity scores.

    Args:
        image: Original PIL Image (RGB).
        detections: List of AnnotatedDetection results.
        line_width: Bounding box line width.
        font_size: Label font size.
        panel_width: Width of the side info panel.

    Returns:
        Annotated PIL Image with side panel.
    """
    # Draw boxes on the image
    annotated = draw_detections(
        image, detections, line_width=line_width, font_size=font_size,
    )

    if not detections:
        return annotated

    # Create side panel
    img_w, img_h = annotated.size
    panel_height = max(img_h, len(detections) * 120)
    panel = Image.new("RGB", (panel_width, panel_height), (30, 30, 30))
    panel_draw = ImageDraw.Draw(panel)
    small_font = _get_font(max(12, font_size - 2))
    title_font = _get_font(font_size)

    class_color_map: dict[str, tuple] = {}
    y_offset = 10

    for i, det in enumerate(detections):
        color = _get_color(det.recognized_class, class_color_map)

        # Detection header
        if det.recognized_class:
            header = f"#{i + 1}: {det.recognized_class.replace('_', ' ')}"
        else:
            header = f"#{i + 1}: Unknown"

        panel_draw.text((10, y_offset), header, fill=color, font=title_font)
        y_offset += 22

        panel_draw.text(
            (15, y_offset),
            f"Similarity: {det.similarity_score:.4f}",
            fill=(200, 200, 200),
            font=small_font,
        )
        y_offset += 18

        panel_draw.text(
            (15, y_offset),
            f"Detection conf: {det.detection_confidence:.4f}",
            fill=(160, 160, 160),
            font=small_font,
        )
        y_offset += 20

        # Show top-5 class scores
        if det.all_scores:
            sorted_scores = sorted(
                det.all_scores.items(), key=lambda x: x[1], reverse=True
            )[:5]
            for cls_name, score in sorted_scores:
                bar_len = int(score * 100)
                bar_color = color if cls_name == det.recognized_class else (100, 100, 100)
                panel_draw.rectangle(
                    [15, y_offset + 2, 15 + bar_len, y_offset + 12],
                    fill=bar_color,
                )
                panel_draw.text(
                    (120, y_offset),
                    f"{cls_name}: {score:.3f}",
                    fill=(180, 180, 180),
                    font=small_font,
                )
                y_offset += 16

        y_offset += 15  # gap between detections

    # Combine image and panel
    combined = Image.new("RGB", (img_w + panel_width, max(img_h, panel_height)), (30, 30, 30))
    combined.paste(annotated, (0, 0))
    combined.paste(panel, (img_w, 0))

    return combined


def save_annotated(
    image: Image.Image,
    detections: list[AnnotatedDetection],
    output_path: str,
    with_panel: bool = False,
    **kwargs,
) -> str:
    """Draw detections and save annotated image to a file.

    Args:
        image: Original PIL Image.
        detections: List of AnnotatedDetection results.
        output_path: Path to save the annotated image.
        with_panel: If True, add a side panel with detailed scores.
        **kwargs: Additional arguments passed to draw_detections.

    Returns:
        Path to the saved image.
    """
    if with_panel:
        result = draw_detections_with_panel(image, detections, **kwargs)
    else:
        result = draw_detections(image, detections, **kwargs)

    result.save(output_path)
    return output_path
