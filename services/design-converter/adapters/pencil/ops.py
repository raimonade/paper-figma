"""
Pencil batch_design Operations Builder
======================================

Builds Pencil MCP batch_design operation scripts from UNNode trees.

Pencil's batch_design tool uses a JavaScript-like script syntax:
  - I(parent, {...})   → Insert a new node
  - C(nodeId, parent)  → Copy an existing node
  - R(nodeId, {...})   → Replace a node
  - U(nodeId, {...})   → Update a node
  - D(nodeId)          → Delete a node
  - M(nodeId, parent)  → Move a node
  - G(nodeId, type, prompt) → Generate/apply image

Each operation returns a binding (node ID) that can be used in subsequent ops.

Reference: Pencil MCP batch_design tool documentation
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONV_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _CONV_ROOT not in sys.path:
    sys.path.insert(0, _CONV_ROOT)

from ir.nodes import (
    AlignItems,
    GradientType,
    ImageFillMode,
    JustifyContent,
    LayoutMode,
    NodeType,
    SizingMode,
    StrokeAlign,
    TextAlign,
    TextTransform,
    UNBlur,
    UNColor,
    UNCornerRadius,
    UNDropShadow,
    UNGradientFill,
    UNGradientStop,
    UNImageFill,
    UNNode,
    UNSolidFill,
    UNStroke,
    UNTextStyle,
    UNVariableBinding,
)

log = logging.getLogger(__name__)

# Counter for generating unique binding names
_binding_counter = 0


def _reset_binding_counter() -> None:
    """Reset the binding name counter (call before building a new tree)."""
    global _binding_counter
    _binding_counter = 0


def _new_binding(prefix: str = "n") -> str:
    """Generate a unique binding name."""
    global _binding_counter
    _binding_counter += 1
    return f"{prefix}{_binding_counter}"


def _escape_string(s: str) -> str:
    """Escape a string for use in JavaScript."""
    # Escape backslashes, quotes, and newlines
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("\t", "\\t")
    return s


def _color_to_hex(color: UNColor) -> str:
    """Convert UNColor to hex string (#RRGGBB or #RRGGBBAA)."""
    r = int(round(color.r * 255))
    g = int(round(color.g * 255))
    b = int(round(color.b * 255))
    a = color.a

    if a >= 1.0:
        return f"#{r:02X}{g:02X}{b:02X}"
    else:
        ai = int(round(a * 255))
        return f"#{r:02X}{g:02X}{b:02X}{ai:02X}"


def _sizing_to_pencil(mode: SizingMode, value: float) -> str:
    """Convert UNSize sizing mode to Pencil sizing string."""
    if mode == SizingMode.FIXED:
        return str(round(value, 2))
    elif mode == SizingMode.FILL:
        return '"fill_container"'
    elif mode == SizingMode.HUG:
        return f'"fit_content({int(round(value))})"'
    else:
        return str(round(value, 2))


def _layout_mode_to_pencil(mode: LayoutMode) -> str:
    """Convert LayoutMode to Pencil layout string."""
    mapping = {
        LayoutMode.NONE: '"none"',
        LayoutMode.HORIZONTAL: '"horizontal"',
        LayoutMode.VERTICAL: '"vertical"',
    }
    return mapping.get(mode, '"none"')


def _text_align_to_pencil(align: TextAlign) -> str:
    """Convert TextAlign to Pencil textAlign string."""
    mapping = {
        TextAlign.LEFT: '"left"',
        TextAlign.CENTER: '"center"',
        TextAlign.RIGHT: '"right"',
        TextAlign.JUSTIFY: '"justify"',
    }
    return mapping.get(align, '"left"')


def _justify_to_pencil(justify: JustifyContent) -> str:
    """Convert JustifyContent to Pencil justifyContent string."""
    mapping = {
        JustifyContent.START: '"start"',
        JustifyContent.CENTER: '"center"',
        JustifyContent.END: '"end"',
        JustifyContent.SPACE_BETWEEN: '"space-between"',
        JustifyContent.SPACE_AROUND: '"space-around"',
    }
    return mapping.get(justify, '"start"')


def _align_to_pencil(align: AlignItems) -> str:
    """Convert AlignItems to Pencil alignItems string."""
    mapping = {
        AlignItems.START: '"start"',
        AlignItems.CENTER: '"center"',
        AlignItems.END: '"end"',
        AlignItems.STRETCH: '"stretch"',
    }
    return mapping.get(align, '"start"')


def _build_fill(fill: Union[UNSolidFill, UNGradientFill, UNImageFill]) -> str:
    """Build a fill expression for Pencil."""
    if isinstance(fill, UNSolidFill):
        if fill.color:
            return f'"{_color_to_hex(fill.color)}"'
        return '"#FFFFFF"'

    elif isinstance(fill, UNGradientFill):
        # Gradient fills are complex - use color as fallback for now
        if fill.stops and fill.stops[0].color:
            return f'"{_color_to_hex(fill.stops[0].color)}"'
        return '"#FFFFFF"'

    elif isinstance(fill, UNImageFill):
        # Image fills need to be applied separately via G() operation
        # Return transparent for now
        return '"#FFFFFF00"'

    return '"#FFFFFF"'


def _build_fills(fills: List[Union[UNSolidFill, UNGradientFill, UNImageFill]]) -> str:
    """Build fills array or single fill."""
    if not fills:
        return '"#FFFFFF00"'  # Transparent

    # Use first fill
    return _build_fill(fills[0])


def _build_corner_radius(cr: UNCornerRadius) -> str:
    """Build corner radius value."""
    if cr.is_uniform():
        return str(round(cr.tl, 2))
    return f"[{round(cr.tl, 2)}, {round(cr.tr, 2)}, {round(cr.br, 2)}, {round(cr.bl, 2)}]"


def _build_stroke(stroke: UNStroke) -> Optional[str]:
    """Build stroke properties dict string for Pencil batch_design.

    Uses a flat format to avoid nesting issues:
    {"strokeAlign": "inside", "strokeFill": "#FF8000", "strokeThickness": 2}
    """
    if not stroke or not stroke.fill:
        return None

    parts = []

    # Align
    align_map = {
        StrokeAlign.INSIDE: '"inside"',
        StrokeAlign.CENTER: '"center"',
        StrokeAlign.OUTSIDE: '"outside"',
    }
    if stroke.align:
        parts.append(f'"strokeAlign": {align_map.get(stroke.align, "\"center\"")}')

    # Stroke fill color
    if stroke.fill and isinstance(stroke.fill, UNSolidFill):
        color = stroke.fill.color
        if color:
            parts.append(f'"strokeFill": "{_color_to_hex(color)}"')

    # Thickness - use simple number for uniform stroke
    if stroke.thickness:
        t = stroke.thickness
        if t.all is not None:
            parts.append(f'"strokeThickness": {round(t.all, 2)}')

    if not parts:
        return None

    return ", ".join(parts)


def _build_text_style(ts: Optional[UNTextStyle]) -> str:
    """Build text style properties."""
    if not ts:
        return ""

    parts = []

    if ts.font_family:
        parts.append(f'"fontFamily": "{_escape_string(ts.font_family)}"')

    if ts.font_size:
        parts.append(f'"fontSize": {round(ts.font_size, 2)}')

    if ts.font_weight:
        parts.append(f'"fontWeight": "{ts.font_weight}"')

    if ts.letter_spacing is not None:
        parts.append(f'"letterSpacing": {round(ts.letter_spacing, 4)}')

    if ts.line_height is not None:
        parts.append(f'"lineHeight": {round(ts.line_height, 4)}')

    if ts.text_align:
        parts.append(f'"textAlign": {_text_align_to_pencil(ts.text_align)}')

    if ts.text_decoration:
        dec = ts.text_decoration.lower()
        if dec == "underline":
            parts.append('"underline": true')
        elif dec == "strikethrough":
            parts.append('"strikethrough": true')

    if ts.text_transform:
        transform = ts.text_transform.lower()
        # Pencil doesn't have textTransform, skip for now

    return ", ".join(parts)


def _build_node_props(node: UNNode, is_root: bool = False) -> str:
    """Build the properties dict for a node insertion."""
    props = []

    # Type
    type_map = {
        NodeType.FRAME: "frame",
        NodeType.TEXT: "text",
        NodeType.RECTANGLE: "rectangle",
        NodeType.ELLIPSE: "ellipse",
        NodeType.LINE: "line",
        NodeType.GROUP: "group",
        NodeType.PATH: "path",
    }
    props.append(f'"type": "{type_map.get(node.type, "frame")}"')

    # Name
    if node.name:
        props.append(f'"name": "{_escape_string(node.name)}"')

    # Position (only for root or non-layout parents)
    if is_root or True:  # Include x/y, they're ignored in flexbox anyway
        props.append(f'"x": {round(node.x, 2)}')
        props.append(f'"y": {round(node.y, 2)}')

    # Size
    props.append(f'"width": {_sizing_to_pencil(node.width.mode, node.width.value)}')
    props.append(f'"height": {_sizing_to_pencil(node.height.mode, node.height.value)}')

    # Rotation
    if node.rotation:
        props.append(f'"rotation": {round(node.rotation, 2)}')

    # Opacity
    if node.opacity is not None and node.opacity < 1.0:
        props.append(f'"opacity": {round(node.opacity, 4)}')

    # Visible
    if not node.visible:
        props.append('"enabled": false')

    # Frame-specific properties
    if node.type == NodeType.FRAME:
        # Layout
        props.append(f'"layout": {_layout_mode_to_pencil(node.layout if node.layout else LayoutMode.NONE)}')

        # Justify content and align items (only for flex layouts)
        if node.layout in (LayoutMode.HORIZONTAL, LayoutMode.VERTICAL):
            if node.justify_content and node.justify_content != JustifyContent.START:
                props.append(f'"justifyContent": {_justify_to_pencil(node.justify_content)}')
            if node.align_items and node.align_items != AlignItems.START:
                props.append(f'"alignItems": {_align_to_pencil(node.align_items)}')

        # Gap
        if node.gap:
            props.append(f'"gap": {round(node.gap, 2)}')

        # Padding
        if node.padding:
            p = node.padding
            if p.top == p.right == p.bottom == p.left:
                props.append(f'"padding": {round(p.top, 2)}')
            elif p.top == p.bottom and p.right == p.left:
                props.append(f'"padding": [{round(p.top, 2)}, {round(p.right, 2)}]')
            else:
                props.append(f'"padding": [{round(p.top, 2)}, {round(p.right, 2)}, {round(p.bottom, 2)}, {round(p.left, 2)}]')

        # Clip
        if node.clip_content:
            props.append('"clip": true')

        # Corner radius
        if node.corner_radius and (node.corner_radius.tl > 0 or node.corner_radius.tr > 0 or
                                    node.corner_radius.br > 0 or node.corner_radius.bl > 0):
            props.append(f'"cornerRadius": {_build_corner_radius(node.corner_radius)}')

        # Fill (background color)
        if node.fills:
            props.append(f'"fill": {_build_fills(node.fills)}')

        # Stroke (flat properties)
        if node.strokes:
            stroke_props = _build_stroke(node.strokes[0])
            if stroke_props:
                props.append(stroke_props)

    # Rectangle-specific properties
    elif node.type == NodeType.RECTANGLE:
        # Corner radius
        if node.corner_radius and (node.corner_radius.tl > 0 or node.corner_radius.tr > 0 or
                                    node.corner_radius.br > 0 or node.corner_radius.bl > 0):
            props.append(f'"cornerRadius": {_build_corner_radius(node.corner_radius)}')

        # Fill
        if node.fills:
            props.append(f'"fill": {_build_fills(node.fills)}')

        # Stroke (DISABLED - nested object format not supported by Pencil batch_design)
        # TODO: Investigate correct stroke format for Pencil
        # if node.strokes:
        #     stroke_str = _build_stroke(node.strokes[0])
        #     if stroke_str:
        #         props.append(f'"stroke": {stroke_str}')

    # Text-specific properties
    elif node.type == NodeType.TEXT:
        # Content
        if node.text_content:
            props.append(f'"content": "{_escape_string(node.text_content)}"')

        # Text style
        ts_props = _build_text_style(node.text_style)
        if ts_props:
            props.append(ts_props)

        # Fill (text color)
        if node.fills:
            props.append(f'"fill": {_build_fills(node.fills)}')

        # Text growth (auto sizing)
        if node.text_style and node.text_style.text_auto_resize:
            tsar = node.text_style.text_auto_resize
            if tsar.value == "auto":
                props.append('"textGrowth": "auto"')
            elif tsar.value == "width":
                props.append('"textGrowth": "fixed-width"')
            elif tsar.value == "height_and_width":
                props.append('"textGrowth": "fixed-width-height"')

    return "{" + ", ".join(props) + "}"


def build_batch_ops(
    node: UNNode,
    parent_ref: str = "document",
    is_root: bool = True,
) -> List[str]:
    """
    Build batch_design operations for a UNNode tree.

    Returns a list of operation strings that can be joined into a script.
    """
    if not node.visible:
        return []

    ops = []
    node_ref = _new_binding("n")

    # Build the insert operation
    props = _build_node_props(node, is_root=is_root)

    # Determine how to reference the parent:
    # - "document" is a special keyword, don't quote
    # - Binding names (like n1, artboard1) are variables, don't quote
    # - Actual node IDs (like "s5d65") need quotes
    if parent_ref == "document" or parent_ref.isidentifier():
        # Variable/special reference - no quotes
        ops.append(f'{node_ref}=I({parent_ref}, {props})')
    else:
        # Actual node ID string - needs quotes
        ops.append(f'{node_ref}=I("{parent_ref}", {props})')

    # Handle image fills (need G operation)
    if node.fills:
        for fill in node.fills:
            if isinstance(fill, UNImageFill) and fill.url:
                # Apply image via G operation
                ops.append(f'G({node_ref}, "stock", "{_escape_string(fill.url)}")')
                break

    # Handle effects (shadows, blur)
    if node.effects:
        for effect in node.effects:
            if isinstance(effect, UNDropShadow):
                # Pencil effects are applied via update
                effect_props = []
                effect_props.append('"type": "shadow"')
                effect_props.append('"shadowType": "outer"')
                if effect.offset_x or effect.offset_y:
                    effect_props.append(f'"offset": {{\"x\": {round(effect.offset_x, 2)}, \"y\": {round(effect.offset_y, 2)}}}')
                if effect.blur:
                    effect_props.append(f'"blur": {round(effect.blur, 2)}')
                if effect.spread:
                    effect_props.append(f'"spread": {round(effect.spread, 2)}')
                if effect.color:
                    effect_props.append(f'"color": "{_color_to_hex(effect.color)}"')

                # Note: effects need U() operation after insert
                # For simplicity, we'll add them to children processing

            elif isinstance(effect, UNBlur):
                # Background blur
                pass  # TODO: Handle blur effects

    # Process children
    for child in node.children:
        child_ops = build_batch_ops(child, parent_ref=node_ref, is_root=False)
        ops.extend(child_ops)

    return ops


def build_batch_script(root: UNNode) -> str:
    """
    Build a complete batch_design script from a UNNode tree.

    Returns a JavaScript-like script string for Pencil's batch_design tool.
    """
    _reset_binding_counter()
    ops = build_batch_ops(root, parent_ref="document", is_root=True)
    return "\n".join(ops)


def build_batch_script_with_parent(root: UNNode, parent_id: str) -> str:
    """
    Build a batch_design script that inserts into a specific parent.

    Args:
        root: The root UNNode to convert
        parent_id: The Pencil node ID to insert into

    Returns:
        A batch_design script string
    """
    _reset_binding_counter()
    ops = build_batch_ops(root, parent_ref=parent_id, is_root=False)
    return "\n".join(ops)


# ---------------------------------------------------------------------------
# High-level API for PencilWriter
# ---------------------------------------------------------------------------

def build_artboard_script(node: UNNode) -> Tuple[str, str]:
    """
    Build a script to create an artboard from a FRAME node.

    Returns (script, binding_name) tuple.
    """
    _reset_binding_counter()

    # Artboard is a top-level frame
    binding = _new_binding("artboard")
    props = _build_node_props(node, is_root=True)

    # Ensure it has placeholder flag for work-in-progress
    if '"placeholder"' not in props:
        props = props.rstrip("}") + ', "placeholder": true}'

    script = f'{binding}=I(document, {props})'

    # Add children
    for child in node.children:
        child_ops = build_batch_ops(child, parent_ref=binding, is_root=False)
        script += "\n" + "\n".join(child_ops)

    # Remove placeholder when done
    script += f'\nU({binding}, {{"placeholder": false}})'

    return script, binding


def build_child_script(node: UNNode, parent_id: str) -> Tuple[str, str]:
    """
    Build a script to insert a node as a child of an existing parent.

    Returns (script, binding_name) tuple.
    """
    _reset_binding_counter()
    ops = build_batch_ops(node, parent_ref=parent_id, is_root=False)
    binding = ops[0].split("=")[0] if ops else "node"
    return "\n".join(ops), binding
