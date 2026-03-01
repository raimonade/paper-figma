"""
Paper Reader  —  Paper Design → Universal Node Tree
====================================================
Reads a Paper Design artboard (or any node) by calling ``get_jsx``
on the Paper MCP server, then parses the returned JSX markup into a
``UNNode`` tree that any downstream adapter can consume.

Parsing pipeline
----------------
1. ``PaperClient.get_jsx(node_id, mode="inline-styles")``
   → raw JSX string with ``style={{...}}`` props
2. ``JsxParser``  — lightweight recursive-descent tokenizer
   → flat list of ``_Tag`` tokens, then a tree of ``_Element`` dicts
3. ``_element_to_node``  — maps each ``_Element`` to a ``UNNode``
   using CSS utilities from ``utils/css.py`` and ``utils/color.py``
4. Full ``UNNode`` tree returned to the caller

Supported Paper node types
--------------------------
  <div>    → FRAME or RECTANGLE (depending on children)
  <span>   → TEXT (inline text)
  <p>      → TEXT  (paragraph)
  <h1>–<h6> → TEXT (heading)
  <img>    → IMAGE fill inside a FRAME
  <svg>    → PATH (geometry extracted from first <path> child)
  <circle> → ELLIPSE
  <rect>   → RECTANGLE

Usage
-----
    from adapters.paper import PaperReader

    with PaperReader() as reader:
        tree = reader.read_node("TO-0")   # Wallet Original artboard
        print(tree)

    # Or without context manager
    reader = PaperReader()
    reader.connect()
    tree = reader.read_node("1G2-0")
    reader.disconnect()
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Sys-path: allow running this file directly from the repo root
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONV_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _CONV_ROOT not in sys.path:
    sys.path.insert(0, _CONV_ROOT)

from adapters.base import BaseReader, NodeNotFoundError
from adapters.paper.client import PaperClient, PaperToolError
from ir.nodes import (
    AlignItems,
    BlendMode,
    GradientType,
    ImageFillMode,
    JustifyContent,
    LayoutMode,
    NodeType,
    SizingMode,
    StrokeAlign,
    TextAlign,
    TextAutoResize,
    TextTransform,
    UNBlur,
    UNColor,
    UNCornerRadius,
    UNDropShadow,
    UNGradientFill,
    UNGradientStop,
    UNImageFill,
    UNNode,
    UNPadding,
    UNSize,
    UNSolidFill,
    UNStroke,
    UNStrokeThickness,
    UNTextStyle,
)
from utils.color import (
    normalize_hex,
    parse_css_color,
    parse_css_gradient,
    parse_paper_background_image,
)
from utils.css import apply_css_to_node, parse_inline_style

log = logging.getLogger(__name__)


# ===========================================================================
# JSX Tokenizer / Parser
# ===========================================================================

# Regex patterns used by the tokenizer ─────────────────────────────────────

# Opening tag:  <div  or  <div   (no closing slash yet)
_RE_OPEN_TAG = re.compile(r"<([A-Za-z][A-Za-z0-9]*)\b", re.DOTALL)

# Self-closing slash at end of attributes:  />
_RE_SELF_CLOSE = re.compile(r"/\s*>")

# Closing tag:  </div>
_RE_CLOSE_TAG = re.compile(r"</\s*([A-Za-z][A-Za-z0-9]*)\s*>", re.DOTALL)

# JSX expression value:  ={{ ... }}   (matches balanced braces one-level deep)
# We use a manual scanner instead of regex because braces can nest.

# String attribute value: ="..."  or  ='...'
_RE_STR_ATTR = re.compile(r"""=(?:"([^"]*)"|'([^']*)')""", re.DOTALL)

# Plain attribute (no value):  disabled  checked
_RE_PLAIN_ATTR = re.compile(r"([A-Za-z][A-Za-z0-9_\-:.]*)")

# Number-ish string that should stay numeric
_RE_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")


@dataclass
class _Element:
    """Parsed representation of one JSX element."""

    tag: str  # "div", "span", "img", …
    props: Dict[str, Any] = field(default_factory=dict)
    children: List["_Element | str"] = field(default_factory=list)
    # Source position (byte offset in the original JSX string, for debug)
    start: int = 0


# ---------------------------------------------------------------------------
# Low-level JSX scanner helpers
# ---------------------------------------------------------------------------


def _scan_object(src: str, pos: int) -> Tuple[str, int]:
    """
    Starting at *pos* (which should be the opening ``{``), scan to the
    matching ``}`` respecting nested braces and JS string literals.
    Returns (raw_content_between_braces, new_pos_after_closing_brace).
    """
    assert src[pos] == "{"
    depth = 0
    i = pos
    in_single = False
    in_double = False
    while i < len(src):
        c = src[i]
        if in_single:
            if c == "\\" and i + 1 < len(src):
                i += 2
                continue
            if c == "'":
                in_single = False
        elif in_double:
            if c == "\\" and i + 1 < len(src):
                i += 2
                continue
            if c == '"':
                in_double = False
        else:
            if c == "'":
                in_single = True
            elif c == '"':
                in_double = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return src[pos + 1 : i], i + 1
        i += 1
    return src[pos + 1 :], len(src)


def _parse_js_value(val: str) -> Any:
    """
    Convert a raw JS value string (the inner part of ``{expr}``) to a
    Python value: number, bool, None, list, or string.
    """
    val = val.strip()
    if not val:
        return None
    # Boolean / null
    if val == "true":
        return True
    if val == "false":
        return False
    if val == "null" or val == "undefined":
        return None
    # Number
    if _RE_NUMBER.match(val):
        return float(val) if "." in val else int(val)
    # JS string  'xxx'  or  "xxx"
    if (val.startswith("'") and val.endswith("'")) or (
        val.startswith('"') and val.endswith('"')
    ):
        return val[1:-1]
    # Object literal  { key: value, … }
    if val.startswith("{"):
        return _parse_js_object(val)
    # Array literal  [ … ]
    if val.startswith("["):
        return _parse_js_array(val)
    # Fallback: return as string
    return val


def _parse_js_object(src: str) -> Dict[str, Any]:
    """
    Parse a JS object literal string like ``{key: 'val', num: 42}``.
    Returns a Python dict.  Handles nested objects and arrays.
    """
    src = src.strip()
    if src.startswith("{"):
        src = src[1:]
    if src.endswith("}"):
        src = src[:-1]

    result: Dict[str, Any] = {}
    i = 0
    src = src.strip()

    while i < len(src):
        # Skip whitespace and commas
        while i < len(src) and src[i] in " \t\n\r,":
            i += 1
        if i >= len(src):
            break

        # Read key (may be bare identifier or quoted string)
        key = ""
        if src[i] in ('"', "'"):
            quote = src[i]
            i += 1
            while i < len(src) and src[i] != quote:
                key += src[i]
                i += 1
            i += 1  # skip closing quote
        else:
            while i < len(src) and src[i] not in (": \t\n\r,{}[]"):
                key += src[i]
                i += 1

        key = key.strip()
        if not key:
            i += 1
            continue

        # Skip whitespace + colon
        while i < len(src) and src[i] in " \t\n\r:":
            i += 1
        if i >= len(src):
            break

        # Read value
        c = src[i]
        if c == "{":
            raw, i = _scan_object(src, i)
            val = _parse_js_object("{" + raw + "}")
        elif c == "[":
            raw, i = _scan_array(src, i)
            val = _parse_js_array("[" + raw + "]")
        elif c in ('"', "'"):
            quote = c
            i += 1
            s = ""
            while i < len(src) and src[i] != quote:
                if src[i] == "\\" and i + 1 < len(src):
                    i += 1
                    esc = src[i]
                    s += {"n": "\n", "t": "\t", "r": "\r"}.get(esc, esc)
                else:
                    s += src[i]
                i += 1
            i += 1  # skip closing quote
            val = s
        else:
            # Read until comma or end
            start = i
            while i < len(src) and src[i] not in (",\n\r}]"):
                i += 1
            val = _parse_js_value(src[start:i].strip())

        if key:
            result[key] = val

    return result


def _scan_array(src: str, pos: int) -> Tuple[str, int]:
    """Scan from opening ``[`` to matching ``]``."""
    assert src[pos] == "["
    depth = 0
    i = pos
    in_s = in_d = False
    while i < len(src):
        c = src[i]
        if in_s:
            if c == "\\" and i + 1 < len(src):
                i += 2
                continue
            if c == "'":
                in_s = False
        elif in_d:
            if c == "\\" and i + 1 < len(src):
                i += 2
                continue
            if c == '"':
                in_d = False
        else:
            if c == "'":
                in_s = True
            elif c == '"':
                in_d = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return src[pos + 1 : i], i + 1
        i += 1
    return src[pos + 1 :], len(src)


def _parse_js_array(src: str) -> List[Any]:
    """Parse a JS array literal like ``['a', 2, {x: 1}]``."""
    src = src.strip()
    if src.startswith("["):
        src = src[1:]
    if src.endswith("]"):
        src = src[:-1]
    if not src.strip():
        return []

    items: List[Any] = []
    i = 0
    while i < len(src):
        while i < len(src) and src[i] in " \t\n\r,":
            i += 1
        if i >= len(src):
            break
        c = src[i]
        if c == "{":
            raw, i = _scan_object(src, i)
            items.append(_parse_js_object("{" + raw + "}"))
        elif c == "[":
            raw, i = _scan_array(src, i)
            items.append(_parse_js_array("[" + raw + "]"))
        elif c in ('"', "'"):
            quote = c
            i += 1
            s = ""
            while i < len(src) and src[i] != quote:
                if src[i] == "\\" and i + 1 < len(src):
                    i += 1
                else:
                    s += src[i]
                i += 1
            i += 1
            items.append(s)
        else:
            start = i
            while i < len(src) and src[i] not in (",\n\r]"):
                i += 1
            items.append(_parse_js_value(src[start:i].strip()))

    return items


# ---------------------------------------------------------------------------
# Attribute parser
# ---------------------------------------------------------------------------


def _parse_attrs(src: str) -> Dict[str, Any]:
    """
    Parse JSX attribute string into a Python dict.

    Handles:
      - ``style={{key: val, ...}}``   → dict
      - ``className="..."``           → str
      - ``onClick={fn}``              → skipped (function refs)
      - ``disabled``                  → True  (boolean prop)
      - ``data-testid="x"``           → str
    """
    props: Dict[str, Any] = {}
    i = 0
    src = src.strip()

    while i < len(src):
        # Skip whitespace
        while i < len(src) and src[i] in " \t\n\r":
            i += 1
        if i >= len(src):
            break

        # Read attribute name
        attr_name = ""
        while i < len(src) and src[i] not in ("= \t\n\r/>{}"):
            attr_name += src[i]
            i += 1
        attr_name = attr_name.strip()
        if not attr_name:
            i += 1
            continue

        # Skip whitespace
        while i < len(src) and src[i] in " \t\n\r":
            i += 1
        if i >= len(src) or src[i] != "=":
            # Boolean prop
            props[attr_name] = True
            continue

        i += 1  # skip "="

        # Skip whitespace after "="
        while i < len(src) and src[i] in " \t\n\r":
            i += 1
        if i >= len(src):
            break

        c = src[i]
        if c == "{":
            # JSX expression value: {value} or {{object}}
            raw, i = _scan_object(src, i)
            if raw.strip().startswith("{"):
                # Double-brace: style={{ ... }}
                inner_raw, _ = _scan_object(raw.strip(), 0)
                props[attr_name] = _parse_js_object("{" + inner_raw + "}")
            else:
                props[attr_name] = _parse_js_value(raw)
        elif c in ('"', "'"):
            # String value: ="..." or ='...'
            quote = c
            i += 1
            s = ""
            while i < len(src) and src[i] != quote:
                if src[i] == "\\" and i + 1 < len(src):
                    i += 1
                    s += src[i]
                else:
                    s += src[i]
                i += 1
            i += 1  # skip closing quote
            props[attr_name] = s
        else:
            # Unquoted value (rare in JSX, e.g. tabIndex=0)
            start = i
            while i < len(src) and src[i] not in (" \t\n\r/>"):
                i += 1
            props[attr_name] = _parse_js_value(src[start:i])

    return props


# ---------------------------------------------------------------------------
# Full JSX tree parser
# ---------------------------------------------------------------------------


class JsxParser:
    """
    Recursive-descent parser for Paper's JSX output.

    Paper emits valid JSX:
      - One root element
      - Elements with ``style={{...}}`` and/or ``className="..."``
      - Text nodes as bare strings between tags
      - Self-closing ``<img />`` and ``<svg ... />`` tags

    This parser builds a tree of ``_Element`` objects.
    """

    def __init__(self, jsx: str) -> None:
        self._src = jsx
        self._pos = 0

    # ── Public entry-point ─────────────────────────────────────────────────

    def parse(self) -> Optional[_Element]:
        """Parse the JSX string and return the root ``_Element``, or None."""
        self._skip_whitespace()
        # Skip JSX comments  {/* ... */}
        self._src = re.sub(r"\{/\*.*?\*/\}", "", self._src, flags=re.DOTALL)
        # Skip HTML comments  <!-- ... -->
        self._src = re.sub(r"<!--.*?-->", "", self._src, flags=re.DOTALL)
        # Remove React imports / const declarations if any
        self._src = re.sub(
            r"^(import\s.*?;\s*|export\s+default\s+|const\s+\w+\s*=\s*\(.*?\)\s*=>\s*)",
            "",
            self._src,
            flags=re.DOTALL | re.MULTILINE,
        )
        # Paper wraps JSX in parentheses: ( <div>...</div> )
        # Strip leading/trailing parentheses
        self._src = self._src.strip()
        if self._src.startswith("(") and self._src.endswith(")"):
            self._src = self._src[1:-1].strip()
        self._pos = 0
        self._skip_whitespace()
        return self._parse_element()

    # ── Internal parser ────────────────────────────────────────────────────

    def _skip_whitespace(self) -> None:
        while self._pos < len(self._src) and self._src[self._pos] in " \t\n\r":
            self._pos += 1

    def _parse_element(self) -> Optional[_Element]:
        """Parse one element (including its children). Advances self._pos."""
        self._skip_whitespace()
        pos = self._pos
        src = self._src

        # Must start with "<"
        if pos >= len(src) or src[pos] != "<":
            return None

        # Find tag name
        m = _RE_OPEN_TAG.match(src, pos)
        if not m:
            return None
        tag = m.group(1).lower()  # normalise to lowercase
        self._pos = m.end()

        # Read attributes up to > or />
        attr_src, is_self_closing = self._read_attrs()
        props = _parse_attrs(attr_src)

        elem = _Element(tag=tag, props=props, start=pos)

        if is_self_closing:
            return elem

        # Read children until </tag>
        while self._pos < len(src):
            self._skip_whitespace()
            if self._pos >= len(src):
                break
            c = src[self._pos]

            # Check for closing tag
            m_close = _RE_CLOSE_TAG.match(src, self._pos)
            if m_close and m_close.group(1).lower() == tag:
                self._pos = m_close.end()
                break

            # Child element
            if c == "<":
                # Peek ahead — could be closing tag for an ancestor
                if src[self._pos : self._pos + 2] == "</":
                    break
                child_elem = self._parse_element()
                if child_elem:
                    elem.children.append(child_elem)
            elif c == "{":
                # JSX expression child  {someVar}  or  {/* comment */}
                raw, self._pos = _scan_object(src, self._pos)
                text = raw.strip().strip("'\"")
                if text and not text.startswith("/*"):
                    elem.children.append(text)
            else:
                # Plain text content
                end = self._pos
                while end < len(src) and src[end] not in ("<", "{"):
                    end += 1
                text = src[self._pos : end].strip()
                if text:
                    elem.children.append(text)
                self._pos = end

        return elem

    def _read_attrs(self) -> Tuple[str, bool]:
        """
        Read the attribute string between the tag name and ``>`` or ``/>``.

        Returns (attr_string, is_self_closing).
        Advances self._pos past the closing > or />.
        """
        src = self._src
        start = self._pos
        depth = 0  # track { } nesting so we don't stop inside style={{...}}
        in_single = in_double = False

        while self._pos < len(src):
            c = src[self._pos]
            if in_single:
                if c == "\\" and self._pos + 1 < len(src):
                    self._pos += 2
                    continue
                if c == "'":
                    in_single = False
            elif in_double:
                if c == "\\" and self._pos + 1 < len(src):
                    self._pos += 2
                    continue
                if c == '"':
                    in_double = False
            else:
                if c == "'":
                    in_single = True
                elif c == '"':
                    in_double = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                elif depth == 0:
                    if c == ">":
                        attr_src = src[start : self._pos]
                        self._pos += 1
                        return attr_src, False
                    if (
                        c == "/"
                        and self._pos + 1 < len(src)
                        and src[self._pos + 1] == ">"
                    ):
                        attr_src = src[start : self._pos]
                        self._pos += 2
                        return attr_src, True
            self._pos += 1

        return src[start : self._pos], False


# ===========================================================================
# Style → CSS dict normalisation
# ===========================================================================

_UNITLESS_PROPS = frozenset(
    {
        "opacity",
        "zIndex",
        "fontWeight",
        "lineHeight",
        "flex",
        "flexGrow",
        "flexShrink",
        "order",
        "aspectRatio",
    }
)

# CSS properties that Paper emits as numbers (pixels) without units
_PIXEL_PROPS = frozenset(
    {
        "width",
        "height",
        "minWidth",
        "minHeight",
        "maxWidth",
        "maxHeight",
        "top",
        "left",
        "right",
        "bottom",
        "margin",
        "marginTop",
        "marginRight",
        "marginBottom",
        "marginLeft",
        "padding",
        "paddingTop",
        "paddingRight",
        "paddingBottom",
        "paddingLeft",
        "borderRadius",
        "borderTopLeftRadius",
        "borderTopRightRadius",
        "borderBottomLeftRadius",
        "borderBottomRightRadius",
        "fontSize",
        "letterSpacing",
        "wordSpacing",
        "gap",
        "rowGap",
        "columnGap",
        "borderWidth",
        "outlineWidth",
        "strokeWidth",
        "offsetX",
        "offsetY",
        "blur",
        "spread",
        "left",
        "top",
        "x",
        "y",
    }
)


def _jsx_style_to_css_dict(style: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert a JSX style object (camelCase keys, numeric pixel values)
    to a standard CSS dict (camelCase keys, string values with units).

    Example:
        {"width": 390, "backgroundColor": "#050508"} →
        {"width": "390px", "backgroundColor": "#050508"}
    """
    out: Dict[str, str] = {}
    for key, val in style.items():
        if val is None:
            continue
        val_str = str(val)

        # Add "px" unit to numeric pixel properties
        if isinstance(val, (int, float)) and key in _PIXEL_PROPS:
            val_str = f"{val}px"
        elif isinstance(val, (int, float)) and key not in _UNITLESS_PROPS:
            # Guess: if it looks like a dimension, add px
            if val > 0 and key not in ("opacity", "zIndex", "flex"):
                val_str = f"{val}px"

        out[key] = val_str
    return out


# ===========================================================================
# Element → UNNode mapping
# ===========================================================================

# Tags that are always text containers
_TEXT_TAGS = frozenset({"span", "p", "h1", "h2", "h3", "h4", "h5", "h6", "label", "a"})

# Tags that are typically frame/layout containers
_FRAME_TAGS = frozenset(
    {"div", "section", "article", "main", "header", "footer", "nav", "aside"}
)

# Tags for shapes
_IMG_TAGS = frozenset({"img", "image"})
_SVG_TAGS = frozenset({"svg"})
_CIRCLE_TAGS = frozenset({"circle", "ellipse"})
_RECT_TAGS = frozenset({"rect"})
_PATH_TAGS = frozenset({"path", "polyline", "polygon", "line"})

# Figma/Paper flex-direction → LayoutMode
_DIRECTION_MAP = {
    "row": LayoutMode.HORIZONTAL,
    "row-reverse": LayoutMode.HORIZONTAL,
    "column": LayoutMode.VERTICAL,
    "column-reverse": LayoutMode.VERTICAL,
}

# CSS justify-content → JustifyContent
_JUSTIFY_MAP = {
    "flex-start": JustifyContent.START,
    "start": JustifyContent.START,
    "center": JustifyContent.CENTER,
    "flex-end": JustifyContent.END,
    "end": JustifyContent.END,
    "space-between": JustifyContent.SPACE_BETWEEN,
    "space-around": JustifyContent.SPACE_AROUND,
}

# CSS align-items → AlignItems
_ALIGN_MAP = {
    "flex-start": AlignItems.START,
    "start": AlignItems.START,
    "center": AlignItems.CENTER,
    "flex-end": AlignItems.END,
    "end": AlignItems.END,
    "stretch": AlignItems.STRETCH,
    "baseline": AlignItems.START,
}

# CSS text-align → TextAlign
_TEXT_ALIGN_MAP = {
    "left": TextAlign.LEFT,
    "center": TextAlign.CENTER,
    "right": TextAlign.RIGHT,
    "justify": TextAlign.JUSTIFY,
}

# CSS text-transform → TextTransform
_TEXT_TRANSFORM_MAP = {
    "uppercase": TextTransform.UPPERCASE,
    "lowercase": TextTransform.LOWERCASE,
    "capitalize": TextTransform.CAPITALIZE,
    "none": TextTransform.NONE,
}

# CSS font-weight aliases
_WEIGHT_ALIASES = {
    "thin": "100",
    "extralight": "200",
    "light": "300",
    "normal": "400",
    "regular": "400",
    "medium": "500",
    "semibold": "600",
    "bold": "700",
    "extrabold": "800",
    "black": "900",
}

# Heading tag → default font size
_HEADING_SIZES = {"h1": 32, "h2": 28, "h3": 24, "h4": 20, "h5": 18, "h6": 16}


def _px(val: str, default: float = 0.0) -> float:
    """Parse a CSS pixel string like '16px' or '1.5rem' → float."""
    if val is None:
        return default
    v = str(val).strip().lower()
    if v.endswith("px"):
        try:
            return float(v[:-2])
        except ValueError:
            return default
    if v.endswith("rem"):
        try:
            return float(v[:-3]) * 16.0
        except ValueError:
            return default
    try:
        return float(v)
    except ValueError:
        return default


def _collect_text(elem: _Element) -> str:
    """Recursively collect all text content from an element tree."""
    parts: List[str] = []
    for child in elem.children:
        if isinstance(child, str):
            parts.append(child)
        elif isinstance(child, _Element):
            parts.append(_collect_text(child))
    return " ".join(p for p in parts if p).strip()


def _is_text_only(elem: _Element) -> bool:
    """Return True if all children are strings (no nested elements)."""
    return all(isinstance(c, str) for c in elem.children)


# ---------------------------------------------------------------------------
# Tailwind CSS class parser (Paper uses Tailwind classes extensively)
# ---------------------------------------------------------------------------

# Tailwind spacing scale (4px base unit)
_TW_SPACING = {
    "px": "1px", "0": "0px", "0.5": "2px", "1": "4px", "1.5": "6px", "2": "8px",
    "2.5": "10px", "3": "12px", "3.5": "14px", "4": "16px", "5": "20px",
    "6": "24px", "7": "28px", "8": "32px", "9": "36px", "10": "40px",
    "11": "44px", "12": "48px", "14": "56px", "16": "64px", "20": "80px",
    "24": "96px", "28": "112px", "32": "128px", "36": "144px", "40": "160px",
    "44": "176px", "48": "192px", "52": "208px", "56": "224px", "60": "240px",
    "64": "256px", "72": "288px", "80": "320px", "96": "384px",
}

_TW_BORDER_RADIUS = {
    "none": "0px", "sm": "2px", "": "4px", "DEFAULT": "4px", "md": "6px",
    "lg": "8px", "xl": "12px", "2xl": "16px", "3xl": "24px", "full": "9999px",
}


def _parse_tailwind_class(cls: str) -> Dict[str, str]:
    """
    Parse a single Tailwind CSS class and return CSS properties.

    Handles common Paper-generated Tailwind classes:
      - w-[390px], h-[844px] → width/height
      - size-full → width: 100%; height: 100%
      - p-5, px-4, py-2, pt-3 → padding variants
      - bg-[#050508] → backgroundColor
      - flex, flex-col, flex-row → display/flexDirection
      - gap-3, gap-[10px] → gap
      - rounded-3xl → borderRadius
      - text-[14px] → fontSize
    """
    css: Dict[str, str] = {}

    # Width with arbitrary value: w-[390px], w-[50%]
    m = re.match(r"^w-\[([^\]]+)\]$", cls)
    if m:
        val = m.group(1)
        css["width"] = val if val.endswith(("px", "%", "rem", "em", "vw")) else f"{val}px"
        return css

    # Height with arbitrary value: h-[844px], h-[100%]
    m = re.match(r"^h-\[([^\]]+)\]$", cls)
    if m:
        val = m.group(1)
        css["height"] = val if val.endswith(("px", "%", "rem", "em", "vh")) else f"{val}px"
        return css

    # Size utilities: size-full, size-10, size-[100px]
    if cls == "size-full":
        css["width"] = "100%"
        css["height"] = "100%"
        return css
    m = re.match(r"^size-(\d+)$", cls)
    if m:
        sp = _TW_SPACING.get(m.group(1))
        if sp:
            css["width"] = sp
            css["height"] = sp
        return css
    m = re.match(r"^size-\[([^\]]+)\]$", cls)
    if m:
        val = m.group(1)
        css["width"] = val if val.endswith(("px", "%", "rem", "em")) else f"{val}px"
        css["height"] = css["width"]
        return css

    # Width presets: w-full, w-auto, w-10, w-1/2
    if cls == "w-full":
        css["width"] = "100%"
        return css
    if cls == "w-auto":
        css["width"] = "auto"
        return css
    if cls == "w-screen":
        css["width"] = "100vw"
        return css
    m = re.match(r"^w-(\d+)$", cls)
    if m:
        sp = _TW_SPACING.get(m.group(1))
        if sp:
            css["width"] = sp
        return css
    m = re.match(r"^w-1/(\d+)$", cls)
    if m:
        css["width"] = f"{100 // int(m.group(1))}%"
        return css

    # Height presets: h-full, h-auto, h-10, h-screen
    if cls == "h-full":
        css["height"] = "100%"
        return css
    if cls == "h-auto":
        css["height"] = "auto"
        return css
    if cls == "h-screen":
        css["height"] = "100vh"
        return css
    m = re.match(r"^h-(\d+)$", cls)
    if m:
        sp = _TW_SPACING.get(m.group(1))
        if sp:
            css["height"] = sp
        return css

    # Min/max dimensions
    m = re.match(r"^(min|max)-w-\[([^\]]+)\]$", cls)
    if m:
        prop = f"{m.group(1)}Width"
        val = m.group(2)
        css[prop] = val if val.endswith(("px", "%", "rem", "em", "vw")) else f"{val}px"
        return css
    m = re.match(r"^(min|max)-h-\[([^\]]+)\]$", cls)
    if m:
        prop = f"{m.group(1)}Height"
        val = m.group(2)
        css[prop] = val if val.endswith(("px", "%", "rem", "em", "vh")) else f"{val}px"
        return css

    # Padding: p-5, px-4, py-2, pt-3, pb-4, pl-2, pr-2
    for prefix, props in [
        ("p", ("paddingTop", "paddingRight", "paddingBottom", "paddingLeft")),
        ("px", ("paddingLeft", "paddingRight")),
        ("py", ("paddingTop", "paddingBottom")),
        ("pt", ("paddingTop",)),
        ("pr", ("paddingRight",)),
        ("pb", ("paddingBottom",)),
        ("pl", ("paddingLeft",)),
    ]:
        m = re.match(rf"^{prefix}-(\d+(?:\.\d+)?)$", cls)
        if m:
            sp = _TW_SPACING.get(m.group(1), f"{float(m.group(1)) * 4}px")
            for prop in props:
                css[prop] = sp
            return css
        m = re.match(rf"^{prefix}-\[(\d+(?:\.\d+)?)(px|rem|em)?\]$", cls)
        if m:
            val = m.group(1)
            unit = m.group(2) or "px"
            for prop in props:
                css[prop] = f"{val}{unit}"
            return css

    # Margin: m-4, mx-auto, my-2, mt-3, etc.
    for prefix, props in [
        ("m", ("marginTop", "marginRight", "marginBottom", "marginLeft")),
        ("mx", ("marginLeft", "marginRight")),
        ("my", ("marginTop", "marginBottom")),
        ("mt", ("marginTop",)),
        ("mr", ("marginRight",)),
        ("mb", ("marginBottom",)),
        ("ml", ("marginLeft",)),
    ]:
        m = re.match(rf"^{prefix}-(\d+(?:\.\d+)?)$", cls)
        if m:
            sp = _TW_SPACING.get(m.group(1), f"{float(m.group(1)) * 4}px")
            for prop in props:
                css[prop] = sp
            return css
        if cls == f"{prefix}-auto":
            for prop in props:
                css[prop] = "auto"
            return css

    # Gap: gap-3, gap-[10px]
    m = re.match(r"^gap-(\d+(?:\.\d+)?)$", cls)
    if m:
        sp = _TW_SPACING.get(m.group(1), f"{float(m.group(1)) * 4}px")
        css["gap"] = sp
        return css
    m = re.match(r"^gap-\[([^\]]+)\]$", cls)
    if m:
        val = m.group(1)
        css["gap"] = val if val.endswith(("px", "rem", "em")) else f"{val}px"
        return css

    # Display: flex, hidden, block
    if cls == "flex":
        css["display"] = "flex"
        return css
    if cls == "hidden":
        css["display"] = "none"
        return css
    if cls == "block":
        css["display"] = "block"
        return css
    if cls == "inline":
        css["display"] = "inline"
        return css
    if cls == "inline-block":
        css["display"] = "inline-block"
        return css

    # Flex direction
    if cls == "flex-row":
        css["flexDirection"] = "row"
        return css
    if cls == "flex-col":
        css["flexDirection"] = "column"
        return css
    if cls == "flex-row-reverse":
        css["flexDirection"] = "row-reverse"
        return css
    if cls == "flex-col-reverse":
        css["flexDirection"] = "column-reverse"
        return css

    # Flex wrap
    if cls == "flex-wrap":
        css["flexWrap"] = "wrap"
        return css
    if cls == "flex-nowrap":
        css["flexWrap"] = "nowrap"
        return css

    # Justify content
    for tw, jc in [
        ("justify-start", "flex-start"), ("justify-end", "flex-end"),
        ("justify-center", "center"), ("justify-between", "space-between"),
        ("justify-around", "space-around"), ("justify-evenly", "space-evenly"),
    ]:
        if cls == tw:
            css["justifyContent"] = jc
            return css

    # Align items
    for tw, ai in [
        ("items-start", "flex-start"), ("items-end", "flex-end"),
        ("items-center", "center"), ("items-baseline", "baseline"),
        ("items-stretch", "stretch"),
    ]:
        if cls == tw:
            css["alignItems"] = ai
            return css

    # Flex shrink/grow
    if cls == "shrink-0":
        css["flexShrink"] = "0"
        return css
    if cls == "shrink":
        css["flexShrink"] = "1"
        return css
    if cls == "grow-0":
        css["flexGrow"] = "0"
        return css
    if cls == "grow":
        css["flexGrow"] = "1"
        return css

    # Background color: bg-[#050508], bg-red-500
    m = re.match(r"^bg-\[([^\]]+)\]$", cls)
    if m:
        css["backgroundColor"] = m.group(1)
        return css

    # Text color: text-[#71717A], text-white, text-gray-500
    m = re.match(r"^text-\[([^\]]+)\]$", cls)
    if m:
        css["color"] = m.group(1)
        return css
    if cls == "text-white":
        css["color"] = "#FFFFFF"
        return css
    if cls == "text-black":
        css["color"] = "#000000"
        return css

    # Font size: text-[14px], text-sm, text-lg
    m = re.match(r"^text-\[(\d+(?:\.\d+)?)(px|rem|em)?\]$", cls)
    if m:
        val = m.group(1)
        unit = m.group(2) or "px"
        css["fontSize"] = f"{val}{unit}"
        return css
    for tw, size in [
        ("text-xs", "12px"), ("text-sm", "14px"), ("text-base", "16px"),
        ("text-lg", "18px"), ("text-xl", "20px"), ("text-2xl", "24px"),
    ]:
        if cls == tw:
            css["fontSize"] = size
            return css

    # Font weight: font-bold, font-semibold, etc.
    for tw, wt in [
        ("font-thin", "100"), ("font-extralight", "200"), ("font-light", "300"),
        ("font-normal", "400"), ("font-medium", "500"), ("font-semibold", "600"),
        ("font-bold", "700"), ("font-extrabold", "800"), ("font-black", "900"),
    ]:
        if cls == tw:
            css["fontWeight"] = wt
            return css

    # Font family
    if cls == "font-sans":
        css["fontFamily"] = "sans-serif"
        return css
    if cls == "font-serif":
        css["fontFamily"] = "serif"
        return css
    if cls == "font-mono":
        css["fontFamily"] = "monospace"
        return css

    # Border radius: rounded, rounded-lg, rounded-3xl, rounded-[8px]
    if cls == "rounded":
        css["borderRadius"] = "4px"
        return css
    m = re.match(r"^rounded-(\w+)$", cls)
    if m:
        key = m.group(1)
        if key in _TW_BORDER_RADIUS:
            css["borderRadius"] = _TW_BORDER_RADIUS[key]
        return css
    m = re.match(r"^rounded-\[([^\]]+)\]$", cls)
    if m:
        val = m.group(1)
        css["borderRadius"] = val if val.endswith(("px", "%", "rem", "em")) else f"{val}px"
        return css

    # Position
    if cls == "relative":
        css["position"] = "relative"
        return css
    if cls == "absolute":
        css["position"] = "absolute"
        return css
    if cls == "fixed":
        css["position"] = "fixed"
        return css
    if cls == "sticky":
        css["position"] = "sticky"
        return css

    # Positioning with arbitrary values: left-[50%], top-[20px]
    for prop, tw_prefix in [
        ("left", "left"), ("top", "top"), ("right", "right"), ("bottom", "bottom"),
    ]:
        m = re.match(rf"^{tw_prefix}-\[(\d+(?:\.\d+)?)(px|%|rem|em|vw|vh)?\]$", cls)
        if m:
            val = m.group(1)
            unit = m.group(2) or "px"
            css[prop] = f"{val}{unit}"
            return css

    # Overflow
    if cls == "overflow-clip":
        css["overflow"] = "hidden"
        return css
    if cls == "overflow-hidden":
        css["overflow"] = "hidden"
        return css
    if cls == "overflow-auto":
        css["overflow"] = "auto"
        return css
    if cls == "overflow-scroll":
        css["overflow"] = "scroll"
        return css

    # Opacity: opacity-50, opacity-[0.5]
    m = re.match(r"^opacity-(\d+)$", cls)
    if m:
        css["opacity"] = str(int(m.group(1)) / 100)
        return css
    m = re.match(r"^opacity-\[([^\]]+)\]$", cls)
    if m:
        css["opacity"] = m.group(1)
        return css

    # Text alignment
    if cls == "text-left":
        css["textAlign"] = "left"
        return css
    if cls == "text-center":
        css["textAlign"] = "center"
        return css
    if cls == "text-right":
        css["textAlign"] = "right"
        return css

    # Text transform
    if cls == "uppercase":
        css["textTransform"] = "uppercase"
        return css
    if cls == "lowercase":
        css["textTransform"] = "lowercase"
        return css
    if cls == "capitalize":
        css["textTransform"] = "capitalize"
        return css

    # Letter spacing: tracking-[-0.2px], tracking-tight
    m = re.match(r"^tracking-\[([^\]]+)\]$", cls)
    if m:
        val = m.group(1)
        css["letterSpacing"] = val if val.endswith(("px", "em", "rem")) else f"{val}px"
        return css

    # Line height: leading-4, leading-[18px]
    m = re.match(r"^leading-(\d+(?:\.\d+)?)$", cls)
    if m:
        sp = _TW_SPACING.get(m.group(1))
        if sp:
            css["lineHeight"] = sp
        return css
    m = re.match(r"^leading-\[([^\]]+)\]$", cls)
    if m:
        val = m.group(1)
        css["lineHeight"] = val if val.endswith(("px", "rem", "em", "%")) else f"{val}px"
        return css

    return css


def _parse_tailwind_classes(class_names: str) -> Dict[str, str]:
    """
    Parse a className string (space-separated classes) into CSS properties.
    Later classes override earlier ones for the same property.
    """
    result: Dict[str, str] = {}
    for cls in class_names.split():
        props = _parse_tailwind_class(cls)
        result.update(props)
    return result


def _get_style(elem: _Element) -> Dict[str, str]:
    """Extract the CSS dict from an element's style prop and Tailwind className."""
    css: Dict[str, str] = {}

    # 1. Parse Tailwind classes first (lower priority)
    class_names = elem.props.get("className", "")
    if class_names and isinstance(class_names, str):
        css.update(_parse_tailwind_classes(class_names))

    # 2. Parse inline style prop (higher priority, overrides Tailwind)
    raw = elem.props.get("style", {})
    if isinstance(raw, dict):
        css.update(_jsx_style_to_css_dict(raw))
    elif isinstance(raw, str):
        css.update(parse_inline_style(raw))

    return css


# ---------------------------------------------------------------------------
# Fill / stroke / effect extraction from a CSS dict
# ---------------------------------------------------------------------------


def _gradient_dict_to_fill(grad: Dict[str, Any]) -> UNGradientFill:
    """Convert a parsed gradient dict to UNGradientFill."""
    # Map gradient type string to enum
    type_map = {
        "linear": GradientType.LINEAR,
        "radial": GradientType.RADIAL,
        "angular": GradientType.ANGULAR,
        "diamond": GradientType.DIAMOND,
    }
    gradient_type = type_map.get(grad.get("type", "linear"), GradientType.LINEAR)

    # Convert stops
    stops = []
    for stop in grad.get("stops", []):
        color_hex = stop.get("color", "#000000FF")
        color = UNColor.from_hex(color_hex)
        position = stop.get("position", 0.0)
        stops.append(UNGradientStop(color=color, position=position))

    return UNGradientFill(
        gradient_type=gradient_type,
        rotation=grad.get("rotation", 180.0),
        stops=stops,
        opacity=grad.get("opacity", 1.0),
    )


def _extract_fills(css: Dict[str, str], node: UNNode) -> None:
    """Parse background/fill-related CSS properties and attach fills to node."""
    bg = css.get("background") or css.get("backgroundColor") or css.get("fill")
    bg_image = css.get("backgroundImage")

    if bg_image:
        # Try gradient first
        grad = parse_css_gradient(bg_image)
        if grad:
            node.fills.append(_gradient_dict_to_fill(grad))
            return
        # Try background-image URL  →  image fill
        m = re.search(r'url\(["\']?(.+?)["\']?\)', bg_image)
        if m:
            node.fills.append(UNImageFill(url=m.group(1), mode=ImageFillMode.FILL))
            return

    if bg and bg not in ("none", "transparent", "initial", "inherit"):
        # Might be a gradient expressed in background shorthand
        if "gradient" in bg:
            grad = parse_css_gradient(bg)
            if grad:
                node.fills.append(_gradient_dict_to_fill(grad))
                return
        # Solid color - parse to hex string, then convert to UNColor
        try:
            hex_color = parse_css_color(bg)
            if hex_color:
                color = UNColor.from_hex(hex_color)
                node.fills.append(UNSolidFill(color=color))
        except Exception:
            pass


def _extract_strokes(css: Dict[str, str], node: UNNode) -> None:
    """Parse border/stroke CSS properties and attach strokes to node."""
    border = css.get("border") or css.get("outline")
    border_color = css.get("borderColor")
    border_width_str = css.get("borderWidth")

    # Handle shorthand like "1px solid #333"
    if border and border not in ("none", "0"):
        m = re.match(r"(\d+(?:\.\d+)?)px\s+\w+\s+(.+)", border.strip())
        if m:
            try:
                bw = float(m.group(1))
                bc_hex = parse_css_color(m.group(2).strip())
                if bc_hex:
                    bc = UNColor.from_hex(bc_hex)
                    node.strokes.append(
                        UNStroke(
                            fill=UNSolidFill(color=bc),
                            thickness=UNStrokeThickness.uniform(bw),
                            align=StrokeAlign.INSIDE,
                        )
                    )
            except Exception:
                pass
        return

    if border_color and border_color not in ("none", "transparent"):
        bw = _px(border_width_str, 1.0) if border_width_str else 1.0
        try:
            bc_hex = parse_css_color(border_color)
            if bc_hex:
                bc = UNColor.from_hex(bc_hex)
                node.strokes.append(
                    UNStroke(
                        fill=UNSolidFill(color=bc),
                        thickness=UNStrokeThickness.uniform(bw),
                        align=StrokeAlign.INSIDE,
                    )
                )
        except Exception:
            pass


def _extract_shadows(css: Dict[str, str], node: UNNode) -> None:
    """Parse box-shadow / filter:drop-shadow CSS and attach effects."""
    box_shadow = css.get("boxShadow") or css.get("filter")
    if not box_shadow or box_shadow in ("none", ""):
        return

    # Simple single box-shadow: "0px 4px 16px 0px rgba(124,58,237,0.5)"
    # or "0 0 40px rgba(124,58,237,0.3)"
    pattern = re.compile(
        r"(-?\d+(?:\.\d+)?)px\s+(-?\d+(?:\.\d+)?)px\s+(\d+(?:\.\d+)?)px"
        r"(?:\s+(-?\d+(?:\.\d+)?)px)?"
        r"\s+(rgba?\([^)]+\)|#[0-9a-fA-F]{3,8}|\w+)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(box_shadow):
        try:
            ox = float(m.group(1))
            oy = float(m.group(2))
            blur = float(m.group(3))
            spread = float(m.group(4)) if m.group(4) else 0.0
            color_hex = parse_css_color(m.group(5))
            if color_hex:
                color = UNColor.from_hex(color_hex)
                node.effects.append(
                    UNDropShadow(
                        color=color,
                        offset_x=ox,
                        offset_y=oy,
                        blur=blur,
                        spread=spread,
                    )
                )
        except Exception:
            pass


def _extract_corner_radius(css: Dict[str, str]) -> UNCornerRadius:
    """Parse border-radius CSS and return a UNCornerRadius."""
    uniform = css.get("borderRadius")
    if uniform:
        r = _px(uniform)
        # Might be "50%" → treat as very large radius
        if "%" in str(uniform):
            r = 9999.0
        return UNCornerRadius.all(r)

    tl = _px(css.get("borderTopLeftRadius", "0"))
    tr = _px(css.get("borderTopRightRadius", "0"))
    br = _px(css.get("borderBottomRightRadius", "0"))
    bl = _px(css.get("borderBottomLeftRadius", "0"))
    return UNCornerRadius(tl=tl, tr=tr, br=br, bl=bl)


def _extract_opacity(css: Dict[str, str]) -> float:
    """Return opacity from CSS, default 1.0."""
    op = css.get("opacity")
    if op is None:
        return 1.0
    try:
        return max(0.0, min(1.0, float(op)))
    except ValueError:
        return 1.0


def _extract_layout(css: Dict[str, str], node: UNNode) -> None:
    """Parse flexbox CSS properties and set layout on node."""
    display = css.get("display", "").lower()
    flex_dir = css.get("flexDirection", "column").lower()

    if display != "flex":
        node.layout = LayoutMode.NONE
        return

    node.layout = _DIRECTION_MAP.get(flex_dir, LayoutMode.VERTICAL)
    node.gap = _px(css.get("gap") or css.get("rowGap") or css.get("columnGap"), 0.0)
    node.justify_content = _JUSTIFY_MAP.get(
        css.get("justifyContent", "flex-start").lower(), JustifyContent.START
    )
    node.align_items = _ALIGN_MAP.get(
        css.get("alignItems", "flex-start").lower(), AlignItems.START
    )

    # Padding
    pt = _px(css.get("paddingTop") or css.get("padding"), 0.0)
    pr = _px(css.get("paddingRight") or css.get("padding"), 0.0)
    pb = _px(css.get("paddingBottom") or css.get("padding"), 0.0)
    pl = _px(css.get("paddingLeft") or css.get("padding"), 0.0)
    # Shorthand padding overrides individual if both present
    if "padding" in css and not any(
        k in css for k in ("paddingTop", "paddingRight", "paddingBottom", "paddingLeft")
    ):
        p = _px(css["padding"])
        pt = pr = pb = pl = p
    node.padding = UNPadding(top=pt, right=pr, bottom=pb, left=pl)


def _extract_text_style(css: Dict[str, str], tag: str) -> UNTextStyle:
    """Build a UNTextStyle from CSS typography properties."""
    font_size = _px(css.get("fontSize"), _HEADING_SIZES.get(tag, 14.0))
    raw_weight = css.get("fontWeight", "400")
    font_weight = _WEIGHT_ALIASES.get(str(raw_weight).lower(), str(raw_weight))
    font_family = css.get("fontFamily", "Inter").split(",")[0].strip().strip("'\"")
    font_style = "italic" if "italic" in css.get("fontStyle", "").lower() else "normal"

    line_height_raw = css.get("lineHeight")
    line_height: Optional[float] = None
    if line_height_raw and line_height_raw not in ("normal", ""):
        lh_str = str(line_height_raw)
        if lh_str.endswith("px"):
            line_height = _px(lh_str)
        else:
            try:
                line_height = float(lh_str) * font_size
            except ValueError:
                line_height = None

    letter_spacing = _px(css.get("letterSpacing"), 0.0)

    text_align = _TEXT_ALIGN_MAP.get(
        css.get("textAlign", "left").lower(), TextAlign.LEFT
    )
    text_transform = _TEXT_TRANSFORM_MAP.get(
        css.get("textTransform", "none").lower(), TextTransform.NONE
    )
    text_decoration = css.get("textDecoration", "none")

    return UNTextStyle(
        font_family=font_family,
        font_size=font_size,
        font_weight=font_weight,
        font_style=font_style,
        line_height=line_height,
        letter_spacing=letter_spacing,
        text_align=text_align,
        text_transform=text_transform,
        text_decoration=text_decoration,
        text_auto_resize=TextAutoResize.WIDTH_HEIGHT,
    )


# ---------------------------------------------------------------------------
# Main element → UNNode converter
# ---------------------------------------------------------------------------

_node_counter = 0


def _next_node_id() -> str:
    global _node_counter
    _node_counter += 1
    return f"paper-{_node_counter}"


def _element_to_node(
    elem: _Element,
    parent_width: float = 0.0,
    parent_height: float = 0.0,
    depth: int = 0,
    dimension_map: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Optional[UNNode]:
    """
    Convert one ``_Element`` (and its subtree) into a ``UNNode``.

    Strategy:
      - <div> / <section> … → FRAME (with auto-layout if display:flex)
      - <span> / <p> / <h*> → TEXT
      - <img> → FRAME with image fill
      - <svg> → PATH (extract first <path> child) or FRAME
      - <circle> / <ellipse> → ELLIPSE
      - <rect> → RECTANGLE
    """
    tag = elem.tag.lower()
    css = _get_style(elem)

    # Determine size - track whether dimensions are percentage-based
    width_css = css.get("width", "")
    height_css = css.get("height", "")

    # Track sizing mode: explicit pixels, percentage, or auto
    width_is_fill = False  # 100% → FILL mode
    height_is_fill = False
    width_is_hug = False   # auto/hug content
    height_is_hug = False

    width = _px(width_css, parent_width)
    height = _px(height_css, parent_height)

    # Handle percentage-based dimensions
    if "%" in str(width_css):
        try:
            pct = float(str(width_css).replace("%", "")) / 100.0
            width = parent_width * pct
            if pct >= 1.0:  # 100% or more → FILL mode
                width_is_fill = True
        except ValueError:
            pass
    if "%" in str(height_css):
        try:
            pct = float(str(height_css).replace("%", "")) / 100.0
            height = parent_height * pct
            if pct >= 1.0:  # 100% or more → FILL mode
                height_is_fill = True
        except ValueError:
            pass

    # Auto dimensions → HUG mode
    if str(width_css).lower() in ("auto", ""):
        width_is_hug = True
    if str(height_css).lower() in ("auto", ""):
        height_is_hug = True

    # Position (absolute layout)
    x = _px(css.get("left") or css.get("x"), 0.0)
    y = _px(css.get("top") or css.get("y"), 0.0)

    opacity = _extract_opacity(css)
    corner_radius = _extract_corner_radius(css)

    # ------------------------------------------------------------------
    # TEXT nodes
    # ------------------------------------------------------------------
    if tag in _TEXT_TAGS or (
        tag in _FRAME_TAGS and _is_text_only(elem) and elem.children
    ):
        text_content = _collect_text(elem)
        if not text_content and tag not in _TEXT_TAGS:
            # Empty div → continue as frame
            pass
        else:
            ts = _extract_text_style(css, tag)
            # For text nodes: use HUG unless CSS explicitly specifies a dimension
            # (don't inherit parent dimensions for auto-sizing text)
            text_width = UNSize.hug()
            text_height = UNSize.hug()
            if width_css and str(width_css).lower() not in ("auto", "", "none"):
                text_width = UNSize.fixed(width)
            elif width_is_fill:
                text_width = UNSize.fill()
            if height_css and str(height_css).lower() not in ("auto", "", "none"):
                text_height = UNSize.fixed(height)
            elif height_is_fill:
                text_height = UNSize.fill()
            node = UNNode(
                type=NodeType.TEXT,
                id=_next_node_id(),
                name=elem.props.get("id", elem.props.get("data-name", tag)),
                x=x,
                y=y,
                width=text_width,
                height=text_height,
                text_content=text_content,
                text_style=ts,
                opacity=opacity,
                source_tool="paper",
            )
            # Text color
            color_str = css.get("color") or css.get("fill")
            if color_str and color_str not in ("none", "transparent", "inherit"):
                try:
                    c_hex = parse_css_color(color_str)
                    if c_hex:
                        c = UNColor.from_hex(c_hex)
                        node.fills.append(UNSolidFill(color=c))
                except Exception:
                    pass
            _extract_shadows(css, node)
            return node

    # ------------------------------------------------------------------
    # IMAGE nodes
    # ------------------------------------------------------------------
    if tag in _IMG_TAGS:
        src = elem.props.get("src", "")
        node = UNNode(
            type=NodeType.FRAME,
            id=_next_node_id(),
            name=elem.props.get("alt", elem.props.get("id", "image")),
            x=x,
            y=y,
            width=UNSize.fixed(max(width, 1.0)),
            height=UNSize.fixed(max(height, 1.0)),
            opacity=opacity,
            corner_radius=corner_radius,
            source_tool="paper",
        )
        mode_str = css.get("objectFit", "fill").lower()
        mode_map = {
            "fill": ImageFillMode.FILL,
            "contain": ImageFillMode.FIT,
            "cover": ImageFillMode.FILL,
            "none": ImageFillMode.FILL,
        }
        node.fills.append(
            UNImageFill(url=src, mode=mode_map.get(mode_str, ImageFillMode.FILL))
        )
        return node

    # ------------------------------------------------------------------
    # ELLIPSE nodes
    # ------------------------------------------------------------------
    if tag in _CIRCLE_TAGS:
        r = float(elem.props.get("r", elem.props.get("rx", width / 2 or 20)))
        cx = float(elem.props.get("cx", x))
        cy = float(elem.props.get("cy", y))
        w = h = r * 2
        node = UNNode(
            type=NodeType.ELLIPSE,
            id=_next_node_id(),
            name=elem.props.get("id", "ellipse"),
            x=cx - r,
            y=cy - r,
            width=UNSize.fixed(w),
            height=UNSize.fixed(h),
            opacity=opacity,
            source_tool="paper",
        )
        _extract_fills(css, node)
        return node

    # ------------------------------------------------------------------
    # RECTANGLE (SVG <rect>)
    # ------------------------------------------------------------------
    if tag in _RECT_TAGS:
        rx = _px(str(elem.props.get("rx", css.get("borderRadius", "0"))))
        node = UNNode(
            type=NodeType.RECTANGLE,
            id=_next_node_id(),
            name=elem.props.get("id", "rect"),
            x=x,
            y=y,
            width=UNSize.fixed(max(width, 1.0)),
            height=UNSize.fixed(max(height, 1.0)),
            corner_radius=UNCornerRadius.all(rx),
            opacity=opacity,
            source_tool="paper",
        )
        _extract_fills(css, node)
        return node

    # ------------------------------------------------------------------
    # PATH nodes  (SVG <path>, <polyline>, <polygon>, <line>)
    # ------------------------------------------------------------------
    if tag in _PATH_TAGS:
        d = elem.props.get("d", "")
        fill_color_str = elem.props.get("fill", css.get("fill", ""))
        stroke_color_str = elem.props.get("stroke", css.get("stroke", ""))
        stroke_width = float(
            elem.props.get("strokeWidth", elem.props.get("stroke-width", 1.0))
        )
        node = UNNode(
            type=NodeType.PATH,
            id=_next_node_id(),
            name=elem.props.get("id", "path"),
            x=x,
            y=y,
            width=UNSize.fixed(max(width, 1.0)),
            height=UNSize.fixed(max(height, 1.0)),
            geometry=d,
            opacity=opacity,
            source_tool="paper",
        )
        if fill_color_str and fill_color_str not in ("none", ""):
            try:
                c_hex = parse_css_color(fill_color_str)
                if c_hex:
                    c = UNColor.from_hex(c_hex)
                    node.fills.append(UNSolidFill(color=c))
            except Exception:
                pass
        if stroke_color_str and stroke_color_str not in ("none", ""):
            try:
                sc_hex = parse_css_color(stroke_color_str)
                if sc_hex:
                    sc = UNColor.from_hex(sc_hex)
                    node.strokes.append(
                        UNStroke(
                            fill=UNSolidFill(color=sc),
                            thickness=UNStrokeThickness.uniform(stroke_width),
                            align=StrokeAlign.CENTER,
                        )
                    )
            except Exception:
                pass
        return node

    # ------------------------------------------------------------------
    # SVG wrapper → extract geometry or render as FRAME
    # ------------------------------------------------------------------
    if tag in _SVG_TAGS:
        # Try to extract the first <path> child as geometry
        path_child = next(
            (
                c
                for c in elem.children
                if isinstance(c, _Element) and c.tag in _PATH_TAGS
            ),
            None,
        )
        if path_child:
            d = path_child.props.get("d", "")
            if d:
                node = UNNode(
                    type=NodeType.PATH,
                    id=_next_node_id(),
                    name=elem.props.get("id", "svg"),
                    x=x,
                    y=y,
                    width=UNSize.fixed(max(width, 24.0)),
                    height=UNSize.fixed(max(height, 24.0)),
                    geometry=d,
                    opacity=opacity,
                    source_tool="paper",
                )
                fill_str = path_child.props.get("fill", css.get("fill", ""))
                if fill_str and fill_str not in ("none", "currentColor", ""):
                    try:
                        c_hex = parse_css_color(fill_str)
                        if c_hex:
                            c = UNColor.from_hex(c_hex)
                            node.fills.append(UNSolidFill(color=c))
                    except Exception:
                        pass
                return node

        # Fallback: SVG rendered as a FRAME placeholder
        node = UNNode(
            type=NodeType.FRAME,
            id=_next_node_id(),
            name=elem.props.get("id", "svg"),
            x=x,
            y=y,
            width=UNSize.fixed(max(width, 24.0)),
            height=UNSize.fixed(max(height, 24.0)),
            opacity=opacity,
            source_tool="paper",
        )
        _extract_fills(css, node)
        return node

    # ------------------------------------------------------------------
    # FRAME (div, section, …)
    # ------------------------------------------------------------------

    def _make_unsize(value: float, is_fill: bool, is_hug: bool) -> UNSize:
        """Create UNSize with correct sizing mode based on CSS analysis."""
        if is_fill:
            return UNSize.fill()
        if is_hug or value <= 0:
            return UNSize.hug()
        return UNSize.fixed(max(value, 1.0))

    node = UNNode(
        type=NodeType.FRAME,
        id=_next_node_id(),
        name=elem.props.get("id", elem.props.get("data-name", tag)),
        x=x,
        y=y,
        width=_make_unsize(width, width_is_fill, width_is_hug),
        height=_make_unsize(height, height_is_fill, height_is_hug),
        opacity=opacity,
        corner_radius=corner_radius,
        clip_content="hidden" in css.get("overflow", "").lower(),
        source_tool="paper",
    )

    _extract_fills(css, node)
    _extract_strokes(css, node)
    _extract_shadows(css, node)
    _extract_layout(css, node)

    # Position / sizing modes from flexbox context
    align_self = css.get("alignSelf", "").lower()
    flex_val = css.get("flex", "").strip()
    if flex_val in ("1", "1 1 0%", "1 1 auto") or align_self == "stretch":
        node.width = UNSize.fill()

    # Recurse into children
    child_width = width or parent_width
    child_height = height or parent_height
    for child in elem.children:
        if isinstance(child, str):
            text = child.strip()
            if text:
                # Loose text inside a div → create a text node
                text_node = UNNode(
                    type=NodeType.TEXT,
                    id=_next_node_id(),
                    name="text",
                    text_content=text,
                    text_style=_extract_text_style(css, tag),
                    source_tool="paper",
                )
                color_str = css.get("color")
                if color_str:
                    try:
                        c_hex = parse_css_color(color_str)
                        if c_hex:
                            c = UNColor.from_hex(c_hex)
                            text_node.fills.append(UNSolidFill(color=c))
                    except Exception:
                        pass
                node.children.append(text_node)
        elif isinstance(child, _Element):
            child_node = _element_to_node(
                child,
                parent_width=child_width,
                parent_height=child_height,
                depth=depth + 1,
                dimension_map=dimension_map,
            )
            if child_node:
                node.children.append(child_node)

    return node


# ===========================================================================
# PaperReader — BaseReader implementation
# ===========================================================================


class PaperReader(BaseReader):
    """
    Read a Paper Design artboard (or any node) and return a UNNode tree.

    Parameters
    ----------
    host        : Paper Desktop host (default '127.0.0.1')
    port        : Paper Desktop MCP port (default 29979)
    jsx_mode    : 'inline-styles' (default) or 'tailwind'

    Example
    -------
    ::

        with PaperReader() as reader:
            boards = reader.list_nodes()
            tree   = reader.read_node("TO-0")
    """

    tool_name = "paper"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 29979,
        jsx_mode: str = "inline-styles",
        use_sse: bool = False,  # Default to direct mode (more reliable)
    ) -> None:
        self._client = PaperClient(host=host, port=port, use_sse=use_sse)
        self._jsx_mode = jsx_mode

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._client.connect()

    def disconnect(self) -> None:
        self._client.disconnect()

    # ── BaseReader interface ───────────────────────────────────────────────

    def read_node(self, node_id: str) -> UNNode:
        """
        Read the node (artboard or layer) identified by ``node_id`` from
        Paper Design, parse its JSX representation, and return a UNNode tree.

        Parameters
        ----------
        node_id : str
            Paper node ID (e.g. "TO-0" for Wallet Original).

        Returns
        -------
        UNNode — root of the converted node tree.

        Raises
        ------
        NodeNotFoundError   if the node does not exist.
        PaperConnectionError if Paper Desktop is not running.
        """
        global _node_counter, _dimension_map
        _node_counter = 0  # reset for deterministic IDs
        _dimension_map = {}  # reset dimension cache

        try:
            jsx = self._client.get_jsx(node_id, mode=self._jsx_mode)
        except PaperToolError as exc:
            if "not found" in str(exc).lower() or "404" in str(exc):
                raise NodeNotFoundError(node_id, tool="paper") from exc
            raise

        if not jsx or not jsx.strip():
            raise NodeNotFoundError(node_id, tool="paper")

        log.debug("Paper get_jsx returned %d chars for node %s", len(jsx), node_id)

        # Fetch tree summary to get actual computed dimensions for all nodes
        # This is essential because Paper's CSS uses percentage-based sizes
        # that don't translate directly to Figma's absolute positioning
        dim_map = self._fetch_dimension_map(node_id)

        parser = JsxParser(jsx)
        root_elem = parser.parse()

        if root_elem is None:
            raise ValueError(
                f"Could not parse JSX for Paper node '{node_id}'. "
                f"JSX preview: {jsx[:200]}"
            )

        node = _element_to_node(root_elem, dimension_map=dim_map)
        if node is None:
            raise ValueError(
                f"Element-to-node conversion returned None for Paper node '{node_id}'."
            )

        # Stamp source info
        node.source_id = node_id
        node.source_tool = "paper"
        if not node.name or node.name in ("div", "section", "main"):
            node.name = f"Paper [{node_id}]"

        # Fix: Get actual dimensions from Paper API for artboards
        # JSX parsing loses width/height for root artboards
        try:
            info = self._client.get_node_info(node_id)
            if info:
                w = info.get("width")
                h = info.get("height")
                if w and h and (node.width.value == 0 or node.height.value == 0):
                    log.debug(
                        "Fixing root dimensions for %s: %sx%s (was %sx%s)",
                        node_id, w, h, node.width.value, node.height.value
                    )
                    node.width = UNSize.fixed(float(w))
                    node.height = UNSize.fixed(float(h))
        except Exception as exc:
            log.warning("Could not fetch node info for %s: %s", node_id, exc)

        return node

    def _fetch_dimension_map(self, root_id: str, depth: int = 10) -> Dict[str, Tuple[int, int]]:
        """
        Fetch tree summary from Paper and extract a map of node_id → (width, height).

        This gives us the ACTUAL computed dimensions from Paper's layout engine,
        which is essential for accurately converting percentage-based layouts to Figma.
        """
        dim_map: Dict[str, Tuple[int, int]] = {}
        try:
            result = self._client.call_tool(
                "get_tree_summary",
                {"nodeId": root_id, "depth": depth}
            )
            summary = result.get("summary", "") if isinstance(result, dict) else str(result)

            # Parse lines like: Frame "Wallet Original" (TO-0) 390×844
            for line in summary.split("\n"):
                # Match pattern: Type "Name" (ID) W×H
                m = re.search(r'\(([A-Z0-9-]+)\)\s*(\d+)×(\d+)', line)
                if m:
                    node_id = m.group(1)
                    width = int(m.group(2))
                    height = int(m.group(3))
                    dim_map[node_id] = (width, height)
        except Exception as exc:
            log.warning("Could not fetch tree summary for %s: %s", root_id, exc)

        return dim_map

    def list_nodes(self) -> List[Dict[str, Any]]:
        """
        Return a list of top-level artboards in the current Paper file.
        Each item: {"id": str, "name": str, "type": "artboard",
                    "width": int, "height": int}
        """
        try:
            boards = self._client.list_artboards()
            return [
                {
                    "id": b.get("id", ""),
                    "name": b.get("name", ""),
                    "type": "artboard",
                    "width": b.get("width", 0),
                    "height": b.get("height", 0),
                }
                for b in boards
            ]
        except Exception as exc:
            log.warning("PaperReader.list_nodes failed: %s", exc)
            return []

    def get_file_info(self) -> Dict[str, Any]:
        """Return metadata about the currently open Paper file."""
        try:
            return self._client.get_basic_info()
        except Exception:
            return {}

    def screenshot(self, node_id: str, scale: float = 1.0) -> Optional[bytes]:
        """Capture a PNG screenshot of the given node via the Paper MCP."""
        return self._client.screenshot(node_id, scale=scale)

    # ── Convenience ────────────────────────────────────────────────────────

    def read_artboard_by_name(self, name: str) -> UNNode:
        """
        Find an artboard by display name and read it.
        Raises NodeNotFoundError if no artboard with that name exists.
        """
        ab = self._client.get_artboard_by_name(name)
        if not ab:
            raise NodeNotFoundError(name, tool="paper")
        return self.read_node(ab["id"])

    def __repr__(self) -> str:
        return f"<PaperReader client={self._client!r}>"
