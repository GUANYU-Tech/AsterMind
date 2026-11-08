"""Rasterize only the locally authored SVG shapes; does not execute HTML/JS."""

from pathlib import Path
import math
import re
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SCALE = 2
REGULAR = "C:/Windows/Fonts/msyh.ttc"
BOLD = "C:/Windows/Fonts/msyhbd.ttc"
STYLES = {
    "box": {"fill": "#f7f9fd", "stroke": "#cad4e5", "width": 1.5},
    "blue": {"fill": "#eaf0ff", "stroke": "#6584c5"},
    "green": {"fill": "#eaf6f1", "stroke": "#6ba98e"},
    "orange": {"fill": "#fff4e6", "stroke": "#d7a452"},
    "line": {"stroke": "#8494ae", "width": 2, "arrow": True},
    "flow0": {"stroke": "#3467ce", "width": 3},
    "flow1": {"stroke": "#bd7a29", "width": 3},
}


def point(x, y):
    return float(x) * SCALE, float(y) * SCALE


def segments(data):
    tokens = re.findall(r"[MLHV]|-?\d+(?:\.\d+)?", data)
    paths, current, position, i, command = [], [], (0, 0), 0, None
    while i < len(tokens):
        if tokens[i].isalpha():
            command = tokens[i]
            i += 1
        if command in ("M", "L"):
            position = float(tokens[i]), float(tokens[i + 1])
            i += 2
            if command == "M":
                if current: paths.append(current)
                current = []
                command = "L"
        elif command == "H":
            position = float(tokens[i]), position[1]
            i += 1
        elif command == "V":
            position = position[0], float(tokens[i])
            i += 1
        else:
            raise ValueError("unsupported static path command")
        current.append(point(*position))
    if current: paths.append(current)
    return paths


def main():
    source = (ROOT / "docs/architecture.html").read_text(encoding="utf-8")
    for name, markup in zip(("architecture", "devices", "sharing"), re.findall(r"<svg\b.*?</svg>", source, flags=re.S)):
        svg = ET.fromstring(markup)
        _, _, width, height = map(float, svg.attrib["viewBox"].split())
        image = Image.new("RGB", (int(width*SCALE), int(height*SCALE)), "white")
        draw = ImageDraw.Draw(image)
        for element in svg:
            tag, attrs = element.tag, element.attrib
            classes = attrs.get("class", "").split()
            style = {}
            for cls in classes: style.update(STYLES.get(cls, {}))
            style.update({key: attrs[key] for key in ("fill", "stroke") if key in attrs})
            line_width = int(round(float(attrs.get("stroke-width", style.get("width", 1))) * SCALE))
            if tag == "rect":
                x, y = float(attrs["x"]), float(attrs["y"])
                w, h = float(attrs["width"]), float(attrs["height"])
                draw.rounded_rectangle((*point(x,y), *point(x+w,y+h)), radius=float(attrs.get("rx", 0))*SCALE,
                    fill=style.get("fill", "white"), outline=style.get("stroke"), width=line_width)
            elif tag == "text":
                font = ImageFont.truetype(BOLD if "title" in classes else REGULAR, (13 if "small" in classes else 16)*SCALE)
                draw.text(point(attrs["x"], attrs["y"]), element.text or "", font=font,
                          anchor="ms" if attrs.get("text-anchor") == "middle" else "ls",
                          fill="#65738a" if "small" in classes else "#23304a")
            elif tag == "path":
                for path in segments(attrs["d"]):
                    color = style.get("stroke", "#8494ae")
                    if "dashed" in classes:
                        for a,b in zip(path,path[1:]):
                            length = math.dist(a,b)
                            for start in range(0, int(length), 10*SCALE):
                                end = min(start+6*SCALE,length)
                                draw.line([(a[0]+(b[0]-a[0])*t/length,a[1]+(b[1]-a[1])*t/length)
                                           for t in (start,end)], fill=color, width=line_width)
                    else:
                        draw.line(path, fill=color, width=line_width)
                    if style.get("arrow") and len(path)>1:
                        a,b=path[-2:]
                        angle=math.atan2(b[1]-a[1],b[0]-a[0])
                        draw.polygon([b]+[(b[0]-12*math.cos(angle+offset),b[1]-12*math.sin(angle+offset))
                                         for offset in (-.45,.45)],fill=color)
        target = ROOT / f"docs/{name}.png"
        image.save(target)
        print(target.name, image.size)


if __name__ == "__main__":
    main()
