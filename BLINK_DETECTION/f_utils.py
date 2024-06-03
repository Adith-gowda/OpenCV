import numpy as np

def get_areas(rects):
    areas = []
    for (x1, y1, x2, y2) in rects:
        areas.append((x2 - x1) * (y2 - y1))
    return areas
