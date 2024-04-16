from supervision.geometry.core import Point

def ccw(a: Point, b: Point):
    return a.x*b.y - a.y*b.x