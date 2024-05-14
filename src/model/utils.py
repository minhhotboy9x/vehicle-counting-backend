from supervision.geometry.core import Point
import numpy as np
import cv2

def ccw(a: Point, b: Point):
    return a.x*b.y - a.y*b.x

def transform_points(transformer, points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points

    reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
    transformed_points = cv2.perspectiveTransform(reshaped_points, transformer)
    return transformed_points.reshape(-1, 2)

def calculate_distance(point1: list, point2: list) -> float:
    return ((point1[0]-point2[0]) ** 2 + (point1[1]-point2[1]) ** 2) ** 0.5
