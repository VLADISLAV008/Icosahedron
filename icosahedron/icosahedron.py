import cv2

import numpy as np

from PIL import Image

# задаємо розміри зображення
height, width = 1000, 1000
im = np.zeros((height, width, 3), np.uint8)  # Image.new('RGB', (100, 100))

points_list = np.array([[0, 0, 0, 1],
                        [1, 1, 1, 1],
                        [1, 1, 0, 1]])

accumulate_matrix = np.identity(4)

# Start coordinate, here (0, 0)
# represents the top left corner of image
start_point = (0, 0)

# End coordinate, here (250, 250)
# represents the bottom right corner of image
end_point = (250, 250)

# Green color in BGR
color = (0, 255, 0)

# Line thickness of __ px
lineThickness = 2


class Triangle:
    def __init__(self, point1_init, point2_init, point3_init, colour):
        self.point1 = point1_init
        self.point2 = point2_init
        self.point3 = point3_init
        self.colour = colour

    def list_points(self):
        return np.array([self.point1.list_coordinate(),
                         self.point2.list_coordinate(),
                         self.point3.list_coordinate(),
                         [0, 0, 0, 1]])

    def get_colour(self):
        return self.colour


class Point:
    def __init__(self, x_init, y_init, z_init, w_init):
        self.x = x_init
        self.y = y_init
        self.z = z_init
        self.w = w_init

    def list_coordinate(self):
        return [self.x, self.y, self.z, self.w]


side = 10.0
scale = 100.0
points = [Point(0.809, 0.5, 0.588, 1), Point(0.309, -0.5, 0.951, 1),
          Point(-0.309, 0.5, 0.951, 1), Point(-0.809, -0.5, 0.588, 1),
          Point(-1, 0.5, 0, 1), Point(-0.809, -0.5, -0.588, 1),
          Point(-0.309, 0.5, -0.951, 1), Point(0.309, -0.5, -0.951, 1),
          Point(0.809, 0.5, -0.588, 1), Point(1, -0.5, 0, 1),
          Point(0, 1.118, 0, 1), Point(0, -1.118, 0, 1)]

colours = [(0, 255, 255), (0, 250, 0), (204, 22, 35), (0, 1, 250)]

triangles = [Triangle(points[0], points[2], points[1], colours[0]),
             Triangle(points[1], points[9], points[0], colours[2]),
             Triangle(points[3], points[1], points[2], colours[2]),
             Triangle(points[2], points[0], points[10], colours[1]),
             Triangle(points[10], points[6], points[4], colours[2]),
             Triangle(points[0], points[9], points[8], colours[0]),
             Triangle(points[1], points[11], points[9], colours[1]),
             Triangle(points[4], points[2], points[10], colours[3]),
             Triangle(points[3], points[2], points[4], colours[0]),
             Triangle(points[3], points[11], points[1], colours[3]),
             Triangle(points[4], points[5], points[3], colours[2]),
             Triangle(points[5], points[11], points[3], colours[0]),
             Triangle(points[6], points[5], points[4], colours[0]),
             Triangle(points[0], points[8], points[10], colours[2]),
             Triangle(points[7], points[11], points[5], colours[1]),
             Triangle(points[6], points[7], points[5], colours[2]),
             Triangle(points[8], points[7], points[6], colours[0]),
             Triangle(points[10], points[8], points[6], colours[3]),
             Triangle(points[8], points[9], points[7], colours[1]),
             Triangle(points[7], points[9], points[11], colours[2])]


def rotate_ox_matrix(phi):
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(phi), - np.sin(phi), 0],
                     [0, np.sin(phi), np.cos(phi), 0],
                     [0, 0, 0, 1]])


def rotate_oy_matrix(phi):
    return np.array([[np.cos(phi), 0, np.sin(phi), 0],
                     [0, 1, 0, 0],
                     [- np.sin(phi), 0, np.cos(phi), 0],
                     [0, 0, 0, 1]])


def rotate_oz_matrix(phi):
    return np.array([[np.cos(phi), - np.sin(phi), 0, 0],
                     [np.sin(phi), np.cos(phi), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def projection_matrix1(angle, aspect, front, back):
    rad_angle = angle * np.pi / 180.0
    return np.array([[1.0 / np.tan(rad_angle / 2.0) / aspect, 0., 0., 0.],
                     [0., 1.0 / np.tan(rad_angle / 2.0), 0., 0.],
                     [0., 0., (-front - back) / (front - back), (2.0 * back * front) / (front - back)],
                     [0., 0., 1., 0.]])


def projection_matrix2(a, b, c):
    return np.array([[1., 0., 0., -1 / a],
                     [0., 1., 0., -1 / b],
                     [0., 0., 1., -1 / c],
                     [0., 0., 0., 1.]])


projection_matrix = np.array([[1., 0., 0., 1.],
                              [0., 1., 0., 1.],
                              [0., 0., 0., 0.],
                              [0., 0., 0., 1.]])

scale_matrix = np.array([[scale, 0., 0., 0.],
                         [0., scale, 0., 0.],
                         [0., 0., scale, 0.],
                         [0., 0., 0., 1.]])


def rotation_matrix(x, y):
    return np.array([[1., 0., 0., x],
                     [0., 1., 0., y],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])


def coordinate_matrix(triangle: Triangle):
    return np.array(triangle.list_points())


def normal(triangle, matrix):
    coordinate = matrix @ coordinate_matrix(triangle).T
    coordinate = coordinate.T
    v1 = [coordinate[1][0] - coordinate[0][0], coordinate[1][1] - coordinate[0][1], coordinate[1][2] - coordinate[0][2]]
    v2 = [coordinate[2][0] - coordinate[1][0], coordinate[2][1] - coordinate[1][1], coordinate[2][2] - coordinate[1][2]]
    return np.array([v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0], 1])


def view_triangle(triangle, matrix):
    n = normal(triangle, matrix)
    angle = np.arccos(
        n[2] / np.sqrt(np.power(n[0], 2) + np.power(n[1], 2) + np.power(n[2], 2)))
    if angle <= np.pi / 2:
        return True
    else:
        return False


def cos_angle_int(triangle, matrix):
    n = normal(triangle, matrix)
    return n[2] / np.sqrt(np.power(n[0], 2) + np.power(n[1], 2) + np.power(n[2], 2))


def change_colour(triangle, value):
    colour = triangle.get_colour()
    return int(colour[0] * float(value)), int(colour[1] * float(value)), int(colour[2] * float(value))


def draw_triangle(triangle: Triangle):
    projected_points = scale_matrix @ projection_matrix @ accumulate_matrix @ coordinate_matrix(triangle).T
    projected_points = rotation_matrix(400.0, 400.0) @ projected_points
    pts = np.array([projected_points[0][0], projected_points[1][0], projected_points[0][1], projected_points[1][1],
                    projected_points[0][2], projected_points[1][2]], np.int64)
    pts = pts.reshape((- 1, 2))
    cv2.fillPoly(im, [pts], change_colour(triangle, cos_angle_int(triangle, accumulate_matrix)))


while True:
    cv2.waitKey(33)
    if cv2.waitKey(33) == ord('w'):
        accumulate_matrix = rotate_ox_matrix(np.pi / 30) @ accumulate_matrix
    if cv2.waitKey(33) == ord('a'):
        accumulate_matrix = rotate_oz_matrix(np.pi / 30) @ accumulate_matrix
    if cv2.waitKey(33) == ord('d'):
        accumulate_matrix = rotate_oy_matrix(np.pi / 30) @ accumulate_matrix
    for i in triangles:
        if view_triangle(i, accumulate_matrix):
            draw_triangle(i)

    cv2.imshow("Output", im)
    im = np.zeros((height, width, 3), np.uint8)
