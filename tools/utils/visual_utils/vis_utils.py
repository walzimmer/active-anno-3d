# import mayavi.mlab as mlab
import numpy as np
import torch
import open3d as o3d

from scipy.spatial.transform import Rotation as R

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

# TUMTraf Dataset
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200


# TUMTraf dataset IDs
id_to_class_name_mapping = {
    "0": {
        "class_label_de": "PKW",
        "class_label_en": "Car",
        "color_hex": "#00ccf6",
        "color_rgb": (0, 204, 246),
        "color_bgr": (246, 204, 0),
        "color_bgr_normalized": (0.964705882, 0.8, 0),
        "color_rgb_normalized": (0, 0.8, 0.96),
    },
    "1": {
        "class_label_de": "LKW",
        "class_label_en": "Truck",
        "color_hex": "#3FE9B9",
        "color_rgb": (63, 233, 185),
        "color_bgr": (185, 233, 63),
        "color_bgr_normalized": (0.71372549, 1, 0.337254902),
        "color_rgb_normalized": (0.25, 0.91, 0.72),
    },
    "2": {
        "class_label_de": "Anhänger",
        "class_label_en": "Trailer",
        "color_hex": "#5AFF7E",
        "color_rgb": (90, 255, 126),
        "color_bgr": (126, 255, 90),
        "color_bgr_normalized": (0.494117647, 1, 0.352941176),
        "color_rgb_normalized": (0.35, 1, 0.49),
    },
    "3": {
        "class_label_de": "Van",
        "class_label_en": "Van",
        "color_hex": "#EBCF36",
        "color_rgb": (235, 207, 54),
        "color_bgr": (54, 207, 235),
        "color_bgr_normalized": (0.211764706, 0.811764706, 0.921568627),
        "color_rgb_normalized": (0.92, 0.81, 0.21),
    },
    "4": {
        "class_label_de": "Motorrad",
        "class_label_en": "Motorcycle",
        "color_hex": "#B9A454",
        "color_rgb": (185, 164, 84),
        "color_bgr": (84, 164, 185),
        "color_bgr_normalized": (0.329411765, 0.643137255, 0.725490196),
        "color_rgb_normalized": (0.72, 0.64, 0.33),
    },
    "5": {
        "class_label_de": "Bus",
        "class_label_en": "Bus",
        "color_hex": "#D98A86",
        "color_rgb": (217, 138, 134),
        "color_bgr": (134, 138, 217),
        "color_bgr_normalized": (0.525490196, 0.541176471, 0.850980392),
        "color_rgb_normalized": (0.85, 0.54, 0.52),
    },
    "6": {
        "class_label_de": "Person",
        "class_label_en": "Pedestrian",
        "color_hex": "#E976F9",
        "color_rgb": (233, 118, 249),
        "color_bgr": (249, 118, 233),
        "color_bgr_normalized": (0.976470588, 0.462745098, 0.91372549),
        "color_rgb_normalized": (0.91, 0.46, 0.97),
    },
    "7": {
        "class_label_de": "Fahrrad",
        "class_label_en": "Bicycle",
        "color_hex": "#B18CFF",
        "color_rgb": (177, 140, 255),
        "color_bgr": (255, 140, 177),
        "color_bgr_normalized": (1, 0.549019608, 0.694117647),
        "color_rgb_normalized": (0.69, 0.55, 1),
    },
    "8": {
        "class_label_de": "Einsatzfahrzeug",
        "class_label_en": "Emergency_Vehicle",
        "color_hex": "#666bfa",
        "color_rgb": (102, 107, 250),
        "color_bgr": (250, 107, 102),
        "color_bgr_normalized": (0.980392157, 0.419607843, 0.4),
        "color_rgb_normalized": (0.4, 0.42, 0.98),
    },
    "9": {
        "class_label_de": "Unbekannt",
        "class_label_en": "Other",
        "color_hex": "#C7C7C7",
        "color_rgb": (199, 199, 199),
        "color_bgr": (199, 199, 199),
        "color_bgr_normalized": (0.780392157, 0.780392157, 0.780392157),
        "color_rgb_normalized": (0.78, 0.78, 0.78),
    },
    "10": {
        "class_label_de": "Nummernschild",
        "class_label_en": "License_Plate",
        "color_hex": "#000000",
        "color_rgb": (0, 0, 0),
        "color_bgr": (0, 0, 0),
        "color_bgr_normalized": (0, 0, 0),
        "color_rgb_normalized": (0, 0, 0),
    },
}


# Mapping object class name to TUMTraf dataset class ID
class_name_to_id_mapping = {
    "CAR": 0,
    "TRUCK": 1,
    "TRAILER": 2,
    "VAN": 3,
    "MOTORCYCLE": 4,
    "BUS": 5,
    "PEDESTRIAN": 6,
    "BICYCLE": 7,
    "EMERGENCY_VEHICLE": 8,
    "OTHER": 9,
    "LICENSE_PLATE_LOCATION": 10,
}


# MS COCO class name to MS COCO ID mapping
mscoco_class_name_to_id_mapping = {
    "PEDESTRIAN": 0,
    "BICYCLE": 1,
    "CAR": 2,
    "MOTORCYCLE": 3,
    "BUS": 5,
    "TRUCK": 7,
    "TRAILER": 7,
    "EMERGENCY_VEHICLE": 7,
    "OTHER": 2,
    "VAN": 7,
}


def get_corners(cuboid):
    """
    cuboid: list or array [xPos, yPos, zPos, quaternoins[x, y, z, w], l, w, h]
    """

    l = cuboid[7]
    w = cuboid[8]
    h = cuboid[9]

    bounding_box = np.array(
        [
            [-l/2, l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2],
            [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2],
            [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]
        ]
    )

    translation = cuboid[:3]
    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    rotation_quaternion = cuboid[3:7]
    rotation_matrix = R.from_quat(rotation_quaternion).as_matrix()
    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(rotation_matrix, bounding_box) + eight_points.transpose()

    return corner_box.transpose()


def check_corners_within_image(corners):
    valid_corners = 0
    for idx in range(len(corners)):
        corner = corners[idx, :]
        if corner[0] >= 0 and corner[0] < IMAGE_WIDTH and corner[1] >= 0 and corner[1] < IMAGE_HEIGHT:
            valid_corners += 1
    if valid_corners > 1:
        return True
    return False


def get_2d_corner_points(cx, cy, length, width, yaw):
    """
    Find the coordinates of the rectangle with given center, length, width and angle of the longer side
    """

    mp1 = [cx + length / 2 * np.cos(yaw), cy + length / 2 * np.sin(yaw)]
    mp3 = [cx - length / 2 * np.cos(yaw), cy - length / 2 * np.sin(yaw)]

    p1 = [mp1[0] - width / 2 * np.sin(yaw), mp1[1] + width / 2 * np.cos(yaw)]
    p2 = [mp3[0] - width / 2 * np.sin(yaw), mp3[1] + width / 2 * np.cos(yaw)]
    p3 = [mp3[0] + width / 2 * np.sin(yaw), mp3[1] - width / 2 * np.cos(yaw)]
    p4 = [mp1[0] + width / 2 * np.sin(yaw), mp1[1] - width / 2 * np.cos(yaw)]

    px = [p1[0], p2[0], p3[0], p4[0]]
    py = [p1[1], p2[1], p3[1], p4[1]]

    px.append(px[0])
    py.append(py[0])

    return px, py


def filter_point_cloud(pcd):
    
    points = np.array(pcd.points)
    points_filtered = points[~np.all(points == 0, axis=1)]

    # remove points with distance>120
    distances = np.array([np.sqrt(row[0] * row[0] + row[1] * row[1] + row[2] * row[2]) for row in points_filtered])
    points_filtered = points_filtered[distances < 120.0]
    distances = distances[distances < 120.0]
    # remove points with distance<3
    points_filtered = points_filtered[distances > 3.0]

    corner_point_min = np.array([-150, -150, -10])
    corner_point_max = np.array([150, 150, 5])
    points = np.vstack((points_filtered, corner_point_min, corner_point_max))

    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points[:, :3]))

    return pcd




def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(600, 600), draw_origin=True):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig


def draw_sphere_pts(pts, color=(0, 1, 0), fig=None, bgcolor=(0, 0, 0), scale_factor=0.2):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()

    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(600, 600))

    if isinstance(color, np.ndarray) and color.shape[0] == 1:
        color = color[0]
        color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    if isinstance(color, np.ndarray):
        pts_color = np.zeros((pts.__len__(), 4), dtype=np.uint8)
        pts_color[:, 0:3] = color
        pts_color[:, 3] = 255
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], np.arange(0, pts_color.__len__()), mode='sphere',
                          scale_factor=scale_factor, figure=fig)
        G.glyph.color_mode = 'color_by_scalar'
        G.glyph.scale_mode = 'scale_by_vector'
        G.module_manager.scalar_lut_manager.lut.table = pts_color
    else:
        mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='sphere', color=color,
                      colormap='gnuplot', scale_factor=scale_factor, figure=fig)

    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
    mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), line_width=3, tube_radius=None, figure=fig)

    return fig


def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig


def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)

    return fig


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None):
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()

    fig = visualize_pts(points)
    fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))
    if gt_boxes is not None:
        corners3d = boxes_to_corners_3d(gt_boxes)
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

    if ref_boxes is not None and len(ref_boxes) > 0:
        ref_corners3d = boxes_to_corners_3d(ref_boxes)
        if ref_labels is None:
            fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores, max_num=100)
        else:
            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(box_colormap[k % len(box_colormap)])
                mask = (ref_labels == k)
                fig = draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=ref_scores[mask], max_num=100)
    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    return fig


def draw_corners3d(corners3d, fig, color=(1, 1, 1), line_width=2, cls=None, tag='', max_num=500, tube_radius=None):
    """
    :param corners3d: (N, 8, 3)
    :param fig:
    :param color:
    :param line_width:
    :param cls:
    :param tag:
    :param max_num:
    :return:
    """
    import mayavi.mlab as mlab
    num = min(max_num, len(corners3d))
    for n in range(num):
        b = corners3d[n]  # (8, 3)

        if cls is not None:
            if isinstance(cls, np.ndarray):
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%.2f' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)
            else:
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%s' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

        i, j = 0, 5
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)
        i, j = 1, 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)

    return fig