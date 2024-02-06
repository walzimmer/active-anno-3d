import os
import sys
import json
import glob
import numpy as np
import matplotlib as mpl
import open3d as o3d
import cv2
import seaborn as sns
from tqdm import tqdm

from tools.utils.log_utils import parse_log_file
from matplotlib import cm, pyplot as plt
from math import radians
from scipy.spatial.transform import Rotation as R
from typing import List

from tools.utils.visual_utils.vis_utils import get_corners, id_to_class_name_mapping, class_name_to_id_mapping, filter_point_cloud
from tools.utils.det_utils.detections import Detection, detections_to_dict, detections_to_openlabel

os.environ['MPLBACKEND'] = 'Agg'

class VisualizationTUMTraf():
    def __init__(self):
        self.class_to_color = {
        'CAR': (0, 0, 255),       # Red
        'PEDESTRIAN': (255, 0, 0), # Blue
        'VAN': (0, 255, 0),       # Green
        'BICYCLE': (0, 255, 255),  # Yellow
        'TRUCK': (0, 165, 255),    # Orange
        'TRAILER': (255, 0, 255),  # Purple
        'BUS': (180, 105, 255),    # Pink
        'MOTORCYCLE': (19, 69, 139) # Brown
        }

    
    def get_projection_matrix(self, camera_id, lidar_id,  release='R02'):
        projecttion_matrix = None

        if camera_id == 's110_camera_basler_south1_8mm' and lidar_id == 's110_lidar_ouster_south':
            if release == 'R02':    # TUMTraf-I (Intersection dataset)
                projection_matrix = np.array(
                    [
                            [7.04216073e02, -1.37317442e03, -4.32235765e02, -2.03369364e04],
                            [-9.28351327e01, -1.77543929e01, -1.45629177e03, 9.80290034e02],
                            [8.71736000e-01, -9.03453000e-02, -4.81574000e-01, -2.58546000e00],
                        ],
                        dtype=float,
                )
            elif release == "R03": # TUMTraf-C (cooperative dataset)
                intrinsic_camera_matrix = np.array([[-1301.42, 0, 940.389], [0, -1299.94, 674.417], [0, 0, 1]])
                extrinsic_matrix = np.array(
                    [
                        [-0.41205, 0.910783, -0.0262516, 15.0787],
                        [0.453777, 0.230108, 0.860893, 2.52926],
                        [0.790127, 0.342818, -0.508109, 3.67868],
                    ],
                )
                projection_matrix = np.matmul(intrinsic_camera_matrix, extrinsic_matrix)
        
        elif camera_id == "s110_camera_basler_south2_8mm" and lidar_id == "s110_lidar_ouster_south":
            # optimized intrinsics (calibration lidar to base)
            intrinsic_camera_matrix = np.array(
                [[1315.56, 0, 969.353, 0.0], [0, 1368.35, 579.071, 0.0], [0, 0, 1, 0.0]], dtype=float
            )
            # manual calibration, optimizing intrinsics and extrinsics
            extrinsic_matrix_lidar_to_base = np.array(
                [
                    [0.247006, -0.955779, -0.15961, -16.8017],
                    [0.912112, 0.173713, 0.371316, 4.66979],
                    [-0.327169, -0.237299, 0.914685, 6.4602],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            # extrinsic base to south2 camera
            extrinsic_matrix_base_to_camera = np.array(
                [
                    [0.8924758822566284, 0.45096261644035174, -0.01093243630327495, 14.921784677055939],
                    [0.29913535165414396, -0.6097951995429897, -0.7339399539506467, 13.668310799382738],
                    [-0.3376460291207414, 0.6517534297474759, -0.679126369559744, -5.630430017833277],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            extrinsic_matrix_lidar_to_camera = np.matmul(
                extrinsic_matrix_base_to_camera, extrinsic_matrix_lidar_to_base
            )
            projection_matrix = np.matmul(intrinsic_camera_matrix, extrinsic_matrix_lidar_to_camera)
        
        elif camera_id == "s110_camera_basler_south1_8mm" and lidar_id == "s110_lidar_ouster_north":
            # optimized intrinsic (lidar to base calibration)
            intrinsic_camera_matrix = np.array(
                [[1305.59, 0, 933.819, 0], [0, 1320.61, 609.602, 0], [0, 0, 1, 0]], dtype=float
            )

            # extrinsic lidar north to base
            extrinsic_matrix_lidar_to_base = np.array(
                [
                    [-0.064419, -0.997922, 0.00169282, -2.08748],
                    [0.997875, -0.0644324, -0.00969147, 0.226579],
                    [0.0097804, 0.0010649, 0.999952, 8.29723],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            # extrinsic base to south1 camera
            extrinsic_matrix_base_to_camera = np.array(
                [
                    [0.9530205584452789, -0.3026130702071279, 0.013309580025851253, 1.7732651490941862],
                    [-0.1291778833442192, -0.4457786636335154, -0.8857733668968741, 7.609039571774588],
                    [0.27397972486181504, 0.842440925400074, -0.4639271468406554, 4.047780978836272],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            extrinsic_matrix_lidar_to_camera = np.matmul(
                extrinsic_matrix_base_to_camera, extrinsic_matrix_lidar_to_base
            )
            projection_matrix = np.matmul(intrinsic_camera_matrix, extrinsic_matrix_lidar_to_camera)

        elif camera_id == "s110_camera_basler_south2_8mm" and lidar_id == "s110_lidar_ouster_north":
            # projection matrix from s110_lidar_ouster_north to s110_camera_basler_south2
            intrinsic_camera_matrix = np.array(
                [[1282.35, 0.0, 957.578, 0.0], [0.0, 1334.48, 563.157, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=float
            )
            extrinsic_matrix = np.array(
                [
                    [0.37383, -0.927155, 0.0251845, 14.2181],
                    [-0.302544, -0.147564, -0.941643, 3.50648],
                    [0.876766, 0.344395, -0.335669, -7.26891],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            projection_matrix = np.matmul(intrinsic_camera_matrix, extrinsic_matrix)

        elif camera_id == "s110_camera_basler_east_8mm" and lidar_id == "s110_lidar_ouster_south":
            projection_matrix = np.array(
                [
                    [-2666.70160799, -655.44528859, -790.96345758, -33010.77350141],
                    [430.89231274, 66.06703744, -2053.70223986, 6630.65222157],
                    [-0.00932524, -0.96164431, -0.27414094, 11.41820108],
                ]
            )

        elif camera_id == "s110_camera_basler_north_8mm" and lidar_id == "s110_lidar_ouster_south":
            intrinsic_matrix = np.array([[1360.68, 0, 849.369], [0, 1470.71, 632.174], [0, 0, 1]])
            extrinsic_matrix = np.array(
                [
                    [-0.564602, -0.824833, -0.0295815, -12.9358],
                    [-0.458346, 0.343143, -0.819861, 7.22666],
                    [0.686399, -0.449337, -0.571798, -6.75018],
                ],
            )
            projection_matrix = np.matmul(intrinsic_matrix, extrinsic_matrix)

        elif camera_id == "vehicle_camera_basler_16mm" and lidar_id == "vehicle_lidar_robosense":
            projection_matrix = np.array(
                [[1019.929965441548, -2613.286262078907, 184.6794570200418, 370.7180273597151],
                 [589.8963703919744, -24.09642935106967, -2623.908527352794, -139.3143336725661],
                 [0.9841844439506531, 0.1303769648075104, 0.1199281811714172, -0.1664766669273376]])

        else:
            print("Error. Unknown camera passed: ", camera_id, ". Exiting...")
            sys.exit()
    
        return projection_matrix


    @staticmethod
    def get_attribute_by_name(attribute_list, attribute_name):
        for attribute in attribute_list:
            if attribute["name"] == attribute_name:
                return attribute
        return None


    @staticmethod
    def draw_line(img, start_point, end_point, color):
        cv2.line(img, start_point, end_point, color, 2)

    @staticmethod    
    def add_open3d_axis(vis):
        """Add a small 3D axis on Open3D Visualizer"""
        axis = o3d.geometry.LineSet()
        axis.points = o3d.utility.Vector3dVector(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )
        axis.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [0, 2], [0, 3]]))
        axis.colors = o3d.utility.Vector3dVector(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )
        vis.add_geometry(axis)


    @staticmethod
    def color_point_cloud(pcd):

        sorted_z = np.asarray(pcd.points)[np.argsort(np.asarray(pcd.points)[:, 2])[::-1]]
        rows = len(pcd.points)
        pcd.normalize_normals()

        # when Z values are negative, this if else statement switches the min and max
        if sorted_z[0][2] < sorted_z[rows - 1][2]:
            min_z_val = sorted_z[0][2]
            max_z_val = sorted_z[rows - 1][2]
        else:
            max_z_val = sorted_z[0][2]
            min_z_val = sorted_z[rows - 1][2]

        # assign colors to the point cloud file
        cmap_norm = mpl.colors.Normalize(vmin=min_z_val, vmax=max_z_val)
        # example color maps: jet, hsv.  Further colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
        point_colors = plt.get_cmap("jet")(cmap_norm(np.asarray(pcd.points)[:, -1]))[:, 0:3]
        pcd.colors = o3d.utility.Vector3dVector(point_colors)

        return pcd


    def process_data(self, box_data, input_type, use_detections_in_base):

        objects = []

        # Dataset in ASAM OpenLABEL format
        if "openlabel" in box_data:
            for frame_id, frame_obj in box_data["openlabel"]["frames"].items():
                for object_id, label in frame_obj["objects"].items():
                
                    l = float(label["object_data"]["cuboid"]["val"][7])
                    w = float(label["object_data"]["cuboid"]["val"][8])
                    h = float(label["object_data"]["cuboid"]["val"][9])

                    quat_x = float(label["object_data"]["cuboid"]["val"][3])
                    quat_y = float(label["object_data"]["cuboid"]["val"][4])
                    quat_z = float(label["object_data"]["cuboid"]["val"][5])
                    quat_w = float(label["object_data"]["cuboid"]["val"][6])
                    rotation_yaw = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler("xyz", degrees=False)[2]

                    position_3d = [
                        float(label["object_data"]["cuboid"]["val"][0]),
                        float(label["object_data"]["cuboid"]["val"][1]),
                        float(label["object_data"]["cuboid"]["val"][2]),
                    ]

                    category = label["object_data"]["type"].upper()

                    # transform detections from s110_base to lidar
                    if use_detections_in_base and input_type == "detections":
                        rotation_yaw = rotation_yaw + np.deg2rad(103)
                        position_3d = transform_base_to_lidar(np.array(position_3d).reshape(1, 3)).T.flatten()

                    obb = o3d.geometry.OrientedBoundingBox(
                        position_3d,
                        np.array(
                            [
                                [np.cos(rotation_yaw), -np.sin(rotation_yaw), 0],
                                [np.sin(rotation_yaw), np.cos(rotation_yaw), 0],
                                [0, 0, 1],
                            ]
                        ),
                        np.array([l, w, h]),
                    )

                    object_label = [l, w, h, rotation_yaw, position_3d, category]
                    objects.append((object_id, object_label))

        else:
            print("Dataset is not in ASAM OpenLABEL format!")

        return objects

    
    def visualize_bounding_box(self, box_label, use_two_colors, input_type, vis):

        quats = R.from_euler("xyz", [0, 0, box_label[3]], degrees=False).as_quat()
        corners = get_corners(
            [
                box_label[4][0],
                box_label[4][1],
                box_label[4][2],
                quats[0],
                quats[1],
                quats[2],
                quats[3],
                box_label[0],
                box_label[1],
                box_label[2],
            ]
        )
        line_indices = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

        if use_two_colors and input_type == "labels":
            color_green_rgb = (0, 255, 0)
            color_green_bgr_normalized = (color_green_rgb[0] / 255, color_green_rgb[1] / 255, color_green_rgb[2] / 255)
            colors = [color_green_bgr_normalized for _ in range(len(line_indices))]
        
        elif use_two_colors and input_type == "detections":
            color_red_rgb = (255, 0, 0)
            color_red_bgr_normalized = (color_red_rgb[0] / 255, color_red_rgb[1] / 255, color_red_rgb[2] / 255)
            colors = [color_red_bgr_normalized for _ in range(len(line_indices))]
        
        else:
            colors = [
                id_to_class_name_mapping[str(class_name_to_id_mapping[box_label[-1]])]["color_bgr_normalized"]
                for _ in range(len(line_indices))
            ]

        line_set = o3d.geometry.LineSet()

        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        vis.add_geometry(line_set)

        return line_set


    def visualize_pcd_with_boxes(self,
                                pcd_file_path,
                                labels_file_path,
                                dets_file_path,
                                use_detections_in_base,
                                view='bev',
                                save_vis="",
                                show_vis=True,
                                return_vis=False):
        
        
        pcd = o3d.io.read_point_cloud(pcd_file_path)
        
        pcd = filter_point_cloud(pcd)

        pcd = self.color_point_cloud(pcd)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Point Cloud Visualizer", width=1024, height=720)
        vis.get_render_option().background_color = [0.1, 0.1, 0.1]
        vis.get_render_option().point_size = 3.0

        self.add_open3d_axis(vis)

        vis.add_geometry(pcd)

        use_two_colors = labels_file_path != "" and dets_file_path != ""

        if labels_file_path != "":
            with open(labels_file_path, 'r') as labels_file:
                label_data = json.load(labels_file)
            gt_id_label = self.process_data(
                box_data=label_data,
                input_type="labels",
                use_detections_in_base=use_detections_in_base
            )
            for id, box_label in gt_id_label:
                self.visualize_bounding_box(
                    box_label=box_label,
                    use_two_colors=use_two_colors,
                    input_type="labels",
                    vis=vis
                )
        
        if dets_file_path != "":
            with open(dets_file_path, 'r') as dets_file:
                detection_data = json.load(dets_file)
            detection_id_label = self.process_data(
                box_data=detection_data,
                input_type="detections",
                use_detections_in_base=use_detections_in_base
            )
            for id, box_label in detection_id_label:
                self.visualize_bounding_box(
                    box_label=box_label,
                    use_two_colors=use_two_colors,
                    input_type="detections",
                    vis=vis
                )

        if view == 'bev':
            vis.get_view_control().set_zoom(0.17)
            vis.get_view_control().set_front([0.22, 0.141, 0.965])
            vis.get_view_control().set_lookat([22.964, -2.772, -7.230])
            vis.get_view_control().set_up([0.969, -0.148, -0.200])
        
        elif view == 'custom':
            vis.get_view_control().set_zoom(0.08)
            vis.get_view_control().set_front([-0.20, 0, 0])
            vis.get_view_control().set_lookat([27, 1, 8])
            vis.get_view_control().set_up([0, 0, 1])
            

        if show_vis:
            vis.run()
    
        if save_vis != "":
            vis.capture_screen_image(filename=save_vis,
                                     do_render=True)
        
        if return_vis:
            # Capture screen to a numpy array
            # rotate the view 45 degrees around the y-axis
            vis.get_view_control().rotate(0, 45)
            img_buffer = vis.capture_screen_float_buffer(do_render=True)
            img_array = (np.asarray(img_buffer) * 255).astype(np.uint8)
            return img_array



    def visualize_pcd_without_boxes(self,
                                    pcd_file_path,
                                    view='bev',
                                    show_vis=True,
                                    return_vis=False,
                                    save_vis=""):
        
        pcd = o3d.io.read_point_cloud(pcd_file_path)
        
        pcd = filter_point_cloud(pcd)

        pcd = self.color_point_cloud(pcd)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Point Cloud Visualizer", width=1024, height=720)
        vis.get_render_option().background_color = [0.1, 0.1, 0.1]
        vis.get_render_option().point_size = 3.0

        self.add_open3d_axis(vis)

        vis.add_geometry(pcd)

        if view == 'bev':
            vis.get_view_control().set_zoom(0.17)
            vis.get_view_control().set_front([0.22, 0.141, 0.965])
            vis.get_view_control().set_lookat([22.964, -2.772, -7.230])
            vis.get_view_control().set_up([0.969, -0.148, -0.200])
        
        elif view == 'custom':
            vis.get_view_control().set_zoom(0.08)
            vis.get_view_control().set_front([-0.20, 0, 0])
            vis.get_view_control().set_lookat([27, 1, 8])
            vis.get_view_control().set_up([0, 0, 1])

        if show_vis:
            vis.run()

        if save_vis != "":
            vis.capture_screen_image(filename=save_vis,
                                     do_render=True)

        if return_vis:
            img_buffer = vis.capture_screen_float_buffer(do_render=True)
            img_array = (np.asarray(img_buffer) * 255).astype(np.uint8)
            return img_array
    


    def project_pcd_on_image(self,
                             pcd_file_path,
                             img_file_path,
                             labels_file_path,
                             dets_file_path,
                             lidar_id,
                             camera_id,
                             use_two_colors,
                             save_vis_results_path,
                             show_vis=True,
                             return_img=False):
        
        img = cv2.imread(img_file_path, cv2.IMREAD_UNCHANGED)

        pcd = o3d.io.read_point_cloud(pcd_file_path)
        pcd = filter_point_cloud(pcd)

        
        if labels_file_path != "":
            with open(labels_file_path, 'r') as f:
                gt_data = json.load(f)
            gt_id_label = self.process_data(
                box_data=gt_data,
                input_type="labels",
                use_detections_in_base=False
            )
            for id, box_label in gt_id_label:
                quats = R.from_euler("xyz", [0, 0, box_label[3]], degrees=False).as_quat()
                points3d = get_corners([
                    box_label[4][0],
                    box_label[4][1],
                    box_label[4][2],
                    quats[0],
                    quats[1],
                    quats[2],
                    quats[3],
                    box_label[0],
                    box_label[1],
                    box_label[2],
                ])
                points2d = self.project_3d_box_to_2d(
                    points_3d=points3d,
                    camera_id=camera_id,
                    lidar_id=lidar_id
                )
                points2d = [(int(i[0]), int(i[1])) for i in points2d]
                color_labels = (0, 255, 0)  # BGR Format
                self.draw_2d_box(img=img, points_2d=points2d, color=color_labels)
            
        if dets_file_path != "":
            with open(dets_file_path, 'r') as f:
                det_data = json.load(f)
            gt_id_label = self.process_data(
                box_data=det_data,
                input_type="detections",
                use_detections_in_base=False
            )
            for id, box_label in gt_id_label:
                quats = R.from_euler("xyz", [0, 0, box_label[3]], degrees=False).as_quat()
                points3d = get_corners([
                    box_label[4][0],
                    box_label[4][1],
                    box_label[4][2],
                    quats[0],
                    quats[1],
                    quats[2],
                    quats[3],
                    box_label[0],
                    box_label[1],
                    box_label[2],
            ])
                points2d = self.project_3d_box_to_2d(
                    points_3d=points3d,
                    camera_id=camera_id,
                    lidar_id=lidar_id
                )
                points2d = [(int(i[0]), int(i[1])) for i in points2d]
                # color_detections = (0, 0, 255)
                self.draw_2d_box(img=img, points_2d=points2d, color=self.class_to_color[box_label[-1]]) 


        points = np.array(pcd.points)
        points = np.hstack((points, np.ones((points.shape[0], 1))))

        distances = []
        indices_to_keep = []
        for i in range(len(points[:, 0])):
            point = points[i, :]
            distance = np.sqrt((point[0] ** 2) + (point[1] ** 2) + (point[2] ** 2))
            if distance > 2:
                distances.append(distance)
                indices_to_keep.append(i)

        points = points[indices_to_keep, :]

        projection_matrix = self.get_projection_matrix(camera_id=camera_id,
                                                       lidar_id=lidar_id)
        
        points = np.matmul(points[:, :4], projection_matrix.T)

        distances_numpy = np.asarray(distances)
        max_distance = max(distances_numpy)

        m = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=70, vmax=250), cmap=cm.jet)
        
        num_points_within_image = 0
        for i in range(len(points)):

            z_coord = points[i, 2]
            if z_coord > 0:
                pos_x, pos_y = int(points[i, 0] / z_coord), int(points[i, 1] / z_coord)

                if  0 <= pos_x < 1920 and 0 <= pos_y < 1200:
                    num_points_within_image += 1

                    distance_idx = 255 - (int(distances_numpy[i] / max_distance * 255))

                    color_rgba = m.to_rgba(distance_idx)
                    color_rgb = tuple(int(c * 255) for c in color_rgba[:3])

                    cv2.circle(img, (pos_x, pos_y), 2, color_rgb, thickness=-1)


        legend_start_x = 50 
        legend_start_y = img.shape[0] - 30 
        for class_name, color in self.class_to_color.items():
            cv2.circle(img, (legend_start_x, legend_start_y), 15, color, -1)
            cv2.putText(img, class_name, (legend_start_x + 30, legend_start_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            legend_start_x += 200 


        if show_vis:
            cv2.imshow("image", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_vis_results_path:
            cv2.imwrite(save_vis_results_path, img)
        
        if return_img:
            return img



    def project_3d_box_to_2d(self, points_3d, camera_id, lidar_id):

        points_3d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

        # project points to 2D
        projection_matrix = self.get_projection_matrix(camera_id, lidar_id)
        points = np.matmul(projection_matrix, points_3d.T)
        # filter out points behind camera
        points = points[:, points[2] > 0]
        # Divide x and y values by z (camera pinhole model).
        image_points = points[:2] / points[2]

        return image_points.T



    def draw_2d_box(self, img, points_2d, color):
        if len(points_2d) == 8:
                self.draw_line(img, points_2d[0], points_2d[1], color)
                self.draw_line(img, points_2d[1], points_2d[2], color)
                self.draw_line(img, points_2d[2], points_2d[3], color)
                self.draw_line(img, points_2d[3], points_2d[0], color)
                self.draw_line(img, points_2d[4], points_2d[5], color)
                self.draw_line(img, points_2d[5], points_2d[6], color)
                self.draw_line(img, points_2d[6], points_2d[7], color)
                self.draw_line(img, points_2d[7], points_2d[4], color)
                self.draw_line(img, points_2d[0], points_2d[4], color)
                self.draw_line(img, points_2d[1], points_2d[5], color)
                self.draw_line(img, points_2d[2], points_2d[6], color)
                self.draw_line(img, points_2d[3], points_2d[7], color)


    def create_video(self,
                     pcd_dir,
                     img1_dir,
                     img2_dir,
                     labels_dir,
                     dets_dir,
                     log_path,
                     video_name, 
                     fps, 
                     output_path):
        
        # Initializa video
        video = cv2.VideoWriter(os.path.join(output_path, video_name + '.avi'), 
                           cv2.VideoWriter_fourcc(*'XVID'), fps, (1200, 840))
        
        # load files
        pcd_files = sorted(glob.glob(os.path.join(pcd_dir, '*.pcd')))
        label_files = sorted(glob.glob(os.path.join(labels_dir, '*.json')))
        det_files = sorted(glob.glob(os.path.join(dets_dir, '*.json')))
        img1_files = sorted(glob.glob(os.path.join(img1_dir, '*.png')))
        img2_files = sorted(glob.glob(os.path.join(img2_dir, '*.png')))

        # create the evaluation monitor
        # _, eval_dict = parse_log_file(log_path, get_evals=True)

        # eval_fig = plot_mAP(eval_dict=eval_dict,
        #                     save_output=output_path,
        #                     show_fig=False,
        #                     return_fig=True)
        
        # eval_fig = cv2.resize(eval_fig, (700, 420))

        for pcd, labels, dets, img1, img2 in tqdm(zip(pcd_files, label_files, det_files, img1_files, img2_files)):

            frame_pcd = self.visualize_pcd_with_boxes(pcd_file_path=pcd,
                                            labels_file_path=labels,
                                            dets_file_path=dets,
                                            use_detections_in_base=False,
                                            view='bev',
                                            save_vis="",
                                            return_vis=True,
                                            show_vis=False)
            
            frame_pcd = cv2.cvtColor(frame_pcd, cv2.COLOR_RGB2BGR)

            frame_img1 = self.project_pcd_on_image(pcd_file_path=pcd,
                                        img_file_path=img1,
                                        labels_file_path=labels,
                                        dets_file_path=dets,
                                        camera_id='s110_camera_basler_south1_8mm',
                                        lidar_id='s110_lidar_ouster_south',
                                        use_two_colors=True,
                                        show_vis=False,
                                        save_vis_results_path="",
                                        return_img=True)

            frame_img2 = self.project_pcd_on_image(pcd_file_path=pcd,
                                        img_file_path=img2,
                                        labels_file_path=labels,
                                        dets_file_path=dets,
                                        camera_id='s110_camera_basler_south2_8mm',
                                        lidar_id='s110_lidar_ouster_south',
                                        use_two_colors=True,
                                        show_vis=False,
                                        save_vis_results_path="",
                                        return_img=True)
            
            
            frame_img1 = cv2.resize(frame_img1, (500, 420))
            frame_img2 = cv2.resize(frame_img2, (500, 420))

            first_col = np.vstack((frame_img1, frame_img2))
            frame_pcd = cv2.resize(frame_pcd, (700, 840))
            result_fig = np.hstack((first_col, frame_pcd))

            video.write(result_fig)
        video.release()


    def plot_image(img_file_path):

        img = cv2.imread(img_file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img)
        plt.show()


    def hex_to_rgb(self, value):
        value = value.lstrip("#")
        lv = len(value)
        return tuple(int(value[i: i + lv // 3], 16) for i in range(0, lv, lv // 3))



def transform_base_to_lidar(boxes):
    # transforms data from base to south lidar
    transformation_matrix_s110_base_to_lidar_south = np.array(
        [
            [0.21479487, 0.97627129, -0.02752358, 1.36963590],
            [-0.97610281, 0.21553834, 0.02768645, -16.19616413],
            [0.03296187, 0.02091894, 0.99923766, -6.99999999],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000],
        ]
    )
    boxes_homogeneous = np.hstack((boxes, np.ones((boxes.shape[0], 1))))
    # transform boxes from s110 base frame to lidar frame
    boxes_transformed = np.dot(transformation_matrix_s110_base_to_lidar_south, boxes_homogeneous.T).T
    return boxes_transformed[:, :3]


def transform_lidar_into_base(points):
    # transform data from south lidar to base
    transformation_matrix_lidar_south_to_s110_base = np.array(
        [
            [0.21479485, -0.9761028, 0.03296187, -15.87257873],
            [0.97627128, 0.21553835, 0.02091894, 2.30019086],
            [-0.02752358, 0.02768645, 0.99923767, 7.48077521],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000],
        ]
    )
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    # transform point cloud into s110 base frame
    points_transformed = np.dot(transformation_matrix_lidar_south_to_s110_base, points_homogeneous.T).T
    # set z coordinate to zero
    points_transformed[:, 2] = 0.0
    return points_transformed[:, :3]


def plot_mAP(eval_dict, save_output="", show_fig=True, return_fig=False):

    headers = list(eval_dict.keys())
    
    table_data = [headers]  # The first row will be headers
    
    for i in range(len(eval_dict['Categories'])):
        row = [eval_dict[header][i] for header in headers]
        table_data.append(row)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, len(eval_dict['Categories']) + 1))
    ax.axis('off')
    ax.set_facecolor('black') 
    fig.patch.set_facecolor('black') 

    table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center', edges='open')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3)  
    
    for key, cell in table.get_celld().items():
        cell.set_facecolor('black')
        cell.set_edgecolor('black')
        cell.set_text_props(color='white')
        if key[0] == 0:  # This is a header cell
            cell.set_text_props(color='white', weight='bold', size=14)
    
    if show_fig:
        plt.show()
    
    if save_output != "":
        plt.savefig(save_output)
    
    if return_fig:
        fig.canvas.draw()
        img_rgb = np.array(fig.canvas.renderer._renderer)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr
    


if __name__ == "__main__":
    
    mode = 'test'

    root_dir = '/ahmed/data/tumtraf/OpenLABEL'
    pcd_dir = os.path.join(root_dir, mode, 'point_clouds/s110_lidar_ouster_south')
    img1_dir = os.path.join(root_dir, mode, 'images/s110_camera_basler_south1_8mm')
    img2_dir = os.path.join(root_dir, mode, 'images/s110_camera_basler_south2_8mm')
    labels_dir = os.path.join(root_dir, mode, 'labels_point_clouds/s110_lidar_ouster_south')

    dets_dir = '/ahmed/output/evaluation/pv_rcnn_plusplus/C_pp/test/detections_openlabel/ckpt_140'
    log_file = '/ahmed/output/evaluation/pv_rcnn_plusplus/C_pp/test/log_eval_2023_10_11.txt'

    video_save_path = '/ahmed/output/evaluation/pv_rcnn_plusplus/C_pp/test/vis'
    os.makedirs(video_save_path, exist_ok=True)

    epoch_id = dets_dir.split('.')[0].split('_')[-1]

    vis = VisualizationTUMTraf()

    vis.create_video(
            pcd_dir=pcd_dir,
            img1_dir=img1_dir,
            img2_dir=img2_dir,
            labels_dir=labels_dir,
            dets_dir=dets_dir,
            log_path=log_file,
            video_name=f"pvrcnn_tumtraf_epoch_{epoch_id}",
            fps=5,
            output_path=video_save_path
        )


