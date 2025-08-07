import numpy as np
import open3d as o3d
import config
import utils.common_utils as common


def show_mesh_with_pcd(mesh, pcd):
    """
    visualize mesh and pcd
    :param mesh:
    :param pcd:
    :return:
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    vis.add_geometry(pcd)

    box3d = mesh.get_minimal_oriented_bounding_box()
    mesh.compute_vertex_normals()

    vis.add_geometry(mesh)
    vis.add_geometry(box3d)

    vis.run()
    vis.destroy_window()


def show_mesh_with_box(mesh_obj):
    """
    visualize mesh and bounding box
    :param mesh_obj:
    :return:
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    box3d = mesh_obj.get_minimal_oriented_bounding_box()
    box_points = box3d.get_box_points()

    # box_mesh.compute_vertex_normals()
    points = np.asarray(box_points)
    print(points)
    print("height = ", np.ptp(points[:, 2]))

    mesh_obj.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh_obj])
    vis.add_geometry(mesh_obj)

    vis.add_geometry(box3d)

    # vis.add_geometry(mixed_pcd)
    vis.run()
    vis.destroy_window()


def show_pc_with_box(pc, box):
    pcd = common.pc_numpy_2_o3d(pc)
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)
    vis.add_geometry(box)
    rgb_color = [245 / 255, 144 / 255, 1 / 255]

    pcd.paint_uniform_color(rgb_color)

    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def show_bg_with_boxes(v2x_info):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    for val in v2x_info.vehicles_info.values():
        line_set = common.corner_to_line_set_box(val["corner"])
        vis.add_geometry(line_set)

    pcd = common.pc_numpy_2_o3d(v2x_info.pc)

    rgb_color = [245/255, 144/255, 1/255]

    pcd.paint_uniform_color(rgb_color)

    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def show_obj_with_car_id(v2x_info, car_id):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    corner = v2x_info.vehicles_info[car_id]['corner']
    line_set = common.corner_to_line_set_box(corner)
    vis.add_geometry(line_set)

    # pcd.paint_uniform_color([0, 0, 0])
    pcd = common.pc_numpy_2_o3d(v2x_info.pc)
    if v2x_info.is_ego:
        pcd_color = [245 / 255, 144 / 255, 1 / 255]
    else:
        pcd_color = [1, 1, 1]
    pcd.paint_uniform_color(pcd_color)

    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def show_obj_with_corner(v2x_info, corner):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size

    # render.background_color = np.array(config.lidar_config.render_background_color)

    # line_set = common.corner_to_line_set_box(corner)
    lines_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7],
                          [4, 5], [5, 6], [6, 7], [7, 4]])

    cylinders = []

    for line in lines_box:
        point1 = corner[line[0]]
        point2 = corner[line[1]]

        cylinder = common.create_cylinder_between_points(point1, point2, radius=0.03)

        cylinder.paint_uniform_color([1, 0, 0])
        cylinders.append(cylinder)

    mesh = o3d.geometry.TriangleMesh()
    for cyl in cylinders:
        mesh += cyl
    # vis.add_geometry(line_set)
    vis.add_geometry(mesh)

    # pcd.paint_uniform_color([0, 0, 0])
    pcd = common.pc_numpy_2_o3d(v2x_info.pc)
    if v2x_info.is_ego:
        pcd_color = [0, 0, 1]
        # pcd_color = [245 / 255, 144 / 255, 1 / 255]
    else:
        pcd_color = [0, 100 / 255, 0]
    pcd.paint_uniform_color(pcd_color)

    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def show_pc(v2x_info):
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=130.0)
    vis.add_geometry(axis)

    for val in v2x_info.vehicles_info.values():
        line_set = common.corner_to_line_set_box(val["corner"])
        vis.add_geometry(line_set)

    ego_pcd = common.pc_numpy_2_o3d(v2x_info.pc)

    ego_color = [245 / 255, 144 / 255, 1 / 255]
    ego_pcd.paint_uniform_color(ego_color)
    vis.add_geometry(ego_pcd)

    vis.run()
    # vis.destroy_window()


def ego_pc_and_cp_pc_to_world_coordinate(vis,ego_pc,ego_lidar_pose,cp_pc,cp_lidar_pose):

    points_3d_ego = ego_pc[:, :3]
    points_3d_cp = cp_pc[:, :3]

    homogeneous_points_ego = np.hstack((points_3d_ego, np.ones((points_3d_ego.shape[0], 1))))
    homogeneous_points_cp = np.hstack((points_3d_cp, np.ones((points_3d_cp.shape[0], 1))))

    transformed_homogeneous_points_ego = ego_lidar_pose @ homogeneous_points_ego.T
    transformed_homogeneous_points_cp = cp_lidar_pose @ homogeneous_points_cp.T
    transformed_points_3d_ego = transformed_homogeneous_points_ego[:3].T
    transformed_points_3d_cp = transformed_homogeneous_points_cp[:3].T

    # transformed_points_3d_ego[:, 2] =0
    # transformed_points_3d_cp[:,2]=0

    merged_points = np.vstack((transformed_points_3d_ego, transformed_points_3d_cp))

    ego_pcd = common.pc_numpy_2_o3d(transformed_points_3d_ego)
    cp_pcd = common.pc_numpy_2_o3d(transformed_points_3d_cp)

    ego_color = [1, 1, 1]
    ego_pcd.paint_uniform_color(ego_color)

    cp_color = [1, 1, 1]
    cp_pcd.paint_uniform_color(cp_color)
    vis.add_geometry(cp_pcd)
    
    vis.add_geometry(ego_pcd)

    # vis.run()
     

    return merged_points


def ego_pc_and_cp_pc_to_world_coordinate_vis(ego_pc,ego_lidar_pose,cp_pc,cp_lidar_pose):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
    vis.add_geometry(axis)

    points_3d_ego = ego_pc[:, :3]
    points_3d_cp = cp_pc[:, :3]

    homogeneous_points_ego = np.hstack((points_3d_ego, np.ones((points_3d_ego.shape[0], 1))))
    homogeneous_points_cp = np.hstack((points_3d_cp, np.ones((points_3d_cp.shape[0], 1))))

    transformed_homogeneous_points_ego = ego_lidar_pose @ homogeneous_points_ego.T
    transformed_homogeneous_points_cp = cp_lidar_pose @ homogeneous_points_cp.T
    transformed_points_3d_ego = transformed_homogeneous_points_ego[:3].T
    transformed_points_3d_cp = transformed_homogeneous_points_cp[:3].T

    ego_pcd = common.pc_numpy_2_o3d(transformed_points_3d_ego)
    cp_pcd = common.pc_numpy_2_o3d(transformed_points_3d_cp)

    ego_color = [245 / 255, 144 / 255, 1 / 255]
    ego_pcd.paint_uniform_color(ego_color)

    cp_color = [1, 1, 1]
    cp_pcd.paint_uniform_color(cp_color)
    vis.add_geometry(cp_pcd)
    
    vis.add_geometry(ego_pcd)

    vis.run()

    # return transformed_points_3d



def ego_pc_to_world_coordinate_vis(pc,lidar_pose):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
    vis.add_geometry(axis)

    points_3d = pc[:, :3]

    homogeneous_points = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    transformed_homogeneous_points = lidar_pose @ homogeneous_points.T

    transformed_points_3d = transformed_homogeneous_points[:3].T

    ego_pcd = common.pc_numpy_2_o3d(transformed_points_3d)

    ego_color = [245 / 255, 144 / 255, 1 / 255]
    ego_pcd.paint_uniform_color(ego_color)
    vis.add_geometry(ego_pcd)

    vis.run()

    # return transformed_points_3d



def show_ego_and_cp_pc(ego_info, cp_info):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    for val in ego_info.vehicles_info.values():
        print(val)
        line_set = common.corner_to_line_set_box(val["corner"])
        vis.add_geometry(line_set)

    for val in cp_info.vehicles_info.values():
        line_set = common.corner_to_line_set_box(val["corner"])
        vis.add_geometry(line_set)

    # pcd.paint_uniform_color([0, 0, 0])
    T_cp2ego = np.linalg.inv(ego_info.param["lidar_pose"]) @ cp_info.param["lidar_pose"]
    ego_pcd = common.pc_numpy_2_o3d(ego_info.pc)
    cp_pcd = common.pc_numpy_2_o3d(cp_info.pc).transform(T_cp2ego)

    ego_color = [197 / 255, 112 / 255, 139 / 255]
    ego_pcd.paint_uniform_color(ego_color)
    vis.add_geometry(ego_pcd)
    cp_color = [65 / 255, 176 / 255,73 / 255]
    cp_pcd.paint_uniform_color(cp_color)
    vis.add_geometry(cp_pcd)

    vis.run()
    vis.destroy_window()


def show_ego_and_cp_with_id(ego_info, cp_info, ego_id, cp_id):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    # pcd.paint_uniform_color([0, 0, 0])
    T_cp2ego = np.linalg.inv(ego_info.param["lidar_pose"]) @ cp_info.param["lidar_pose"]
    ego_pcd = common.pc_numpy_2_o3d(ego_info.pc)
    cp_pcd = common.pc_numpy_2_o3d(cp_info.pc).transform(T_cp2ego)

    ego_color = [245 / 255, 144 / 255, 1 / 255]
    ego_pcd.paint_uniform_color(ego_color)
    cp_color = [1, 1, 1]
    cp_pcd.paint_uniform_color(cp_color)
    vis.add_geometry(cp_pcd)

    corner = ego_info.vehicles_info[ego_id]['corner']
    line_set = common.corner_to_line_set_box(corner)
    vis.add_geometry(line_set)

    corner = cp_info.vehicles_info[cp_id]['corner']
    
    line_set = common.corner_to_line_set_box(corner).transform(T_cp2ego)
    vis.add_geometry(line_set)

    vis.add_geometry(ego_pcd)
    vis.add_geometry(cp_pcd)

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    vis.add_geometry(coordinate)
    # vis.get_render_option().background_color = [1.0, 1.0, 1.0]

    vis.run()
    vis.destroy_window()


def show_ego_and_cp_with_corner(ego_info, cp_info, corner):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    # render.background_color = np.array(config.lidar_config.render_background_color)
    lines_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7],
                          [4, 5], [5, 6], [6, 7], [7, 4]])

    cylinders = []

    for line in lines_box:
        point1 = corner[line[0]]
        point2 = corner[line[1]]

        cylinder = common.create_cylinder_between_points(point1, point2, radius=0.05)

        cylinder.paint_uniform_color([1, 0, 0])
        cylinders.append(cylinder)

    mesh = o3d.geometry.TriangleMesh()
    for cyl in cylinders:
        mesh += cyl
    # vis.add_geometry(line_set)
    vis.add_geometry(mesh)

    # pcd.paint_uniform_color([0, 0, 0])
    T_cp2ego = np.linalg.inv(ego_info.param["lidar_pose"]) @ cp_info.param["lidar_pose"]
    ego_pcd = common.pc_numpy_2_o3d(ego_info.pc)
    cp_pcd = common.pc_numpy_2_o3d(cp_info.pc).transform(T_cp2ego)

    # ego_color = [0, 0, 1]
    cp_color = [0, 0, 1]
    ego_color = [0, 75 / 255, 0]
    ego_pcd.paint_uniform_color(ego_color)
    # cp_color = [0, 100 / 255, 0]
    cp_pcd.paint_uniform_color(cp_color)
    vis.add_geometry(cp_pcd)

    line_set = common.corner_to_line_set_box(corner)
    vis.add_geometry(line_set)

    vis.add_geometry(ego_pcd)
    vis.add_geometry(cp_pcd)

    vis.run()
    vis.destroy_window()


def show_ego_and_cp_for_translation(ego_info, cp_info, car_id, corner):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    T_cp2ego = np.linalg.inv(ego_info.param["lidar_pose"]) @ cp_info.param["lidar_pose"]
    ego_pcd = common.pc_numpy_2_o3d(ego_info.pc)
    cp_pcd = common.pc_numpy_2_o3d(cp_info.pc).transform(T_cp2ego)

    ego_color = [245 / 255, 144 / 255, 1 / 255]
    ego_pcd.paint_uniform_color(ego_color)
    cp_color = [1, 1, 1]
    cp_pcd.paint_uniform_color(cp_color)
    vis.add_geometry(cp_pcd)

    ego_corner = ego_info.vehicles_info[car_id]['corner']
    ego_line_set = common.corner_to_line_set_box(ego_corner)
    vis.add_geometry(ego_line_set)

    line_set = common.corner_to_line_set_box(corner, [1, 1, 1])
    vis.add_geometry(line_set)

    vis.add_geometry(ego_pcd)
    vis.add_geometry(cp_pcd)

    vis.run()
    vis.destroy_window()


def show_obj_for_translation(v2x_info, car_id, corner):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    pcd = common.pc_numpy_2_o3d(v2x_info.pc)

    if v2x_info.is_ego:
        pcd_color = [245 / 255, 144 / 255, 1 / 255]
    else:
        pcd_color = [1, 1, 1]
    pcd.paint_uniform_color(pcd_color)

    cur_corner = v2x_info.vehicles_info[car_id]['corner']
    cur_line_set = common.corner_to_line_set_box(cur_corner)
    vis.add_geometry(cur_line_set)

    line_set = common.corner_to_line_set_box(corner, [1, 1, 1])
    vis.add_geometry(line_set)

    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()

