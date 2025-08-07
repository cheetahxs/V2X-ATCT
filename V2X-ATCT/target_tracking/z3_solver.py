import pickle
import numpy as np

from z3 import Real, Solver, And,sat,Or,Abs,Sqrt
import math



def find_point_with_min_distance(points, min_distance=2.0,MaxLimit=20,MinLimit=10,y_min_limit=-2,y_max_limit=2):
    solver = Solver()
    x = Real('x')
    y = Real('y')
    

    solver.add(x >= MinLimit, x <= MaxLimit)
    solver.add(y >= y_min_limit, y <= y_max_limit)

    for (px, py, pz) in points:
        distance_squared = (x - px) ** 2 + (y - py) ** 2
        solver.add(distance_squared >= min_distance ** 2)
    

    if solver.check() == sat:
        model = solver.model()
        x_val = z3_to_float(model[x].as_decimal(10))
        y_val = z3_to_float(model[y].as_decimal(10))
        return (x_val, y_val)
        
    else:
        return -999,-999


def find_point_one_line_2(
    k,
    b,
    point,
    oripoint,
    distance=25
):
    x = Real('x')
    y = Real('y')
    
    solver = Solver()
    solver.add(y == k * x + b)
    solver.add( (x-oripoint[0])*(oripoint[0]-point[0])+(y-oripoint[1])*(oripoint[1]-point[1]) < 0 )
    

    dx = x - oripoint[0]
    dy = y - oripoint[1]
    solver.add(dx * dx + dy * dy <= distance * distance+0.1)
    solver.add(dx * dx + dy * dy >= distance * distance-0.1)
    

    if solver.check() == sat:
        model = solver.model()
        x_val = z3_to_float(model[x].as_decimal(10))
        y_val = z3_to_float(model[y].as_decimal(10))
        return (x_val, y_val)
    else:
        return None


    return



def find_point_on_line(B,
                       C,
                        n, k, b,direction='right'):

    x = Real('x')
    y = Real('y')
    solver = Solver()
    solver.add(y == k * x + b)

    if direction == 'right':
        a_1 = C[0] - B[0]
        b_1 = C[1] - B[1]


        solver.add( ((x-B[0])*b_1 - (y-B[1])*a_1) > 0 )
        
    else :
        a_1 = C[0] - B[0]
        b_1 = C[1] - B[1]
        solver.add( ((x-B[0])*b_1 - (y-B[1])*a_1) < 0 )
    
    dx = x - B[0]
    dy = y - B[1]
    solver.add(dx * dx + dy * dy <= n * n+0.1)
    solver.add(dx * dx + dy * dy >= n * n-0.1)
    
    if solver.check() == sat:
        model = solver.model()
        x_val = z3_to_float(model[x].as_decimal(10))
        y_val = z3_to_float(model[y].as_decimal(10))
        return (x_val, y_val)
    else:
        return None


def z3_to_float(z3_expr):
    s = str(z3_expr)
    if s.endswith('?'):
        s = s[:-1]
    return float(s)


def find_point(k, b, a_dist, b_dist,road_dir,y_range):
    x = Real('x')
    y = Real('y')
    distance = Abs(k * x - y + b) / Sqrt(k * k + 1)

    solver = Solver()
    if road_dir == 'same':
        solver.add(y < k * x + b)
    else:
        solver.add(y > k * x + b)

    solver.add(a_dist <= distance, distance <= b_dist)
    solver.add(y <= y_range[1], y >= y_range[0])

    if solver.check() == sat:
        model = solver.model()
        x_val = float(model[x].as_decimal(10))
        y_val = float(model[y].as_decimal(10))
        return x_val, y_val
    else:
        return None



def calculate_x_y_next_insert(distance,x_lane,y_lane,k,b,condition='x<x_lane'):
    if condition=='x<x_lane':
        x = Real('x')
        distance_constraint = (x - x_lane) ** 2 + (k*x+b - y_lane) ** 2 == distance*distance
        x_constraint = x < x_lane
        solver = Solver()
        solver.add( distance_constraint, x_constraint)
        if solver.check() == sat:
            model = solver.model()                     
            x_value = model[x].as_decimal(5)
            return x_value
        else:
            print("can not find satisfied value!")
            return False




def calculate_x_y_in_lane_with_distance(last_lane_point,lane,distance):
    x0, y0 = last_lane_point
    x_discrete = np.linspace(x0-distance-0.1, x0, 100)

    for i in range(len(x_discrete)):
        x=x_discrete[i]
        y=lane(x)
        if (x-x0)**2 + (y-y0)**2 -distance*distance <0.2:
            return x
        
    return False
    



def calculate_x_and_y_in_init(x0,y0,n,k,b,condition='x<x0'):
    
    if condition == 'x<x0':
        x = Real('x')
        s = Solver()
        distance_constraint_max = (x - x0) * (x - x0) + (x*k+b - y0) * (x*k+b - y0) <= n * n+0.1
        distance_constraint_min = (x - x0) * (x - x0) + (x*k+b - y0) * (x*k+b - y0) >= n * n-0.1
        x_constraint = x < x0
        s.add(And(distance_constraint_max,distance_constraint_min, x_constraint))

        if s.check() == sat:
            m = s.model()
            x_value = m[x].as_decimal(5)
            return x_value
        else:
            print("No solution found.")

def calculate_angle_between_line_and_x_axis(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    if x2 - x1 == 0:
        if y2 - y1 > 0:
            return 90
        elif y2 - y1 < 0:
            return - 90
        else:
            return 0
    slope = (y2 - y1) / (x2 - x1)
    angle_rad = math.atan(slope)
    angle_deg = math.degrees(angle_rad)
    return angle_deg



def get_tangent_y_intercept_poly(x_point, coefficients):

    y_point = np.polyval(coefficients, x_point)
    derivative_coeffs = np.polyder(coefficients)
    slope = np.polyval(derivative_coeffs, x_point)
    y_intercept = y_point - x_point * slope
    return y_intercept


def world_to_ego_coordinate(transformed_x, transformed_y, transformed_z, lidar_pose):
    transformed_point_3d = np.array([[transformed_x, transformed_y, transformed_z]])
    transformed_homogeneous_point = np.hstack((transformed_point_3d, np.ones((transformed_point_3d.shape[0], 1))))
    inverse_lidar_pose = np.linalg.inv(lidar_pose)
    original_homogeneous_point = inverse_lidar_pose @ transformed_homogeneous_point.T
    original_point_3d = original_homogeneous_point[:3].T
    original_x, original_y, original_z = original_point_3d[0][:3]

    return original_x, original_y

def ego_xyz_to_world_coordinate(x, y, z,lidar_pose):
    z = 0
    point_3d = np.array([[x, y, z]])
    homogeneous_point = np.hstack((point_3d, np.ones((point_3d.shape[0], 1))))
    transformed_homogeneous_point = lidar_pose @ homogeneous_point.T
    transformed_point_3d = transformed_homogeneous_point[:3].T
    transformed_x, transformed_y,transformed_z = transformed_point_3d[0][:3]

    return transformed_x, transformed_y

def ego_xyz_to_world_coordinate_require_z(x, y, z,lidar_pose):
    lidar_point=[x,y,z]   
    lidar_point = np.array(lidar_point)
    lidar_point_homogeneous = np.append(lidar_point, 1)
    world_point_homogeneous = np.dot(lidar_pose, lidar_point_homogeneous)
    world_point = world_point_homogeneous[:3]

    return world_point





    