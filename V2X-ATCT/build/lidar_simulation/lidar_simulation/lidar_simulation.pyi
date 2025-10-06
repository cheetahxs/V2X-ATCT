from _typeshed import Incomplete
from astropy.coordinates import Latitude as Latitude

lidar_config: Incomplete
simulation_mode: str
vertical_resolution: float

def lidar_simulation(mesh_obj, sim_mode: str = ...): ...
def lidar_intensity_convert(bg_pcd, obj_pcd): ...
def render_pcd(pointcloud_xyz, average, variance, severity, loss_rate): ...
