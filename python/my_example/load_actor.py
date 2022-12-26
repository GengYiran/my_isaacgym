"""
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Actor Scaling
------------
- Loads a handful of MJCF and URDF assets and scales them using the runtime scaling API
"""

import math
import numpy as np
from isaacgym import gymapi, gymutil

args = gymutil.parse_arguments(
    description="Actor scaling. Demonstrates runtime scaling of actors",
    custom_parameters=[
        {"name": "--min_scale", "type": float, "default": 0.5, "help": "Lower scale value"},
        {"name": "--max_scale", "type": float, "default": 2.0, "help": "Upper scale value"},
        {"name": "--num_columns", "type": int, "default": 4, "help": "Number of actors from the same asset in one row"}
    ]
)

# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load asset
asset_root = "assets"


asset_file = "urdf/handcraft_cabinet/mobility.urdf"

asset_options = gymapi.AssetOptions()
asset_options.density = 500
asset_options.collapse_fixed_joints = True
asset_options.fix_base_link = True
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset=gym.load_asset(sim, asset_root, asset_file, asset_options)

# set up the env grid
spacing = 1
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(1, 1, 1.0)
cam_target = gymapi.Vec3(0, -0, 1.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)


# create env
env = gym.create_env(sim, env_lower, env_upper, args.num_columns)

# add actor
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.5, 0.0)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

actor_handle = gym.create_actor(env, asset, pose, "actor", 0, 1)

while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
