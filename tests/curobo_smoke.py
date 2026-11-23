"""cuRobo smoke test on this GPU: UR5e batch IK + collision-aware MotionGen around a box."""
import time
import torch
from curobo.geom.types import WorldConfig, Cuboid
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

t = TensorDeviceType()
print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
robot_cfg = load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))["robot_cfg"]
world = WorldConfig(cuboid=[Cuboid(name="box", pose=[0.45, 0.0, 0.25, 1, 0, 0, 0], dims=[0.15, 0.6, 0.5])])

# 1) Batch IK
ik = IKSolver(IKSolverConfig.load_from_robot_config(robot_cfg, world, tensor_args=t, num_seeds=20,
                                                    self_collision_check=True, use_cuda_graph=True))
q = torch.rand((1000, ik.dof), **t.as_torch_dict()) * 2 - 1
goal = ik.fk(q).ee_pose
ik.solve_batch(goal)  # warm-up
torch.cuda.synchronize(); t0 = time.time()
r = ik.solve_batch(goal)
torch.cuda.synchronize()
print(f"batch IK: 1000 poses in {time.time()-t0:.3f}s, success {r.success.float().mean().item():.1%}")

# 2) MotionGen around obstacle
mg = MotionGen(MotionGenConfig.load_from_robot_config(robot_cfg, world, t, interpolation_dt=0.01,
                                                      use_cuda_graph=True))
mg.warmup()
start = JointState.from_position(t.to_device([[0.0, -2.2, 1.9, -1.3, -1.57, 0.0]]))
goal = Pose.from_list([0.45, 0.35, 0.35, 0.0, 1.0, 0.0, 0.0])  # beside the box, tool pointing down
t0 = time.time()
res = mg.plan_single(start, goal, MotionGenPlanConfig(max_attempts=5))
print(f"MotionGen: success={res.success.item()} time={time.time()-t0:.3f}s "
      f"traj_points={res.get_interpolated_plan().position.shape[0] if res.success.item() else 0}")
if res.success.item():
    ee = mg.compute_kinematics(res.get_interpolated_plan()[-1:]).ee_pos_seq
    print("final ee pos:", ee.cpu().numpy().round(4))
