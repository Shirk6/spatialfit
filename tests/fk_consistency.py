"""Compare MuJoCo menagerie UR5e (attachment_site) with cuRobo ur5e (tool0) forward kinematics."""
import numpy as np, mujoco, torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.util_file import get_robot_configs_path, join_path, load_yaml

ROOT = Path(__file__).resolve().parents[1]
m = mujoco.MjModel.from_xml_path(str(ROOT / "third_party/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"))
d = mujoco.MjData(m)
site = m.site("attachment_site").id; base = m.body("base").id
t = TensorDeviceType()
cfg = load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))["robot_cfg"]
km = CudaRobotModel(CudaRobotModelConfig.from_data_dict(cfg["kinematics"], tensor_args=t) if False else
                    CudaRobotModelConfig.from_basic_urdf(join_path(__import__('curobo.util_file', fromlist=['x']).get_assets_path(), cfg["kinematics"]["urdf_path"]), "base_link", "tool0", t))
rng = np.random.default_rng(0)
Q = rng.uniform(-np.pi, np.pi, (200, 6))
st = km.get_state(t.to_device(Q))
cp = st.ee_position.cpu().numpy(); cq = st.ee_quaternion.cpu().numpy()  # wxyz
def T(p, qwxyz):
    M = np.eye(4); M[:3, :3] = R.from_quat(qwxyz[[1, 2, 3, 0]]).as_matrix(); M[:3, 3] = p; return M
Tm = []
for q in Q:
    d.qpos[:6] = q; mujoco.mj_forward(m, d)
    Tb = np.eye(4); Tb[:3, :3] = d.xmat[base].reshape(3, 3); Tb[:3, 3] = d.xpos[base]
    Ts = np.eye(4); Ts[:3, :3] = d.site_xmat[site].reshape(3, 3); Ts[:3, 3] = d.site_xpos[site]
    Tm.append(np.linalg.inv(Tb) @ Ts)
Tc = [T(cp[i], cq[i]) for i in range(len(Q))]
# MuJoCo: Tm = A Tc B. Estimate B from first sample given A=I candidates; solve A,B via sample 0, check rest
A_guess = np.eye(4)
B = np.linalg.inv(Tc[0]) @ np.linalg.inv(A_guess) @ Tm[0]
errs = [np.linalg.norm((A_guess @ Tc[i] @ B)[:3, 3] - Tm[i][:3, 3]) for i in range(len(Q))]
print("A=I: B=\n", B.round(4), "\nmax pos err (m):", max(errs))
Rz = np.diag([-1, -1, 1, 1.0])
B2 = np.linalg.inv(Tc[0]) @ np.linalg.inv(Rz) @ Tm[0]
errs2 = [np.linalg.norm((Rz @ Tc[i] @ B2)[:3, 3] - Tm[i][:3, 3]) for i in range(len(Q))]
print("A=Rz(pi): B=\n", B2.round(4), "\nmax pos err (m):", max(errs2))
