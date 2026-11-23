# SpatialFit Chair

用户参数和桌子参数 → 10 个零件（全部由长方体组成）→ 装配顺序规划 → 在 MuJoCo 里由 UR5e 闭环装配（cuRobo 规划搬运路径）→ 记录日志（交给 Blender 渲染）

本仓库包含主流程代码、实验代码（E1–E4）和 Demo 视频的渲染/合成脚本。生成的日志、视频和实验结果不在仓库里，按下面的命令可以重新生成。

## 目录结构

| 目录 | 内容 |
|---|---|
| `spatialfit/` | 核心库：参数映射、零件生成、场景、运动规划、顺序规划、闭环执行、单次试验 |
| `configs/` | 用户/桌子参数：`A.yaml`、`B.yaml`（两把椅子），`bad.yaml`（应被拒绝的输入） |
| `scripts/` | 主流程入口（检查场景、规划顺序、闭环装配、预览）、布局搜索、视频镜头脚本、第三方库安装脚本 |
| `experiments/` | E1 单接头试验、E2/E3/E4 批量装配试验、结果汇总与作图 |
| `render/` | Blender 渲染、视频合成；`render/fonts/` 为 Inter 字体（SIL OFL，见 `LICENSE.txt`） |
| `tests/` | 开发过程中的调试和诊断脚本（cuRobo 冒烟测试、正运动学一致性、抓取/摆放排查等），不是自动化测试 |
| `build/` | 运行输出（日志、规划结果、视频），不入库 |
| `third_party/` | cuRobo 和 mujoco_menagerie，不入库，用 `scripts/setup_third_party.sh` 获取 |

## 环境安装

已验证的环境：RTX 5060 (sm_120) + WSL2 Ubuntu，gcc 13.3，Python 3.10，PyTorch 2.11.0+cu128，CUDA 12.8.1，MuJoCo 3.17.0，cuRobo 0.7.8，Blender 4.5.15 LTS。

```bash
git clone https://github.com/Shirk6/spatialfit.git && cd spatialfit
bash scripts/setup_third_party.sh       # cuRobo v0.7.8 + mujoco_menagerie（只拉 UR5e 目录）-> third_party/
mkdir -p build
```

1. 按 [`SETUP_curobo.md`](SETUP_curobo.md) 建 conda 环境 `sfit`，装 PyTorch、CUDA 工具链并编译 cuRobo。
2. 装其余依赖：
   ```bash
   E=~/miniforge3/envs/sfit
   $E/bin/python -m pip install -r requirements.txt
   $E/bin/python tests/curobo_smoke.py     # UR5e 批量逆解 + 避碰运动规划，能跑通即可
   ```
3. 只有渲染 Demo 视频才需要：Blender 4.5 LTS，以及合成视频用的第二个环境 `mjsim`（Python 3.11，opencv 5 需要 numpy ≥ 2，和 cuRobo 冲突，所以单独建）：
   ```bash
   conda create -n mjsim python=3.11 -y
   ~/miniforge3/envs/mjsim/bin/python -m pip install -r requirements-video.txt
   ```

以下所有命令都在仓库根目录下运行，`python` 指 `sfit` 环境里的 Python。

## 主流程

```bash
E=~/miniforge3/envs/sfit; export PATH=$E/bin:$PATH
python scripts/check_scene.py A                 # 参数映射 + 场景编译 + 兼容性检查（检查装配完成状态的穿透和配合间隙）
python scripts/plan_sequence.py A               # 装配顺序规划 -> build/plan_A.json（含漏斗统计、被拒绝的顺序、无夹具时各中间状态是否稳定）
python scripts/run_build.py --config A --seed 0 # 闭环装配 -> build/A_seed0.npz + _hud.json + _summary.json
python scripts/run_build.py --config A --seed 0 --bias SL:5,0,0   # 给 SL 的第一次观测加 5 mm 偏差，演示卡住 -> 重新观测 -> 重插
MUJOCO_GL=glfw python scripts/preview.py build/A_seed0.npz A build/preview_A.mp4 wide 3   # 快速预览视频
```

把 `A` 换成 `B` 就是第二把椅子。`run_build.py` 的其他参数：`--stage_sigma`（摆放位置误差，m）、`--nest_err`（腿座位置误差，mm）、`--no_perception` / `--no_force`（消融）、`--ba_push 0`（靠背支架旧布局）。

| 模块 | 文件 | 内容 |
|---|---|---|
| 参数映射 | `spatialfit/params.py` | 公式 (1)，桌子匹配检查，尺寸范围检查；不合格的输入直接拒绝 |
| 零件生成 | `spatialfit/geometry.py` | 10 个 PartSpec：长方体、接口、抓取点、依赖关系；定位端 / 浮动端（±1.5 mm）；锥形导向端 |
| 场景 | `spatialfit/scene.py` | MjSpec：UR5e（menagerie）+ 平行夹爪 + 腕部力传感器 + 腿座 + 摆放桌；用约束模拟摆放固定、抓取（柔顺）、夹具夹紧、卡扣锁定 |
| 运动 | `spatialfit/motion.py` | MuJoCo 阻尼最小二乘逆解（公式 6）；cuRobo MotionGen：夹爪碰撞球、附着零件、障碍物从 MuJoCo 同步 |
| 规划器 | `spatialfit/planner.py` | 前置依赖 → 插入路径间隙检查 → 可达性 → 稳定性（MuJoCo 仿真）；漏斗统计 |
| 执行器 | `spatialfit/executor.py` | 观测 → 规划 → 抓取并检查 → 搬运 → 按实测位姿对准 → 导纳插入（公式 7）→ 卡住时退回、重新观测、重试 → 锁定 → 松开并检查 |
| 单次试验 | `spatialfit/trial.py` | 一次完整装配（带随机扰动和消融开关），供 `run_build.py` 和批量实验共用 |

## 实验复现

先跑一遍 `scripts/plan_sequence.py A` 和 `B`，批量实验会读 `build/plan_<cfg>.json`。

**E1 单接头试验**（不用机器人，比较定位端+浮动端和两端刚性两种接头对腿间距误差的容忍度；3 种设计 × 17 个误差 × 20 次）：

```bash
python experiments/e1_coupon.py --trials 20 --jobs 8 \
  --errors 0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4
# -> experiments/results/E1/e1_trials.csv
```

**E2/E3 批量装配 + 消融**（条件：`CL` 闭环、`NOF` 无力反馈、`NOP` 不重新观测、`OL` 开环；同一种子的所有条件看到相同的扰动；已完成的试验会跳过，可以中断后续跑）：

```bash
python experiments/run_trials.py --plan "CL:A,B;NOF:A;NOP:A;OL:A" --seeds 0-49 \
  --nest_err 1.0 --stage_sigma 2.0 --stage_yaw 1.0 --out experiments/results/E23
```

**E4 压力测试**（所有扰动加倍）：

```bash
python experiments/run_trials.py --plan "CL:A,B;NOF:A;NOP:A;OL:A" --seeds 0-49 \
  --nest_err 2.0 --stage_sigma 4.0 --stage_yaw 2.0 --out experiments/results/E4
```

**靠背支架旧布局对照**（可选，`--ba_push 0`）：

```bash
python experiments/run_trials.py --plan "CL:A,B" --seeds 0-19 --ba_push 0 --out experiments/results/E2_rack0
```

单次试验约 1 分钟（墙钟时间记在每个 `*_summary.json` 的 `wall_time` 里）。要并行，开 N 个进程分别跑 `--shard k/N`（k = 0…N−1，输出目录相同），每个进程占一份 cuRobo 显存：

```bash
for k in 0 1 2 3; do python experiments/run_trials.py --plan "CL:A,B;NOF:A;NOP:A;OL:A" --seeds 0-49 \
  --out experiments/results/E23 --shard $k/4 > experiments/results/E23_w$k.log 2>&1 & done; wait
```

**汇总和作图**（E23 必需，E4、E2_rack0 存在时一起处理）：

```bash
python experiments/analyze.py --fig experiments/results/figs
# -> experiments/results/tables/*.csv|json，图 -> experiments/results/figs/*.png
```

## Demo 视频（build/video/SpatialFit_demo.mp4，1080p，30 fps，94 s）

前置：跑过 `plan_sequence.py A`、`run_build.py --config A --seed 0 --bias SL:5,0,0`（日志名 `A_seed0_bias`）和 `run_build.py --config B --seed 0`，并导出 Blender 用的场景几何：

```bash
python render/export_assets.py A build/render_A
python render/export_assets.py B build/render_B
```

然后：

```bash
E=~/miniforge3/envs/sfit; B=~/opt/blender-4.5.15-linux-x64/blender; P=~/miniforge3/envs/mjsim/bin/python
$E/bin/python scripts/video_fall2.py                      # 第 6 步 vs 第 7 步松开夹具 + 相同 3.6 N 推力（120 Hz，记录力/重心/支撑面）
$E/bin/python scripts/video_fall2_frames.py               # 慢动作帧表 -> build/video/fall2_frames.json
for k in 6 7; do $B -b --factory-startup -P render/blender_render_fall.py -- build/render_A build/fall$k.npz build/video/fall$k/ \
   --engine cycles --res 1280x720 --samples 20 --frames $($E/bin/python -c "import json;print(','.join(map(str,json.load(open('build/video/fall2_frames.json'))['log_frames'])))") \
   --cam -0.10,0.52,0.20,2.0,1.72,-21,-100,-93,42; done
$E/bin/python scripts/video_chair_frames.py                # A->B 参数变形 + 爆炸图 + 结尾 (build/video/shot_*.json)
$B -b --factory-startup -P render/blender_chair_shots.py -- build/video/shot_gen.json build/video/gen/ --samples 24
$B -b --factory-startup -P render/blender_chair_shots.py -- build/video/shot_end.json build/video/end/ --samples 24
$P scripts/video_split_frames.py                           # 只列出分屏实际用到的帧 -> build/video/tiles.json, tile_{A,B}.frames
$E/bin/python scripts/video_tags.py                        # 分屏里"正在处理的零件"标签的屏幕坐标
for s in A B; do log=$([ $s = A ] && echo A_seed0_bias || echo B_seed0)
  $B -b --factory-startup -P render/blender_render2.py -- build/render_$s build/$log.npz build/video/tile_$s/ \
     --engine cycles --res 1280x720 --samples 16 --frames $(cat build/video/tile_$s.frames); done
$P render/compose_final.py                                 # 标题、字幕、参数面板、标签、A|B 按步骤同步、交叉淡化
```

渲染整段装配视频（不只是分屏用到的帧）：`$B -b --factory-startup -P render/blender_render2.py -- build/render_A build/A_seed0.npz build/A.mp4 --engine cycles`。

## 设计说明和参考结果

**布局**
- 椅子转了 180°，靠背一侧朝向机器人。机器人底座放在 0.40 m 高的台座上。
- 腿竖立在地上，导轨平放在地上，都在重力下自然静止，没有固定约束。
- 座板和靠背的摆放位姿是把装配位姿绕机器人底座轴转一个角度：座板 −90°，平放在右侧置物架上；靠背 180°，直立在机器人正后方的支架里。这样搬运时主要是底座关节转动。
- 导轨放在左侧，腿放在左前方。

**抓取**：都夹零件本身，零件上没有专门的夹持块。
- 腿：夹角头两侧；
- 导轨：夹梁身两侧；
- 座板：夹侧边。座板前沿外伸 50 mm，所以也提供了夹前沿的抓取方式；
- 靠背：从上方夹靠背板上沿，或者从外侧夹插柱。

**抓取方式选择**：执行时对每种抓法做前瞻检查，确认按这个抓法拿起来以后，能放到装配位置，而且不会和场景碰撞、机器人也不会自碰撞。

**复现时应得到的结果**（2026-11-23）
- A、B 的规划顺序：LR → LF → RF → RR → SR → SL → FR → BR → SE → BA。
- 漏斗统计：10! 种顺序 → 满足前置依赖的 1,088 种 → 通过几何检查的 160 种。
- A、B 种子 0 都完整装配成功，夹具释放后最大位移 0.43 mm。
- 10 个零件的搬运全部由 cuRobo 规划，没有用到备选路径。
- 给侧梁 SL 的第一次观测加 5 mm 偏差：第一次插入卡住，重新观测后插入成功，整椅装配完成。

**cuRobo 规划失败的两个历史原因**（已修复，排查脚本见 `tests/curobo_diag.py`、`tests/place_table.py`）
- 夹爪碰撞球偏胖，造成机器人自碰撞的误判；
- 把等价关节解取成了需要手腕反向绕一大圈的那个，并且座板摆放的远近、高度和装配时不一致，导致构型变化过大。
