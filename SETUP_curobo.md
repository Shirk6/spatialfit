# 在 RTX 5060 (sm_120) + WSL2 上安装 cuRobo：已验证可行（2026-11-22）

结果：`curobo` v0.7.8 的单元测试 **257 个通过、11 个跳过、0 个失败**；`tests/curobo_smoke.py` 测试了 UR5e 的批量逆解（1000 个位姿用时 0.23 s）和避碰运动规划（用时 0.07 s），都正常运行。

| 组件 | 版本 |
|---|---|
| Python | 3.10（conda 环境 `sfit`） |
| PyTorch | 2.11.0+cu128（架构列表包含 sm_120） |
| CUDA 工具链 | 12.8.1（conda，`nvidia/label/cuda-12.8.1`） |
| 系统编译器 | gcc 13.3 |
| warp-lang | **1.8.1**（1.19 去掉了 `wp.torch`，cuRobo 0.7.8 会因此报错） |
| numpy | < 2 |

```bash
conda create -n sfit python=3.10 -y
E=~/miniforge3/envs/sfit
$E/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
conda install -y -n sfit -c nvidia/label/cuda-12.8.1 cuda-toolkit
$E/bin/python -m pip install "numpy<2" setuptools_scm wheel ninja pytest
bash scripts/setup_third_party.sh                          # 拉取 cuRobo v0.7.8（和 menagerie）到 third_party/
cd third_party/curobo
# TORCH_CUDA_ARCH_LIST 按显卡改：RTX 50 系 12.0，RTX 40 系 8.9，RTX 30 系 8.6
# conda 版 CUDA 的头文件和库在 targets/x86_64-linux 下，需要手动指定路径
export PATH=$E/bin:$PATH CUDA_HOME=$E TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8 \
       CPATH=$E/targets/x86_64-linux/include LIBRARY_PATH=$E/targets/x86_64-linux/lib:$E/lib
$E/bin/python -m pip install -e . --no-build-isolation      # 本机约 3.5 分钟
$E/bin/python -m pip install "warp-lang==1.8.1"             # 必须在安装 cuRobo 之后再降级
$E/bin/python -m pytest tests -q                            # 期望：257 passed
```

踩过的坑：
1. `fatal error: cuda_runtime.h: No such file or directory`：没有设置 `CPATH`。
2. `module 'warp' has no attribute 'torch'`：warp 版本太新，需要降到 1.8.1。
