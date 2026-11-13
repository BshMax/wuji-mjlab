# Z5 转笔独立部署包

整个 `z5-pen-spin-deploy/` 目录可以复制到其他机器，也可以改名或移动。请保留其内部相对路径：

```text
z5-pen-spin-deploy/
  README.md
  requirements-sim.txt
  deploy/                           # 推理、视觉和硬件控制代码
  generated/deploy/z5-pen-spin/       # 已导出模型、参考轨迹、配置和仿真场景
  LICENSES/
  NOTICE
```

运行不需要原始训练仓库、`model_final.pt`、Isaac Lab 或 Newton。包内的 `export_policy.py` 用于在完整训练仓库中重新导出；接收方直接运行现成模型即可。`policy_config.json` 中的原 checkpoint 绝对路径仅记录来源，部署时不会读取该路径。

## 安装与检查

以下以 Linux / Python 3.12 为例。在解压后的包根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-sim.txt

# 纯推理与控制链路检查，不连接硬件。
python deploy/z5_pen_spin/scripts/play_real.py \
  --steps 100 --hold-seconds 0 --no-realtime

# 无窗口 MuJoCo 检查。
python deploy/z5_pen_spin/scripts/play_real.py \
  --backend mujoco --steps 100 --hold-seconds 0 --no-realtime

# 有图形桌面时查看仿真。
python deploy/z5_pen_spin/scripts/play_real.py \
  --backend mujoco --viewer --hold-seconds 0
```

`requirements-sim.txt` 固定了二进制场景对应的 MuJoCo 版本。模型推理使用 CPU，不要求 CUDA。相机、ONNX Runtime 等部署依赖也列在运行时 requirements 中；Unitree SDK 需另外配置。

## 实机配置

详细操作见 [部署说明](deploy/z5_pen_spin/README.md)。实机需要：

- 配置相机内参及笔/掌标记角点，或接入符合说明中协议的视觉位姿发布程序。
- 安装与当前 Python ABI 匹配的 `unitree_sdk2py` 和 CycloneDDS。
- 指定硬件网卡与 `--wire-convention legacy` 或 `--wire-convention urdf20261103`。

可先运行 `check_pose.py` 只读核对视觉，再使用 `play_real.py --backend real`。

## 验证范围

已完成导出数值一致性、观测/控制契约、合成视觉和无硬件运行检查。MuJoCo 前 100 步通过；首段长程回放在第 586 个控制步触发笔轴误差门限，尚未验证长程稳定性或真机效果。原始检查结果见 `generated/deploy/z5-pen-spin/deployment_checks.json`。
