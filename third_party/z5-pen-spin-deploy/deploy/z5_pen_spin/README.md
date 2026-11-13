# Z5 转笔部署

参照 `wuji-mjlab/deploy/reorient` 的策略、视觉接收、控制循环与硬件驱动分层，实现本项目 SuperMimic 的 Z5 转笔部署。默认 checkpoint：

```text
logs/dex_supermimic/z5-pen-spin/z5-com-xi-native-resume20000-e8192-utd1-20261105-101255/model_final.pt
```

训练模型的输入是 163 维，输出是 22 个关节的残差。转笔必须同步播放参考轨迹；`wuji-mjlab` 的方块目标、20 关节策略和绝对位置 EMA 控制不能直接用于此模型。

## 导出与无硬件运行

在仓库根目录执行，导出使用现有训练环境，不需要启动 Isaac Lab 仿真或 GPU：

```bash
./scripts/python.sh deploy/z5_pen_spin/scripts/export_policy.py

# 查看该次训练选用的 9 段轨迹及长度。
./scripts/python.sh deploy/z5_pen_spin/scripts/play_real.py --list-clips

# 验证推理、观测、EMA、关节限位和日志；默认 backend=mock，不连接硬件。
./scripts/python.sh deploy/z5_pen_spin/scripts/play_real.py \
  --steps 100 --hold-seconds 0 --no-realtime \
  --log generated/deploy/z5-pen-spin/mock_trace.npz

# 在导出的真实模型上进行 CPU MuJoCo 闭环回放。
./scripts/python.sh deploy/z5_pen_spin/scripts/play_real.py \
  --backend mujoco --steps 100 --hold-seconds 0 --no-realtime \
  --log generated/deploy/z5-pen-spin/mujoco_trace.npz

# 交互查看；需要图形桌面。不要同时使用 --no-realtime。
./scripts/python.sh deploy/z5_pen_spin/scripts/play_real.py \
  --backend mujoco --viewer --hold-seconds 0
```

默认目录为 `generated/deploy/z5-pen-spin/`，由 `.gitignore` 排除，可重新生成。`--checkpoint`、`--output` 可指定其他路径，导出时需要 checkpoint 同目录的 `environment_config.json`、其中记录的 canonical MJB 和原始轨迹目录。导出器复用训练的 `SingleHandTrackingReader`，保留 wrist→palm 变换、21 个关键点的顺序、逐帧参考和模型内归一化；会验证轨迹哈希、维度、控制周期和关节顺序。

| 产物 | 内容 |
|---|---|
| `policy.ts` | 默认 CPU TorchScript 推理，包含归一化和输出 tanh |
| `policy.onnx` | ONNX 推理，用 `--engine onnx` 选择 |
| `policy_config.json` | 关节顺序、观测/动作参数、限位、来源 checkpoint 哈希等 |
| `reference.npz` | 训练选用的 9 段轨迹，8,918 帧；已转换到 palm 坐标系 |
| `scene.mjb` | 可选 MuJoCo 回放模型，约 100 MB，需匹配配置中的 MuJoCo 版本 |
| `validation.json` | TorchScript/ONNX 与原 Actor 的数值误差及数据规模 |

部署机只需复制 `deploy/` 和 bundle；用部署机的 `python` 替换示例中的 `./scripts/python.sh`。实机推理不导入 Isaac Lab、Newton、mjlab 或训练包。依赖见 [requirements.txt](requirements.txt)：TorchScript 需要 NumPy/Torch，ONNX 需要 NumPy/ONNX Runtime；实机需要 PyZMQ 和 Unitree SDK；本项目的相机入口另需 OpenCV contrib。相机也可使用独立进程和环境。

`unitree_sdk2py` 需要匹配解释器 ABI 和 CycloneDDS。现有 `Deploy/dexhand_sysid/third_party/dex5/unitree_sdk2_python` 中的原生扩展是 CPython 3.10 构建，不能直接加载进训练环境的 Python 3.12。可用兼容的部署 Python 环境，或重新编译 SDK。代码不自动修改训练环境。

## 转交给其他人

单独复制 `deploy/` 只包含代码，还需附带 `generated/deploy/z5-pen-spin/` 中的导出模型与参考轨迹。可在本机一次打包：

```bash
./scripts/python.sh deploy/z5_pen_spin/scripts/package_deploy.py
```

产物为 `generated/releases/z5-pen-spin-deploy.tar.gz`，包含部署代码、两个推理模型、参考轨迹、配置、MuJoCo 场景与对应版本依赖说明。无需提供原始 checkpoint 或整个训练仓库；运行日志不会打包。接收方解压后按包根目录的 README 操作，使用本机 `python` 运行；其内容也可见 [独立部署包说明](PORTABLE.md)。

## 视觉位姿

```text
相机与标记 → pen_world_observer → ZMQ → play_real → Z5HandDriver → DDS
                                          ↑                      ↓
                                    参考轨迹 + Actor          关节 q / dq
```

复制 [config/camera.example.json](config/camera.example.json) 为本机配置，填写相机内参、畸变、标定分辨率以及每个标记四个角点的实测位置。示例中的空内参和空标记表需要填写后才能启动，仓库不包含这台相机及 Z5 安装夹具的标定数据。

`pen_markers` 的角点在**笔本体坐标系**中，单位米，原点为笔中心，轴向为 **+X**。`palm_markers` 的角点在训练资产 **palm_link 坐标系**中；若标记贴在手腕或夹具上，需要将实测的安装变换算进角点，不能把 AprilTag 坐标系直接当作 palm。每个条目的格式是：

```json
{
  "id": 0,
  "corners_m": [
    [-0.01, 0.01, 0.0],
    [0.01, 0.01, 0.0],
    [0.01, -0.01, 0.0],
    [-0.01, -0.01, 0.0]
  ]
}
```

这个例子表示原点处 20 mm 的平面标记，不代表实际笔/手的安装位置。四个角必须对应 OpenCV 检测返回的左上、右上、右下、左下顺序；ID、字典和角点位置要与实物一致。默认使用笔 `DICT_4X4_50`、手掌 `DICT_APRILTAG_36h11`。Observer 每帧分别求解相机中的笔和掌位姿，再计算 `T_palm_pen = inverse(T_camera_palm) @ T_camera_pen`。两者同帧可见、PnP 重投影误差合格时才发布有效位姿。

```bash
python deploy/z5_pen_spin/scripts/pen_world_observer.py \
  --config /path/to/camera.json --camera 0 --preview

# 另一个终端：只读检查掌坐标系中的实测位姿和参考首帧误差，不启动 DDS。
python deploy/z5_pen_spin/scripts/check_pose.py \
  --bundle /path/to/z5-pen-spin --clip 03_continuous_multi_axis/clip_001
```

默认发布到 `tcp://*:5555`。也可以接入已有视觉程序，发送**单帧、无 topic 前缀**的 ZMQ JSON：

```json
{
  "timestamp": 1794355200.0,
  "frame": "palm_link",
  "quaternion_order": "wxyz",
  "valid": true,
  "pen": {
    "position": [0.01, 0.02, 0.03],
    "quaternion": [1.0, 0.0, 0.0, 0.0]
  }
}
```

`timestamp` 必须是当前帧采集开始时的 Unix 秒数，例子中的时间需替换。发送端与控制端时钟应同步。短时漏检保留上次有效值；无效帧和乱序帧不会刷新有效时间。超过默认 0.3 秒就停止控制。不会将参考轨迹伪装成实机笔观测。`mock` 的笔观测来自参考，仅用于检查数据链路；MuJoCo 的笔观测来自物理状态。

## 实机运行

右手 DDS topic 为 `rt/dex5/right/cmd` 和 `rt/dex5/right/state`，消息为 `unitree_hg.HandCmd_ / HandState_`，22 个 `MotorCmd_`。驱动按名称完成策略的拇指优先顺序与 DDS 的食指优先顺序之间的转换。

实机必须显式指定 `--wire-convention`：

| 选项 | 编码约定 |
|---|---|
| `legacy` | 旧 Z5 MJCF / 旧 `dex5_lonicera` 约定，与训练一致 |
| `urdf20261103` | 新 SDK/URDF 约定；thumb pitch、little pitch、little roll1 的位置和速度都取反；thumb yaw1 上限限制为 1.57 rad |

上述约定来自本机 Z5 系统辨识代码中的映射。新 URDF 还存在拇指轴倾角等几何差异，标量符号转换不能消除这些几何差异。手掌安装方向也应与训练一致，固定掌的世界位姿记录在 bundle 配置中；相对位姿转换不会消除重力方向改变带来的动力学差异。

准备好视觉和匹配 SDK 后，执行：

```bash
python deploy/z5_pen_spin/scripts/play_real.py \
  --bundle /path/to/z5-pen-spin \
  --backend real --wire-convention urdf20261103 \
  --dds-interface eth0 \
  --pose-endpoint tcp://127.0.0.1:5555 \
  --clip 03_continuous_multi_axis/clip_001 \
  --move-seconds 3 --hold-seconds 3 \
  --log logs/deploy/z5-pen-spin-real.npz
```

如 SDK 没有安装到当前环境，可加 `--sdk-path /path/to/unitree_sdk2_python`。`eth0` 替换为连接手部的网卡。默认 PD 为本次训练的 `kp=2.0`、`kd=0.04`，支持 `--kp` / `--kd` 指定硬件值。

启动顺序：等待新鲜视觉 → 等待新鲜关节状态 → 从实测关节平滑移动到首帧 → 首帧闭环保持 3 秒 → 50 Hz 逐帧播放所选轨迹 → 结束并将 PD 增益/前馈清零、关闭 DDS。请提前将笔放在该轨迹首帧对应的手内位置。保持阶段仍由策略闭环控制，参考时钟保持首帧，残差从零初始化。默认只播放一次；不会自动跳回首帧或执行抓笔恢复。

下发目标受 MJCF 关节限位与默认 8 rad/s 的目标变化率限制；`target_guard_interventions` 统计被修改的控制步，NPZ 同时记录原目标与实际下发目标。这些限制位于训练控制器之后，会影响行为。默认笔误差超过 5 cm 或笔轴偏角超过 60°、关节状态过期 0.2 秒、视觉过期 0.3 秒、电机温度超过 65°C、控制循环超时 0.2 秒时退出。Ctrl+C 同样清零增益并关闭驱动。`--no-realtime` 与 `--no-tracking-guard` 仅允许用于无硬件模式。

## 与训练保持一致的部分

| 观测切片 | 内容 |
|---|---|
| `[0:91]` | 当前参考：22 关节、6 个 bounded 笔轴坐标、63 个手部关键点坐标 |
| `[91:113]` | 编码器 q 减默认关节位置 |
| `[113:135]` | 编码器 dq 减默认速度，再乘 0.25 |
| `[135:141]` | 实测笔位姿的 +X/-X 两点，距离笔中心 ±0.1 m，逐元素 `0.15*tanh(point/0.15)` |
| `[141:163]` | 上一步原始 Actor 输出；首步为 0 |

Actor 自带训练得到的 RunningNorm，部署端不再归一化。动作计算为：

```text
residual = clip(action * 0.5, -0.6, 0.6)
ema = 0.3 * residual + 0.7 * previous_ema
desired_target = current_reference_joint_pos + ema
```

不对实测观测添加训练噪声、延迟或随机动作冻结。实机编码器本身就是策略使用的编码器量，无需再减去仿真中的随机 encoder bias。世界模型、critic、FailureHead 和接触传感器不参与部署推理；`native` 接触仅作为 checkpoint 来源信息保存。

## 已完成验证

```bash
./scripts/python.sh -m pytest tests/dex_supermimic/test_z5_deploy.py -q
./scripts/python.sh -m ruff check deploy tests/dex_supermimic/test_z5_deploy.py
# 在已有 OpenCV contrib 的环境中运行合成相机测试。
python -m pytest tests/dex_supermimic/test_z5_deploy_vision.py -q
```

对指定 checkpoint 已完成 TorchScript/ONNX 导出一致性、运行时笔观测与训练几何函数的对照、EMA 与训练 ActionTerm 的逐步对照、DDS 顺序与符号、过期数据/温度处理和 mock 控制循环检查。11 项契约测试通过；另在已有 OpenCV 的环境中通过 1 项合成相机 PnP 测试。

最终导出包的 CPU MuJoCo 首段前 100 步（2 秒）通过默认跟踪门限，最大位置误差约 1.67 cm、笔轴偏角约 7.14°。完整回放在第 586 个控制步触发笔轴 60° 门限并退出。仅在无硬件诊断中使用 `--no-tracking-guard` 播完 970 步后，最大位置误差约 6.89 cm、笔轴偏角约 168°。因此目前确认的是部署数据链路及短程控制可运行，尚未验证 Newton 策略在 MuJoCo 中的长程稳定性。MuJoCo 与训练的 Newton 求解器不同，接触轨迹也对微小数值差异敏感。本次未连接相机或实机电机。
