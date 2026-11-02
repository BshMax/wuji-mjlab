# 转笔 · 部署

[English](deploy.md) · [任务首页](../README_zh.md) · [硬件搭建](setup_zh.md) · [训练](train_zh.md) · [部署](deploy_zh.md)

所有命令均在**仓库根目录**运行。硬件与打印资源见 [硬件搭建](setup_zh.md)，训练与导出见 [训练](train_zh.md)。

**启动顺序：先开相机观测，确认笔的位姿有效，再开策略。`pen-policy` 默认同时打开实时镜像窗口。** 具体命令见[启动相机与策略](#启动相机与策略)。

## 策略模型

| 分组 | 动作 |
| --- | --- |
| `01_single_axis` | 单轴旋转 |
| `02_continuous_single_axis` | 连续单轴旋转 |
| `03_continuous_multi_axis` | 连续多轴旋转 |
| `04_stable_grasp_rotation` | 稳定抓握旋转 |
| `05_multi_axis_two_rotations` | 多轴两次旋转 |

[v2026.10.22 合并资源包](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.10.22/wuji-mjlab-v2026.10.22-assets.zip) 提供转笔硬件、轨迹数据及 5 组配套模型，位于包内 `pen_spin/checkpoints/<动作组>/`。将包内 `pen_spin/checkpoints/` 复制到仓库的 `src/wuji_mjlab/tasks/pen_spin/data/checkpoints/`，保留动作组目录及下方所有配套文件；下载数据集与训练步骤见 [训练](train_zh.md)。

```text
src/wuji_mjlab/tasks/pen_spin/data/checkpoints/<group>/
├── model.pt
├── policy.onnx
└── config.json
```

发布包每组提供 `model.pt`、`policy.onnx` 和 `config.json`，请成套保留。部署使用 `policy.onnx` 与同组参考轨迹。自行训练时，另外保留原始训练目录中的 `params/`，用于复现训练配置。

部署使用仓库提供的 [`pen_spin_policy.json`](../../../../../deploy/pen_spin/config/pen_spin_policy.json)，控制频率为 50 Hz。导出的 `config.json` 随模型保留，运行时不会自动读取该文件。

## 硬件与环境

准备 Wuji Hand 2、250 mm 标签笔、腕部 AprilTag、Hikrobot 相机和厂商 MVS SDK。标签笔对应 178 mm 裸露笔杆、14.9 mm 杆直径。部署环境独立安装：

```bash
pixi install -e pen-spin-deploy
```

核对[相机标定](../../../../../deploy/pen_spin/config/camera.yaml)、[笔标签布局](../../../../../deploy/pen_spin/config/pen_tags.json)和[控制配置](../../../../../deploy/pen_spin/config/control.yaml)。相机内参应对应实际设备；MVS SDK 安装和装置检查见[硬件搭建说明](setup_zh.md)。

## 相机位置与观测质量

**转笔效果非常依赖相机观测**。部署前先找好相机位置，使笔位姿观测稳定、连续，尽量减少遮挡和运动模糊。装配实物图和相机观测参考图见[转笔硬件文档](setup_zh.md#5-相机摆位与光照)。

1. 将笔的两端标签、完整运动范围和腕部基准标签放入视野，避免手指、支架长期遮挡；调整焦距、曝光和补光，让运动中的标签仍然清晰。
2. 尽量采用能看到多个标签面的斜视角。出现 `1 faces / COPLANAR` 时表示当前为单面共面观测，可进一步调整角度以改善位姿稳定性。
3. 确认检测线框与实物贴合，笔轴、头尾和位置一致；检查整个动作过程中是否出现跳变、翻转或持续丢帧。
4. 固定世界坐标后保持相机和手座不动；移动后按预览窗口的 `w` 重新采样腕部基准，并再次检查对齐。

## 启动相机与策略

默认部署只需 **两个终端**，都从仓库根目录运行。`pen-policy` 会同时启动**真机策略控制和实时 MuJoCo 镜像窗口**，显示实际关节、相机观测到的笔和参考动作。这里的镜像窗口用于观察真机，并不意味着只在仿真中运行。

### 1. 先启动相机观测，并保持运行

```bash
# 终端 1：先运行，部署期间不要关闭
pixi run -e pen-spin-deploy pen-observer --preview
```

等待腕部标签采样完成、世界坐标固定，并确认笔的线框位置正确、位姿持续更新，再启动策略。仅打开相机窗口不代表观测已经就绪。相机未就绪时，策略会在连接手之前退出；应先恢复相机观测，再重新运行策略命令。

### D435 / D435i：使用左红外全局快门图像

D435/D435i 的左右红外相机是全局快门，彩色相机是卷帘快门。彩色图转灰度不能改变快门类型。使用官方 `pyrealsense2` 接口直接读取左红外 `IR1 / Y8` 灰度流，无需开启深度图或彩色图。

```bash
# 替代上面的海康相机命令；不要同时启动两个观测器
pixi run -e pen-spin-deploy pen-observer-realsense --preview
# 多台相机时增加：--camera-serial <序列号>
# 曝光只对本次运行生效：--exposure-us 500
```

独立配置为 `deploy/pen_spin/config/realsense.yaml`，默认 848×480、90 FPS、手动曝光 1000 微秒、增益 16、点阵投射器关闭。SDK 从实际启用的图像流读取出厂内参和畸变；不使用原海康 `camera.yaml` 的参数。初始无需棋盘格内参标定；应先以已知尺寸和距离检查尺度。更改分辨率后重选检测区域；1280×720 可用于高分辨率对照，本机该模式最高 30 FPS，不能只改分辨率而保留 90 FPS。不支持的模式会明确报错。

更换相机或移动机位后，重新采样腕部标签；相机和手座在运行中必须保持固定。若改变腕部标签的安装位置，现有 `tag_frame.py` 中“标签到手腕”的变换也需要重新确定，不能只按 `w`。腕部黑框实际边长必须为 50.4 mm。

现有笔上使用自定义 18 码字典，不能直接替换为同编号标准 AprilTag。单码约 13.5 mm，在 848×480、距离 30 cm 时正视宽度约 19 像素，倾斜后更少。建议先在约 25–35 cm 范围评估全运动范围；使用均匀补光支持短曝光，避免反光、模糊和遮挡。全局快门仍会有长曝光运动模糊。

#### 整张 A4 纸检测与二维跟踪

测试打印纸时使用专用模式，不需要先安装到腕部和笔上：

```bash
pixi run -e pen-spin-deploy pen-tag-test --exposure-us 5000
# 或者给观测器增加：--source realsense --detect-only --preview
```

曝光调节写入红外传感器的曝光时间，检测器也使用调节后的图像，不是显示亮度滤镜。长曝光能增加收光量，但运动时更容易模糊。

需要每秒检测约 50 次时运行 `pixi run -e pen-spin-deploy pen-tag-test-50`：独立配置 `realsense_sheet_50.yaml` 使用 848×480、60 FPS 采集，并通过 `--detect-hz 50` 限制检测频率。相机没有原生 50 FPS 档位；每次取最新帧，不通过重复帧凑数。处理负载较高时实际检测率会低于目标。

完整枚举本机左红外 Y8 流后，最高分辨率是 1280×800、30 FPS；1280×720 也最高 30 FPS。需要至少 50 FPS 的采集时最高为 848×480。这里的最高分辨率指红外流，不是深度或彩色流。

当前高速配置为 848×480、90 FPS。远距离整页检测可选独立高清配置：

```bash
pixi run -e pen-spin-deploy pen-tag-test --camera-config deploy/pen_spin/config/realsense_sheet.yaml
```

高清配置为 1280×720、30 FPS、5000 微秒曝光。相同距离下单码边长像素数约为原来的 1.5 倍，代价是更低采集频率；优先用于静止或缓慢移动纸张。提高分辨率仍有距离限制：约 13.5 mm 的笔码在距离 50 cm 时，848 模式约 11 像素宽、1280 模式约 17 像素宽，倾斜后更少。将纸靠近、尽量正对镜头并均匀补光；标签边框不连续通常表示当前帧检测失败，轨迹号变化则还可能来自重复 ID 的关联变化。

此模式直接检测全画面，忽略但不覆盖保存的笔区域，不等待腕部标定，不运行整笔位姿求解，也不发布位姿。能同时检测 36h11 字典的所有标准编号和自定义笔码 0–17；同编号的多个纸面实例分别画框，标注 `APR36h11:0 T1` 或 `PEN:0 T2`。`T` 是本次会话内按位置关联的轨迹号，不是标签印刷编号。移动纸张时显示短轨迹；快速交叉、长时间遮挡或大幅跳动可能导致轨迹换号，不能据此当成独立三维定位。

先将整页置于相机视野内，确认 `APR36h11` / `PEN` 检测数量随展示纸张变化。腕部页为 12 个同编号 0 标签；笔页为 108 个图案、18 种编号。显示数目取决于画面清晰度、像素大小和遮挡，不能保证实拍每帧全部识别。

`q` 或 Esc 退出，`t` 开关轨迹，`r` 重置轨迹号。点击窗口后按 `+`/`=` 增加曝光、`-` 减少曝光；调整只作用于本次运行，不改 YAML。5000 微秒适合先做静态纸张测试；快速移动时应缩短曝光并补光。`--diagnostics <新文件.jsonl>` 可记录每个标签的角点、像素大小和轨迹号，使用同一统计工具查看检测数量及耗时。

原转笔观测器假设码按实体笔的三维布局安装，不能用整页纸验证整笔位姿；纸张误触发整笔一致性求解可能明显降低帧率。需要测试整笔定位时退出本模式，再使用常规观测器。

#### 仅检查相机和观测，不驱动机械手

```bash
# 自动采集 300 帧后退出，输出实际采集频率、曝光及处理耗时
pixi run -e pen-spin-deploy pen-observer-realsense --max-frames 300

# 保存逐帧诊断；文件必须尚不存在，避免覆盖已有记录
pixi run -e pen-spin-deploy pen-observer-realsense --preview --diagnostics /tmp/pen-static.jsonl
# 保持笔静止采样，按 q 退出后统计
pixi run -e pen-spin-deploy python deploy/pen_spin/tools/summarize_observer.py /tmp/pen-static.jsonl --static
```

动态测试另存新文件，统计时去掉 `--static`。诊断会报告有效位姿频率、观察间隔、重投影误差、处理耗时分位数；静态模式增加位置与朝向波动。静态波动代表重复性，不等于绝对定位精度。时间统计不代表曝光到控制的完整延迟。

启动腕部采样期间没有笔位姿属于预期行为；校准后持续丢失需排查可见标签数量、像素大小和光照。接收端会保留最后一次有效位姿，需关注 `obj_age`，不能只看窗口里是否有笔。必须确认有效观测后再启动下方策略；`--no-world-tag` 只供相机坐标系台架检查，不可用于真机策略。

### 2. 再启动策略与实时镜像

先准备同组的 ONNX 和 NPZ；下面以第三组为例，路径必须指向已有文件。

```bash
# 终端 2：真机控制 + 实时镜像，默认打开窗口
pixi run -e pen-spin-deploy pen-policy \
  --policy src/wuji_mjlab/tasks/pen_spin/data/checkpoints/03_continuous_multi_axis/policy.onnx \
  --motion src/wuji_mjlab/tasks/pen_spin/data/rawdata/clips/03_continuous_multi_axis/clip_001.npz
```

**默认不用另开 `pen-sim`，也不用改成 `python -m runtime --sim`。** `pen-policy` 已默认启用镜像；如相机已运行，只执行第二步。同一只手一次只运行一个策略进程，切换动作组前先退出旧策略，相机进程可以保持运行。

部署其他组时，同时更换 ONNX 和 NPZ 路径。`--motion` 指向随包提供的一段 `.npz` 轨迹；多手场景通过 `--hand-sn "<序列号>"` 指定设备。

`pen-sim` 是可选的独立只读观察器，可用于启动策略前检查观测；它不是默认部署流程的必需进程。

**控制进程会驱动真机，启动前清空手指运动范围**。程序缓慢到达起始手型后进入 `FROZEN`：

| 操作 | 效果 |
|---|---|
| 第一次 Enter | 检查笔的起始位姿，进入 `HOLD`，保持第一帧 |
| 第二次 Enter | 进入 `POLICY`，开始沿参考轨迹执行 |
| `r` | 复位并张开手 |
| Ctrl+C | 退出进程；不能替代硬件急停 |

`--hold-last-frame` 可让动作结束后保持最后一帧。更多运行参数见下文。

## 策略约定与运行选项

直接使用 release 提供的 `.npz` 参考轨迹，并选择同一动作组的策略。运行时的观测与动作配置见 [`pen_spin_policy.json`](../../../../../deploy/pen_spin/config/pen_spin_policy.json)。

开始前会拒绝“笔未被看见、位姿过旧、位置偏差超过 50 mm、笔轴偏差超过
60 度或头尾放反”的情况。

`--hold-last-frame` 让参考停在最后一帧（参考速度为零），直到按 `r` 或 Ctrl+C。
`--open-loop` 使用轨迹中的关节状态作为策略推理输入。
`--pen-axis-offset-m d` 让实时笔观测沿笔轴向 A_top 平移 `d`。

## 笔标签布局

`deploy/pen_spin/config/pen_tags.json` 已包含当前 250 mm 笔（露出笔杆 178 mm）的完整布局，直接使用即可。以下命令检查布局能否加载，并输出标签数量和笔总长：

```bash
pixi run -e pen-spin-deploy python - <<'PY'
from lib.pen_pose import PenTagLayout
layout = PenTagLayout("deploy/pen_spin/config/pen_tags.json")
print(f"tags={len(layout)}, length_mm={layout.total_length * 1000:.1f}")
PY
```

预期输出 `tags=18, length_mm=250.0`。实物尺寸及贴图必须与布局一致。

#### 左右红外并排测试与漏检统计

```bash
pixi run -e pen-spin-deploy pen-tag-test-stereo
```

该命令默认使用两路 **1280×800 Y8、30 FPS** 同步采集，目标最多每秒处理 30 对图像，每路 1,024,000 像素。需要对照原高速模式时运行 `pixi run -e pen-spin-deploy pen-tag-test-stereo-50`，其设置为 848×480、60 FPS 采集、最多每秒检测 50 对。屏幕左侧是 IR1，右侧是 IR2；每侧独立检测标准 36h11 和自定义码、绘制编号及轨迹。读取两路各自内参和出厂相机间变换；只接收同帧号、同时间域且时间差不超过 1 ms 的配对，异常帧不拿来比较。曝光和增益是左右红外共用的传感器设置，`+`/`-` 同时作用于两路。

默认以笔贴页为参考，每侧应看到 108 个码（编号 0–17 各 6 个）。点击窗口，按 `1` 改为腕部页参考（12 个 36h11 ID 0），按 `2` 回到笔贴页，`0` 为未知数量，只看检测，不计算漏检率。确保整张纸同时完整出现在左右画面中，再按 `s` 清零并开始累计；`p` 暂停；切换参考页会清零并暂停。`r` 清零统计和轨迹，保留当前累计/暂停状态；`t` 开关轨迹，`i` 开关详细轨迹号，`q`/Esc 退出。

每侧分别显示当前缺失数量和累计漏检率。累计分母为“开始统计后的图像对数量 × 参考标签数量”，每个编号独立计数，额外编号或某个编号的多余检测不能填补其他编号的缺失。纸未放好、出视野、遮挡和误选参考页都会计入缺失，因此这是指定测试条件下的整页数量缺失率，不是无条件的算法真实召回率。

重复编号的空间关联只在各自画面内进行，L/R 轨迹号不能直接视为左右对应的同一实体。本模式不做双目实例匹配或三维融合。JSONL 中的 `joint_miss_bounds` 只是由每个编号左右数量推导的共同缺失数量上下界，不能当作精确互补检测率。`skipped` 是因最新帧策略和处理节奏而未处理的采集帧数，不是标签漏检数。

指定 `--diagnostics <新文件.jsonl>` 可以保存每对图像的左右角点、轨迹、同步时间差、相机曝光及统计；用 `summarize_observer.py` 查看每段统计。无界面批量测试可直接运行观测器并加 `--stereo --detect-only --detect-hz 50 --measure --max-frames 300`，同时指定上面的 50 Hz 配置；`--measure` 表示从第一对图像立即计数，纸必须提前放好。

所选 30 Hz 或 50 Hz 是处理上限。两侧各有上百个标签时检测与绘图耗时会增加，应以窗口中的实际频率和终端耗时为准；不以重复处理旧图像来补足帧率。

高清模式相同距离的标签像素边长约为 848×480 模式的 1.51 倍（本机像素焦距约由 423 增至 638）。这不等于保证识别距离增加 51%，也不等于定位精度同比提高。1280×800 比 1280×720 多出垂直视野，两者像素焦距相同。对照时保持纸张尺寸、角度、照明与曝光相同，每个距离分别测量 10 秒；纸张必须完整处于两侧视野，再按 `s` 清零开始统计。每次切换模式后读取新的实际内参，原笔区域设置保持原样。
