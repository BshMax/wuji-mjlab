# A4 标签：记录、真值和基线对照

本轮仅测试现有 A4 标签纸在约 50 cm 处的二维识别，不代表实体笔转动或三维定位已经合格。默认相机与正式转笔算法不变；`baseline` 仍是默认检测版本。


## 默认常开与按键录制（最新入口）

不加 `--frames` 时窗口持续运行，初始仅预览，不写图像，不会 10 秒自动关闭：

```bash
cd /home/psy/unitree/wuji-mjlab
.pixi/envs/pen-spin-deploy/bin/python deploy/pen_spin/tools/tag_benchmark.py record outputs/tag_eval/session_01
```

点击窗口后：`s` 开始一个新片段；`p` 保存当前片段并暂停；再次 `s` 从新片段继续；`n` 保存并立即开始下一段；`q`/Esc 保存后退出。暂停不会被算作同一片段内的漏检。重新运行相同会话目录会继续编号，不覆盖旧片段。

`d` 开关实时检测，`m` 依次切换 baseline/corners/ecc1/tracked；`1` 正样本、`0` 负样本；`2` 笔贴页、`3` 腕部页；`+`/`-` 调整受支持相机的曝光。这些会改变采集条件或预览负载的操作会先保存并暂停当前片段，再按 `s` 开始新段。没有真值时，实时检测不能给出真实误检率。纯预测仍不算识别成功。

保存结构是 `session_01/clip_0001/`、`clip_0002/`……，每个片段独立完整，包含原图、帧元数据、校验和和当前检测日志。会话根目录的 `events.jsonl` 记录按键引起的开始、停止、条件改变、换源和错误。后续标注、评测、复查都选择具体的 `clip_0001`，不是会话根目录。

不想自动启用画图负载时保持 `d` 关闭，仅记录原图；离线再统一回放四版本。

## 换相机：统一接口和通用设备

相机适配器（Adapter，负责统一不同相机的取图结果）入口为 `lib/camera_sources.py`，支持：

- `realsense`：保留 D435/D435i 双红外，真实内参、硬件时间及同步属性。
- `opencv`：支持通用 UVC（USB Video Class，USB 视频设备标准）设备索引或设备路径，可单路，也可两个独立设备。两台普通 USB 相机不会被宣称为同步双目。
- `python`：专用厂商 SDK 可通过可信本地 `module:factory` 适配器接入；具体设备未确定前，不声称已经支持其专有协议。

通用入口：

```bash
.pixi/envs/pen-spin-deploy/bin/python deploy/pen_spin/tools/tag_benchmark.py record outputs/tag_eval/new_camera_01 \
  --camera-config deploy/pen_spin/config/camera_opencv.yaml
```

编辑 YAML 的 `camera.device`，例如 `0` 或 `/dev/v4l/by-id/...`。未知设备默认保留驱动原生分辨率，指定尺寸/帧率时会核对实际返回值。没有标定则保存内参为 `null`，可做基线二维检测和录制；实验版本需要匹配实际分辨率的标定。不会沿用 D435 的内参，不会复制单路图像冒充双路，不会假设全局快门或把主机接收时刻冒充硬件曝光时刻。

已知标定可在 `calibration.left`（以及可选 right）提供 `width`、`height`、`K`、`dist`、`model: opencv_pinhole`。尺寸不匹配会拒绝启动；鱼眼模型需要专门矫正适配，不能按普通针孔系数直接使用。

保持窗口运行时，可修改同一个 `--camera-config` 文件，然后按 `k`：保存当前片段、释放旧设备、重读配置并打开新设备，重置检测状态。相机不可用时可修复连接后按 `k` 重试，或 `q` 退出；原生 SDK 进程级崩溃仍需重新运行程序。

专用 SDK 插件 YAML 示例：`camera: {backend: python, adapter: my_camera_adapter:open_camera}`。工厂接收配置并返回上下文管理器，提供 `describe()`、`frames()`、`close()`；可选 `adjust_exposure(factor)`。`describe()` 声明真实 `views: [left]` 或 `[left, right]`、backend、pairing、每路 K/dist（未知为 null）；`frames()` 返回 `CameraFrame(images={...}, metadata={frame_number, host_receive_ns, ...})`。图像必须是原始 8 位灰度；同步、快门、转换方式和时间域必须如实声明。新型号到手后只需实现该适配层并用真实硬件验证。

## 1. 准备与一次性设置

先按 `q` 关闭所有相机预览，避免设备占用和 CPU 干扰。将现有 A4 平整固定在板上，距离镜头约 50 cm，整页同时处于左右视野。记录实际距离、光照和运动条件，测试间不要改变标签尺寸或打印缩放。

以下命令均在同一个终端执行：

```bash
cd /home/psy/unitree/wuji-mjlab
TAG_PY="$PWD/.pixi/envs/pen-spin-deploy/bin/python"
TAG_TOOL="$PWD/deploy/pen_spin/tools/tag_benchmark.py"
mkdir -p outputs/tag_eval
```

数据目录、评测输出目录均必须是新名字，不会覆盖已有记录。原图无损保存，数据量可能较大；先用 300 对图像跑通流程。录制队列不足时明确报错并标记不完整，不通过静默丢帧维持表面帧率。

## 2. 记录正样本

```bash
"$TAG_PY" "$TAG_TOOL" record outputs/tag_eval/positive_01 \
  --scene positive --sheet pen --distance-cm 50 --frames 300 --preview
```

默认使用左右 IR1/IR2 Y8、1280×800、30 FPS、5000 微秒曝光。300 对约为 10 秒。先做静止纸，再单独录制倾斜、慢移、重新入画等片段，用新目录区分，不要把不同条件混为一个结果。

手动按 `p` 或 `q` 会保存为完整片段；只有采集异常或写入失败会标成不完整。显式 `--frames N` 仍支持批量自动停止。

记录包含：

- `session.json`：唯一数据集编号、操作员填写的距离、相机内参、实际配置和代码版本。
- `frames.jsonl`：每对图像的文件路径、帧号、设备时间戳及时间域、主机接收时间、曝光、同步差和跳过的源帧数量。
- `left/`、`right/`：原始灰度 PNG，不含框、字或预测结果；每个文件带 SHA-256 校验，评测前检查原图是否改变。

距离是操作员报告值，软件不会假装自己测量了 50 cm。设备时钟与主机时钟未对齐时，不能把二者直接相减计算端到端延迟。

## 3. 独立审核正样本真值

```bash
"$TAG_PY" "$TAG_TOOL" annotate outputs/tag_eval/positive_01/clip_0001 \
  --output outputs/tag_eval/positive_01/clip_0001/truth.json --sheet pen --stride 30
```

首次可用 `--stride 30` 每秒抽一对图，先检查流程。**这只是抽样识别评测，不能据此宣布逐帧跟踪稳定。** 正式连续帧测试使用 `--stride 1`；可通过 `--start N` 从某一帧继续。

每帧先审核左图，再审核右图：

1. 按印刷页自身的左上、右上、右下、左下顺序点击纸张四角。标题所在边是页顶；纸旋转时也按印刷方向点击。
2. 软件按打印文件的解析几何投影出 108 个码的位置和编号，不使用检测器输出生成答案。腕部页用 `--sheet wrist`，共 12 个 ID 0。
3. 检查叠加边框是否准确贴合每个码。纸弯曲、四角选错、打印版本不同或缩放异常时，不要保存；先修正条件。角点误差是相对这份纸角投影真值计算，受人工标注精度影响，不能当成独立的毫米级定位精度。
4. 按 `s` 明确保存并确认这一侧。连续帧可能传播上次纸角作为草稿，仍必须逐帧确认；草稿不会自动变成真值。
5. `c` 清空四角，`n` 跳过，`q` 退出。右键码框可标记确实完全不可见的实例；不要把只是难检的可见码随意排除。

每个计分帧必须左右均审核。若退出时只审核了左侧，重新进入完成右侧后再评测。缺失标注绝不当成负样本。

## 4. 记录并确认负样本

把腕部和笔贴等支持的标签移出两侧视野，保留键盘、文字、屏幕普通内容、方形物体等真实背景。先用 10 秒片段跑通，正式测试增加时长和场景，报告实际数量，不把“短片段零误检”解释成绝对不会误检。

```bash
"$TAG_PY" "$TAG_TOOL" record outputs/tag_eval/negative_01 \
  --scene negative --frames 300 --preview
"$TAG_PY" "$TAG_TOOL" annotate outputs/tag_eval/negative_01/clip_0001 \
  --output outputs/tag_eval/negative_01/clip_0001/truth.json --negative-all
```

`--negative-all` 是操作员对整段无目标图案的明确确认，只允许用于标为负样本的录制。不要把屏幕上真实显示的支持标签图案误标成负样本。

## 5. 同一数据回放四个阶段

```bash
"$TAG_PY" "$TAG_TOOL" evaluate outputs/tag_eval/positive_01/clip_0001 \
  --truth outputs/tag_eval/positive_01/clip_0001/truth.json \
  --output outputs/tag_eval/positive_01/clip_0001/results_01
"$TAG_PY" "$TAG_TOOL" evaluate outputs/tag_eval/negative_01/clip_0001 \
  --truth outputs/tag_eval/negative_01/clip_0001/truth.json \
  --output outputs/tag_eval/negative_01/clip_0001/results_01
```

默认依次运行 `baseline corners ecc1 tracked`。即使显式只选择实验版本，也会自动加入基线。每个版本读取同一批原图和同一真值，各自重置状态。先用独立片段调参，再冻结参数，用没有参与调参的新片段验收；不能用相邻帧拆分来制造独立验证。

- `baseline`：原有检测器，原字典不纠错。
- `corners`：针对约 50 cm 小码，阈值窗口改为 3/13 像素；使用批量等价边框检查、受限小窗口角点精修，以及最多 12 个失败候选的当前帧重解码。
- `ecc1`：在上一步上仅允许一格纠错；要求边框、对比度、单格不确定性和竞争码间隔合格，新纠错实例需要相邻帧确认。
- `tracked`：增加前后向 LK 光流检查、原分辨率局部搜索和当前帧重解码。每三帧全图重搜；跟踪不可靠时立即全图重搜。纯位置预测最多保留两帧，明确标记，绝不计作识别成功。

这些是累积对照，不意味着每一项必然有收益。原实体笔 PnP、标签布局、模型和相机参数不会被实验自动改动。

## 6. 看哪些报告

每个输出目录包含：

- `comparison.txt`：简明对照表。
- `comparison.json`：逐版本相对基线的识别率、误检变化和延迟是否达到 33 ms。
- `baseline.json` 等：真阳性 TP、假阳性 FP、漏检 FN、错编号／错方向、负帧误检率、角点误差、身份切换、连续可见漏检段、重新识别间隔、持续误轨迹、P50/P95/P99/最大耗时及超时比例。
- `*.frames.jsonl`：逐帧左右角点、结果来源、纠错位数、确认状态、图像质量证据、局部搜索范围、重试和拒绝原因、两侧处理与关联耗时。FP/FN 可回查原图。

评测按真实位置和编号一对一匹配，重复编号是不同的物理实例。重复检测不能重复算成功；错编号和错方向会产生 FP 和 FN。

来源 `decoded_exact`、已确认的 `decoded_corrected`、以及从当前图像重新解码的 `tracked_redecoded` 可参与识别计分。`tracked_only` 和未确认纠错候选不参与 TP，但会分别统计预测位置匹配情况及错误候选，不能藏在成功率里。

“误检占比”是 FP/(TP+FP)；“负帧误检率”是负样本中出现至少一个接受误检的帧数/负样本总帧数。没有负样本时显示未测，不用零代替。左右结果分别报告，不能把相关的双目图像当成独立样本来夸大置信度。

处理计时覆盖双路检测、纠错、跟踪、算法校验；排除文件解码、真值评分和画图，三者分别说明。实际在线帧率还受采集、记录和画图影响。**双路算法处理 P95 ≤33 ms 才满足本轮预算**；30 Hz 设置本身不等于低延迟。首次确认与重新识别的等待会表现为逐帧 FN，不能通过不计启动帧来隐藏。

最长漏检段、重新识别间隔和持续误轨迹只在连续审核帧上计算；抽样标注不能得出这些连续性结论，相关指标会显示未测。身份切换另给出实际可比较的相邻匹配次数，不能将没有可比样本的零次数当成稳定。

默认前 10 对只从耗时统计中排除，仍参与准确率统计。非常短的测试可指定 `--warmup 0`。建议正式性能测试重复至少三次，保留所有结果，不能只挑最快一次。

## 7. 可视化复查错误帧

```bash
"$TAG_PY" "$TAG_TOOL" review outputs/tag_eval/positive_01/clip_0001 \
  --truth outputs/tag_eval/positive_01/clip_0001/truth.json \
  --results outputs/tag_eval/positive_01/clip_0001/results_01/ecc1.frames.jsonl --only-errors
```

蓝色是真值，绿色是通过计分的识别，红色是接受误检，橙色是纯预测或未确认候选。`n`/空格下一帧，`p` 上一帧，`l` 仅左侧，`r` 仅右侧，`b` 并排，`q` 退出。真值修改后必须重新评测，再复查。

## 8. 在线观察与完整日志

仅用于看画面和记录性能，实时窗口里的“数量缺失率”不能替代上面的误检评测：

```bash
"$TAG_PY" deploy/pen_spin/scripts/pen_world_observer.py \
  --source realsense --detect-only --stereo --preview --detect-hz 30 \
  --camera-config deploy/pen_spin/config/realsense_sheet_max.yaml \
  --detector-profile tracked --diagnostics outputs/tag_eval/live_tracked_01.jsonl
```

替换为 `baseline`、`corners` 或 `ecc1` 可分别观察。日志文件必须不存在。绿色只代表算法当前帧解码通过，不代表有真值证明正确；橙色表示未确认/预测，不进入成功统计。窗口会单独显示这两类数量。在线日志额外保存曝光、设备时间域、主机接收至结果耗时与预览耗时；没有时钟校准时不宣称曝光到屏幕的完整延迟。

## 9. 不占用相机的工具自检

```bash
"$TAG_PY" "$TAG_TOOL" synthetic outputs/tag_eval/synthetic_01 --frames 30 --full-sheet
"$TAG_PY" "$TAG_TOOL" evaluate outputs/tag_eval/synthetic_01 \
  --truth outputs/tag_eval/synthetic_01/truth.json \
  --output outputs/tag_eval/synthetic_01/results_01
```

该数据由已知几何生成并明确标为 `synthetic`，距离为未定义。包含 108 个小码、像素相位／模糊变化及负样本尾段，仅验证工具和压力性能，不能代替真实 50 cm 验收。当前实验尚未获得真实数据的合格结论，因此不会自动替换默认基线。
