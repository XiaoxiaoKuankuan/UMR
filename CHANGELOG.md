# UMR 代码变更日志

## 2026-11-07：Server-1 音乐舞蹈数据集 BUMI V2 新模型重定向

### 修改内容

- 新增 `robot_configs/humanoid_retarget_bumi_v2_music7286_latest.json`：使用 SuperMimic 最新 BUMI V2 `bumi2.xml`，关节限位完全由该 XML 提供；保留音乐舞蹈数据已验证的振幅限位、接地与时序平滑配置，并复用同一 XML 已验证的 correspondence 数值结果。为保持下游数据契约，专用副本仅把机器人槽位标识由 `bumi2` 改为 `bumi_v2`，数值数组不变。
- 新增 `configs/production/server1_music7286_bumi_v2_gpu67.json`：仅使用物理 GPU 6、7，启用 GPU 源特征预处理与 CPU QP 常驻进程解耦、固定中性 SMPL-X 模板、16 个常驻源特征进程、144 个单线程 CPU worker、有界在途队列与可恢复续跑。
- 新增 `scripts/launch_server1_music7286_bumi_v2_gpu67.sh`：将临时配置、特征缓存、日志和通用临时目录全部固定在 `/data0`，避免写入根分区；关闭进度条并记录任务状态。

### 数据契约与约束

- 输入固定为 `/home/user/music7286_umr/motions_keep` 中筛选后的 6,915 条音乐舞蹈 SMPL-X 动作；数据审计确认 6,915/6,915 均为 `gender=neutral`，且配置禁用个体 betas，因此允许复用固定人体模板。
- 输出保持 `robot_name=bumi_v2`、30 Hz、28 维 qpos，并直接写入现有 `raw_umr` 数据位置；启动前必须清除旧 XML 生成的 `raw_umr`、`npz50` 和打包库，不得复用旧关节限位结果。
- 该任务只生成离线运动学重定向结果，不代表物理闭环、强化学习训练或实机安全验证。

## 2026-11-07：大规模动作重定向吞吐优化

### 修改内容

- `scripts/humanoid_retarget_pipeline_batch.py`
  - 新增可选的常驻 CPU 重定向进程池，避免每条动作重复启动 Python、导入 MuJoCo/SMPL-X 及项目模块。
  - 新增 GPU 源特征 feeder 与 CPU 常驻 worker 的统一有界调度，限制在途缓存数量，并在失败时保留错误尾部和源特征缓存。
  - 新增每 100 条动作的吞吐、源预处理队列和 CPU 队列统计；生产模式不输出逐帧进度条。
  - 新增可选的既有临时配置复用，并支持按间隔汇总 `skip_existing` 日志，避免全量续跑时重复改写五十余万个 JSON 或输出数万行跳过记录。
  - 常驻模式改为按 feeder 需求惰性构建任务，不再在启动前一次性物化五十余万个命令字典；完成旧结果核验后即可立即进入 GPU/CPU 流水线。
  - 新增显式 `fixed_template_example_motion`：仅在调用方确认整个数据集共享人体模板时，复用一个样本的模板配置，避免逐个打开数十万个 NPZ 推断相同的性别和 betas。
- `scripts/retarget_smpl_to_humanoid_surface_vector_batch.py`
  - 将源表面槽位、三角面法向和相对面旋转改为跨帧向量化计算，并使用有界帧块控制临时内存。
  - 新增有界 LRU 静态缓存，按模板、槽位文件、配置和机器人资产身份复用源模板、源表面绑定、MuJoCo 模型及机器人槽位绑定。
  - 未知源格式保守关闭源模板复用；缓存键包含文件大小和纳秒修改时间，资产替换后不会误用旧上下文。
- `scripts/source_self_contact.py`
  - 将每帧候选自接触点对距离改为小批帧向量化；每帧的筛选、排序、身体分段配额及权重规则保持不变。
- `tests/test_source_feature_vectorization.py`
  - 新增固定随机种子的数值一致性和微基准测试，覆盖槽位、法向、面旋转及两类自接触映射。

### 验证记录

- 真实 599 帧动作的源特征缓存共 21 个字段，与旧实现逐字段一致，数值最大绝对差为 0；预处理耗时从 17.57 秒降至 15.83 秒。
- 31 帧真实动作在同一常驻 worker 内，首次预热约 5.06 秒，后续约 0.90 秒；三次输出与未缓存版本逐字段、逐元素一致。
- 8 条真实 MotionMillion 动作完成 GPU 预处理到 CPU 重定向的端到端测试，8/8 成功、0 个源缓存残留；与既有结果的最大数值绝对差为 `2.1457672119140625e-06`，所有非数值字段完全一致。
- 对 MotionMillion 全目录按固定种子抽样 2,048/559,924 个文件：全部为 `gender=neutral`、10 维 betas 全零、`source_format=motionmillion_272`，0 个读取失败；因此生产续跑显式启用固定模板样本，其他数据集默认不启用。
- 单 GPU、2 个 feeder、8 条中长动作的端到端耗时由 62.13 秒降至 53.61 秒；该数字受同时运行的生产任务影响，仅作为受控对比，不外推为全量吞吐。

### 使用约束

- 常驻模式通过批配置 `retarget.persistent_retarget_workers=true` 启用，且必须同时启用 `decoupled_source_preprocess`。
- 只有在机器人、求解器和数据契约均未改变的续跑任务中才可启用 `batch.reuse_temp_configs=true`；新契约必须重新生成逐动作配置。
- `batch.fixed_template_example_motion` 不能用于混合性别、混合 betas 或多骨架数据集；未经过数据审计时保持未设置。
- 临时配置、源特征缓存和输出目录必须显式放在数据盘；本次生产配置均使用 `/data1`。
- 切换全量任务时继续使用原机器人配置的绝对路径，以保持既有输出中的 `robot_xml` 元数据和 `skip_existing` 兼容性。
