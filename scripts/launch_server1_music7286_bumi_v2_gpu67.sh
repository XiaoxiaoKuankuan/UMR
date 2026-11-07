#!/usr/bin/env bash
# 本脚本用于在 Server-1 上把筛选后的 6,915 条音乐舞蹈 SMPL-X 动作重新重定向到最新 BUMI V2 模型。
# 输入：/home/user/music7286_umr/motions_keep 中的 30 Hz SMPL-X NPZ、新版 SuperMimic bumi2.xml，以及与该 XML 对应的已验证 correspondence slots。
# 输出：逐动作 30 Hz、28 维 qpos NPZ；output_link/bumi_v2 在部署时必须指向正式 raw_umr 目录，因此结果会直接写入替换后的数据集位置。
# 并发：仅使用物理 GPU 6、7；每卡 8 个常驻源特征进程，后接 144 个单线程常驻 CPU QP worker，并使用 288 条有界在途队列。
# 约束：脚本不删除任何数据；旧结果的清理必须由启动前的受控发布步骤完成。临时配置、源特征缓存、日志和 TMPDIR 全部位于 /data0，不写根分区。
# 使用：bash scripts/launch_server1_music7286_bumi_v2_gpu67.sh；建议由 tmux 会话 umr-bumi-v2-music7286-gpu67-20261107 托管。
set -euo pipefail

repo=/data0/user/liwei/UMR-throughput-opt-20261107
python_bin=/home/user/umr_venv/bin/python
run_root=/data0/user/liwei/umr_bumi_v2_music7286_latest_20261107
config="$repo/robot_configs/humanoid_retarget_bumi_v2_music7286_latest.json"
batch="$repo/configs/production/server1_music7286_bumi_v2_gpu67.json"
log="$run_root/logs/music7286_bumi_v2_gpu67.log"
status="$run_root/logs/music7286_bumi_v2_gpu67.status"
tmp_root="$run_root/runtime_tmp"

mkdir -p "$run_root/logs" "$run_root/source_features" "$run_root/tmp" "$tmp_root"
cd "$repo"

export TMPDIR="$tmp_root"
export MPLCONFIGDIR="$tmp_root/matplotlib"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
unset HUMANOID_BATCH_PROGRESS
export PYTHONUNBUFFERED=1
export UMR_STATIC_RETARGET_CACHE_MAX=8
export UMR_SURFACE_VECTOR_BATCH_FRAMES=128

printf '[RUN_START] %s host=%s xml_sha=caa309ddd569 gpu=6,7 workers=144 source_feeders_per_gpu=8 persistent_cpu=true\n' "$(date -Is)" "$(hostname)"
printf 'RUNNING %s\n' "$(date -Is)" > "$status"
set +e
"$python_bin" scripts/humanoid_retarget_pipeline_batch.py \
  --config "$config" \
  --batch-config "$batch" \
  > "$log" 2>&1
rc=$?
set -e
printf 'EXIT rc=%s %s\n' "$rc" "$(date -Is)" > "$status"
printf '[RUN_EXIT] %s rc=%s\n' "$(date -Is)" "$rc"
exit "$rc"
