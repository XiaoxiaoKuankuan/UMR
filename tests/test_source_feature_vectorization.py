#!/usr/bin/env python3
"""验证 UMR 源特征向量化优化的数值一致性并给出微基准。

用途：在提交或切换批量重定向任务前，对比旧逐帧公式与新跨帧批处理公式，覆盖
表面槽位、三角面法向、相对面旋转和两类自接触映射。输入由固定随机种子生成，
不依赖真实数据集，也不会写入转换结果；输出为各阶段最大误差与耗时。约束是测试
必须在 UMR 仓库根目录环境中运行，使脚本可导入项目的 MuJoCo/SMPL-X 模块。
使用方式：``python tests/test_source_feature_vectorization.py``。任何断言失败均以非零
退出码结束，禁止在失败时将优化版本用于生产转换。
"""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import retarget_smpl_to_humanoid_surface_vector_batch as optimized
from scripts.source_self_contact import compute_source_self_contact_map_groups


def reference_slots_and_normals(vertices, faces, binding):
    """逐帧复现优化前的槽位和法向实现。"""
    output = np.empty((len(vertices), len(binding["face_ids"]), 3), dtype=np.float32)
    normals = np.empty_like(output)
    bound_faces = np.asarray(faces, dtype=np.int32)[np.asarray(binding["face_ids"], dtype=np.int32)]
    for frame_index, frame_vertices in enumerate(vertices):
        output[frame_index] = optimized.common.dynamic_surface_template_to_world(
            frame_vertices,
            faces,
            binding,
        )
        triangles = frame_vertices[bound_faces]
        normals[frame_index] = optimized.normalize_vectors(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
        ).astype(np.float32)
    return output, normals


def reference_face_rotations(vertices, template_vertices, faces, binding, slot_ids):
    """逐帧复现优化前的面旋转实现。"""
    face_ids = np.asarray(binding["face_ids"], dtype=np.int32)[np.asarray(slot_ids, dtype=np.int32)]
    bound_faces = np.asarray(faces, dtype=np.int32)[face_ids]

    def frame_basis(frame_vertices):
        triangles = np.asarray(frame_vertices, dtype=np.float64)[bound_faces]
        edge = optimized.normalize_vectors(triangles[:, 1] - triangles[:, 0])
        normals = optimized.normalize_vectors(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
        )
        tangent = optimized.normalize_vectors(np.cross(normals, edge))
        return np.stack([edge, tangent, normals], axis=-1)

    template_basis_t = np.swapaxes(frame_basis(template_vertices), 1, 2)
    rotations = np.empty((len(vertices), len(face_ids), 3, 3), dtype=np.float32)
    for frame_index, frame_vertices in enumerate(vertices):
        rotations[frame_index] = (frame_basis(frame_vertices) @ template_basis_t).astype(np.float32)
    return rotations


def assert_contact_maps_equal(left, right):
    """逐模式、逐帧检查自接触映射的顺序、距离和权重。"""
    assert tuple(left) == tuple(right)
    for mode in left:
        assert len(left[mode]) == len(right[mode])
        for expected, actual in zip(left[mode], right[mode]):
            np.testing.assert_array_equal(actual["slot_pairs"], expected["slot_pairs"])
            np.testing.assert_array_equal(actual["distances"], expected["distances"])
            if "weights" in expected:
                np.testing.assert_array_equal(actual["weights"], expected["weights"])


def main():
    rng = np.random.default_rng(20261107)
    frame_count = 48
    vertex_count = 1024
    slot_count = 768
    face_count = 2048
    vertices = rng.normal(size=(frame_count, vertex_count, 3)).astype(np.float32)
    faces = rng.integers(0, vertex_count, size=(face_count, 3), dtype=np.int32)
    binding = {
        "face_ids": rng.integers(0, face_count, size=slot_count, dtype=np.int32),
        "bary": rng.dirichlet(np.ones(3), size=slot_count).astype(np.float32),
    }
    selected_slots = np.sort(rng.choice(slot_count, size=384, replace=False)).astype(np.int32)

    started = time.perf_counter()
    reference_slots, reference_normals = reference_slots_and_normals(vertices, faces, binding)
    reference_slot_seconds = time.perf_counter() - started
    started = time.perf_counter()
    vector_slots, vector_normals = optimized.source_slots_and_normals_for_vertices(vertices, faces, binding)
    vector_slot_seconds = time.perf_counter() - started
    np.testing.assert_array_equal(vector_slots, reference_slots)
    np.testing.assert_allclose(vector_normals, reference_normals, rtol=0.0, atol=1e-7)

    started = time.perf_counter()
    reference_rotations = reference_face_rotations(
        vertices,
        vertices[0],
        faces,
        binding,
        selected_slots,
    )
    reference_rotation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    vector_rotations = optimized.source_face_rotations_for_vertices(
        vertices,
        vertices[0],
        faces,
        binding,
        selected_slots,
    )
    vector_rotation_seconds = time.perf_counter() - started
    np.testing.assert_allclose(vector_rotations, reference_rotations, rtol=0.0, atol=1e-7)

    contact_slots = rng.normal(size=(24, 456, 3)).astype(np.float32)
    contact_ids = np.arange(456, dtype=np.int32)
    part_ids = np.repeat(np.arange(19, dtype=np.int32), 24)

    def non_adjacent(_part_ids, ids):
        row, col = np.triu_indices(len(ids), k=1)
        mask = np.zeros((len(ids), len(ids)), dtype=bool)
        mask[row, col] = _part_ids[ids[row]] != _part_ids[ids[col]]
        return mask

    contact_kwargs = {
        "part_name_to_id": {f"part_{index}": index for index in range(19)},
        "non_adjacent_pair_mask": non_adjacent,
        "modes": ("threshold_global_topk", "source_clearance_body_pair_topk"),
        "threshold": 1.0,
        "max_pairs": 256,
        "body_topk": {"default": 2},
        "log_prefix": "VectorizationTest",
    }
    started = time.perf_counter()
    framewise_maps = compute_source_self_contact_map_groups(
        contact_slots,
        contact_ids,
        part_ids,
        distance_batch_frames=1,
        **contact_kwargs,
    )
    framewise_contact_seconds = time.perf_counter() - started
    started = time.perf_counter()
    vector_maps = compute_source_self_contact_map_groups(
        contact_slots,
        contact_ids,
        part_ids,
        distance_batch_frames=8,
        **contact_kwargs,
    )
    vector_contact_seconds = time.perf_counter() - started
    assert_contact_maps_equal(framewise_maps, vector_maps)

    print(
        "[PASS] 槽位/法向/面旋转/自接触映射与逐帧语义一致；"
        f"槽位 {reference_slot_seconds:.4f}s -> {vector_slot_seconds:.4f}s "
        f"({reference_slot_seconds / vector_slot_seconds:.2f}x)，"
        f"面旋转 {reference_rotation_seconds:.4f}s -> {vector_rotation_seconds:.4f}s "
        f"({reference_rotation_seconds / vector_rotation_seconds:.2f}x)，"
        f"自接触距离批处理 {framewise_contact_seconds:.4f}s -> {vector_contact_seconds:.4f}s "
        f"({framewise_contact_seconds / vector_contact_seconds:.2f}x)。"
    )


if __name__ == "__main__":
    main()
