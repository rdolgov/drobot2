"""Exact one-direction 180 mm rise x 250 mm tread terrain."""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils.configclass import configclass


@configclass
class ExactStairsTerrainCfg(SubTerrainBaseCfg):
    """Four fixed stairs preceded by an approach and followed by a platform."""

    function = MISSING
    rise_m: float = 0.18
    tread_depth_m: float = 0.25
    step_count: int = 4
    stair_start_x_m: float = 1.00
    approach_origin_x_m: float = 0.55
    platform_depth_m: float = 0.75


def exact_stairs_terrain(
    difficulty: float, cfg: ExactStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Build solid stair boxes; difficulty is intentionally ignored."""

    del difficulty
    width = float(cfg.size[1])
    meshes: list[trimesh.Trimesh] = []

    ground = trimesh.creation.box(
        (float(cfg.size[0]), width, 0.10),
        trimesh.transformations.translation_matrix(
            (float(cfg.size[0]) / 2.0, width / 2.0, -0.05)
        ),
    )
    meshes.append(ground)

    for index in range(int(cfg.step_count)):
        height = (index + 1) * float(cfg.rise_m)
        center_x = float(cfg.stair_start_x_m) + (index + 0.5) * float(
            cfg.tread_depth_m
        )
        meshes.append(
            trimesh.creation.box(
                (float(cfg.tread_depth_m), width, height),
                trimesh.transformations.translation_matrix(
                    (center_x, width / 2.0, height / 2.0)
                ),
            )
        )

    top_height = int(cfg.step_count) * float(cfg.rise_m)
    top_start = float(cfg.stair_start_x_m) + int(cfg.step_count) * float(
        cfg.tread_depth_m
    )
    meshes.append(
        trimesh.creation.box(
            (float(cfg.platform_depth_m), width, top_height),
            trimesh.transformations.translation_matrix(
                (
                    top_start + float(cfg.platform_depth_m) / 2.0,
                    width / 2.0,
                    top_height / 2.0,
                )
            ),
        )
    )

    origin = np.array(
        [float(cfg.approach_origin_x_m), width / 2.0, 0.0], dtype=np.float64
    )
    return meshes, origin


ExactStairsTerrainCfg.function = exact_stairs_terrain
