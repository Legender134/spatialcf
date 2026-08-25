"""Native support evidence owners for the staged AI2-THOR adapter split."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from hashlib import sha256
from typing import Any

from spatialcf.adapters.ai2thor.models import (
    _RECEPTACLE_TRIGGER_GRID_QUANTIZATION_M,
    _RECEPTACLE_TRIGGER_GRID_SIDE,
    _RECEPTACLE_TRIGGER_GRID_SIZE,
    AI2ThorFloorEnvelope,
    AI2ThorNativeFeasibilityMap,
    AI2ThorNativePosition,
    AI2ThorNativeReturnError,
    AI2ThorNativeSupportFact,
    AI2ThorNativeSupportKind,
    AI2ThorNavigationFeasibilityMap,
    AI2ThorReceptacleSpawnMap,
    AI2ThorReceptacleSurfacePatch,
    AI2ThorRuntimeIdentity,
    _canonical_json_sha256,
    _native_positions_sha256,
    _receptacle_spawn_source_sha256,
    adapter_spawn_map_from_native,
)
from spatialcf.adapters.base import (
    AdapterOperationError,
    AdapterSpawnMap,
    SourceCaptureFacts,
)
from spatialcf.domain.scene import (
    OBB,
    CollisionObstacle,
    Quaternion,
    Scene,
    SubjectPositionRegion,
    Vec2,
    Vec3,
)
from spatialcf.domain.serialization import canonical_json_bytes
from spatialcf.geometry.regions import (
    conservative_navigation_position_geometry,
    conservative_receptacle_position_geometry,
    planar_polygon_payloads,
)
from spatialcf.geometry.transforms import ai2thor_position_to_world

_REACHABLE_POSITION_QUANTIZATION_M = 1e-6
_STRUCTURAL_OBJECT_TYPES = frozenset({"Ceiling", "Floor", "Wall"})


def canonicalize_ai2thor_reachable_positions(
    positions: tuple[AI2ThorNativePosition, ...],
) -> tuple[AI2ThorNativePosition, ...]:
    """Return stable 1-micrometre identities for one native navigation grid."""

    if (
        type(positions) is not tuple
        or not positions
        or any(type(item) is not AI2ThorNativePosition for item in positions)
    ):
        raise TypeError("reachable positions must be a non-empty exact tuple")
    keyed: list[tuple[tuple[int, int, int], AI2ThorNativePosition]] = []
    seen: set[tuple[int, int, int]] = set()
    for position in positions:
        key = tuple(
            round(value / _REACHABLE_POSITION_QUANTIZATION_M)
            for value in (position.x, position.z, position.y)
        )
        if key in seen:
            raise ValueError("reachable positions contain a duplicate canonical point")
        seen.add(key)
        keyed.append(
            (
                key,
                AI2ThorNativePosition(
                    x=round(key[0] * _REACHABLE_POSITION_QUANTIZATION_M, 6),
                    y=round(key[2] * _REACHABLE_POSITION_QUANTIZATION_M, 6),
                    z=round(key[1] * _REACHABLE_POSITION_QUANTIZATION_M, 6),
                ),
            )
        )
    keyed.sort(key=lambda item: item[0])
    return tuple(position for _, position in keyed)


def bind_ai2thor_reachable_positions(
    reference_positions: tuple[AI2ThorNativePosition, ...],
    observed_positions: tuple[AI2ThorNativePosition, ...],
) -> tuple[AI2ThorNativePosition, ...]:
    """Bind one noisy replay to a frozen native navigation-grid roster."""

    frozen = canonicalize_ai2thor_reachable_positions(reference_positions)
    if (
        type(observed_positions) is not tuple
        or not observed_positions
        or any(type(item) is not AI2ThorNativePosition for item in observed_positions)
    ):
        raise TypeError("observed reachable positions must be a non-empty exact tuple")
    if len(observed_positions) != len(reference_positions):
        raise ValueError("reachable position roster changed")
    maximum_coordinate_ulp_m = max(
        math.ulp(value)
        for position in (*reference_positions, *observed_positions)
        for value in (position.x, position.y, position.z)
    )
    maximum_tolerance_m = _REACHABLE_POSITION_QUANTIZATION_M + 4.0 * max(
        maximum_coordinate_ulp_m,
        math.ulp(_REACHABLE_POSITION_QUANTIZATION_M),
    )
    bucket_span = math.ceil(maximum_tolerance_m / _REACHABLE_POSITION_QUANTIZATION_M)

    def bucket(position: AI2ThorNativePosition) -> tuple[int, int, int]:
        return tuple(
            math.floor(value / _REACHABLE_POSITION_QUANTIZATION_M)
            for value in (position.x, position.y, position.z)
        )

    reference_buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, reference in enumerate(reference_positions):
        reference_buckets.setdefault(bucket(reference), []).append(index)

    matched: set[int] = set()
    for observed in observed_positions:
        center = bucket(observed)
        candidates: list[int] = []
        if bucket_span <= 8:
            for x_offset in range(-bucket_span, bucket_span + 1):
                for y_offset in range(-bucket_span, bucket_span + 1):
                    for z_offset in range(-bucket_span, bucket_span + 1):
                        candidates.extend(
                            reference_buckets.get(
                                (
                                    center[0] + x_offset,
                                    center[1] + y_offset,
                                    center[2] + z_offset,
                                ),
                                (),
                            )
                        )
        else:
            candidates.extend(range(len(reference_positions)))
        within_tolerance = tuple(
            index
            for index in candidates
            if math.dist(
                (
                    reference_positions[index].x,
                    reference_positions[index].y,
                    reference_positions[index].z,
                ),
                (observed.x, observed.y, observed.z),
            )
            <= _REACHABLE_POSITION_QUANTIZATION_M
            + 4.0
            * max(
                *(
                    math.ulp(value)
                    for value in (
                        reference_positions[index].x,
                        reference_positions[index].y,
                        reference_positions[index].z,
                        observed.x,
                        observed.y,
                        observed.z,
                    )
                ),
                math.ulp(_REACHABLE_POSITION_QUANTIZATION_M),
            )
        )
        if len(within_tolerance) != 1 or within_tolerance[0] in matched:
            raise ValueError("reachable position roster changed")
        matched.add(within_tolerance[0])
    if len(matched) != len(reference_positions):
        raise ValueError("reachable position roster changed")
    return frozen


def _grid_axis(
    values: tuple[float, ...],
) -> tuple[tuple[int, ...], dict[int, tuple[float, ...]]] | None:
    grouped: dict[int, list[float]] = {}
    for value in values:
        key = round(value / _RECEPTACLE_TRIGGER_GRID_QUANTIZATION_M)
        grouped.setdefault(key, []).append(value)
    keys = tuple(sorted(grouped))
    if len(keys) != _RECEPTACLE_TRIGGER_GRID_SIDE:
        return None
    actual_min = min(values)
    actual_max = max(values)
    if actual_min >= actual_max:
        return None
    tolerance = 2.0 * _RECEPTACLE_TRIGGER_GRID_QUANTIZATION_M
    for index, key in enumerate(keys):
        expected = actual_min + (actual_max - actual_min) * index / 20.0
        if any(abs(value - expected) > tolerance for value in grouped[key]):
            return None
    return keys, {key: tuple(grouped[key]) for key in keys}


def build_ai2thor_receptacle_surface_patches(
    raw_positions: tuple[AI2ThorNativePosition, ...],
) -> tuple[AI2ThorReceptacleSurfacePatch, ...]:
    """Fail closed unless every raw contiguous block is one complete grid."""

    if (
        type(raw_positions) is not tuple
        or not raw_positions
        or len(raw_positions) % _RECEPTACLE_TRIGGER_GRID_SIZE
        or any(type(item) is not AI2ThorNativePosition for item in raw_positions)
    ):
        return ()
    patches: list[AI2ThorReceptacleSurfacePatch] = []
    for start in range(0, len(raw_positions), _RECEPTACLE_TRIGGER_GRID_SIZE):
        block = raw_positions[start : start + _RECEPTACLE_TRIGGER_GRID_SIZE]
        y_keys = {
            round(item.y / _RECEPTACLE_TRIGGER_GRID_QUANTIZATION_M) for item in block
        }
        x_axis = _grid_axis(tuple(item.x for item in block))
        z_axis = _grid_axis(tuple(item.z for item in block))
        if len(y_keys) != 1 or x_axis is None or z_axis is None:
            return ()
        x_keys, _ = x_axis
        z_keys, _ = z_axis
        cells = tuple(
            (
                round(item.x / _RECEPTACLE_TRIGGER_GRID_QUANTIZATION_M),
                round(item.z / _RECEPTACLE_TRIGGER_GRID_QUANTIZATION_M),
            )
            for item in block
        )
        expected_cells = {(x_key, z_key) for x_key in x_keys for z_key in z_keys}
        if (
            len(set(cells)) != _RECEPTACLE_TRIGGER_GRID_SIZE
            or set(cells) != expected_cells
        ):
            return ()
        patches.append(
            AI2ThorReceptacleSurfacePatch(
                x_min=min(item.x for item in block),
                x_max=max(item.x for item in block),
                native_y=(min(item.y for item in block) + max(item.y for item in block))
                / 2.0,
                z_min=min(item.z for item in block),
                z_max=max(item.z for item in block),
            )
        )
    return tuple(
        sorted(
            patches,
            key=lambda item: (
                item.native_y,
                item.x_min,
                item.z_min,
                item.x_max,
                item.z_max,
            ),
        )
    )


def _canonical_scene_sha256(scene: Scene) -> str:
    """Hash source facts independently of unordered roster presentation."""

    if type(scene) is not Scene:
        raise TypeError("canonical scene digest requires an exact Scene")
    normalized = scene.model_copy(
        update={
            "cameras": tuple(sorted(scene.cameras, key=lambda item: item.camera_id)),
            "objects": tuple(sorted(scene.objects, key=lambda item: item.object_id)),
            "collision_obstacles": tuple(
                sorted(scene.collision_obstacles, key=lambda item: item.obstacle_id)
            ),
            "subject_position_regions": tuple(
                sorted(
                    scene.subject_position_regions,
                    key=lambda item: item.region_id,
                )
            ),
        }
    )
    payload = normalized.model_dump(mode="python", warnings="error")
    payload["pinned_object_ids"] = tuple(sorted(scene.pinned_object_ids))
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _receptacle_scene_sha256(
    scene: Scene,
    surface_patches: tuple[AI2ThorReceptacleSurfacePatch, ...],
) -> str:
    """Preserve the legacy digest unless the new grid contract is active."""

    if surface_patches:
        return _canonical_scene_sha256(scene)
    return _canonical_json_sha256(scene.model_dump(mode="json", warnings="error"))


def _strict_native_position(
    value: object,
    label: str,
) -> AI2ThorNativePosition:
    if type(value) is not AI2ThorNativePosition:
        raise TypeError(f"{label} must be an exact AI2ThorNativePosition")
    return AI2ThorNativePosition(x=value.x, y=value.y, z=value.z)


def _strict_receptacle_spawn_map(
    value: object,
) -> AI2ThorReceptacleSpawnMap:
    if type(value) is not AI2ThorReceptacleSpawnMap:
        raise TypeError("spawn_map must be an exact AI2ThorReceptacleSpawnMap")
    if type(value.positions) is not tuple:
        raise TypeError("spawn map positions must be an exact tuple")
    positions = tuple(
        _strict_native_position(item, "spawn map position") for item in value.positions
    )
    if type(value.runtime_identity) is not AI2ThorRuntimeIdentity:
        raise TypeError("spawn map runtime identity has invalid type")
    return AI2ThorReceptacleSpawnMap(
        scene_id=value.scene_id,
        subject_object_id=value.subject_object_id,
        support_object_id=value.support_object_id,
        native_subject_object_id=value.native_subject_object_id,
        native_support_object_id=value.native_support_object_id,
        runtime_identity=AI2ThorRuntimeIdentity(**asdict(value.runtime_identity)),
        positions=positions,
        positions_sha256=value.positions_sha256,
        scene_sha256=value.scene_sha256,
        source_sha256=value.source_sha256,
        surface_patches=value.surface_patches,
    )


def capture_bound_ai2thor_receptacle_spawn_map(
    spawn_map: AI2ThorReceptacleSpawnMap,
    *,
    fresh_scene: Scene,
    frozen_scene: Scene,
) -> AI2ThorReceptacleSpawnMap:
    """Rebind one exact fresh native query to a validated frozen scene digest.

    Only the scene digest is substituted. Runtime identity, native IDs, exact
    native positions and trigger-grid patches all remain those returned by the
    fresh query, so drift in any native evidence still changes the source
    digest and fails its upstream lineage comparison.
    """

    checked = _strict_receptacle_spawn_map(spawn_map)
    if type(fresh_scene) is not Scene or type(frozen_scene) is not Scene:
        raise TypeError("capture-bound spawn scenes must be exact Scene values")
    expected_fresh_scene_sha256 = _receptacle_scene_sha256(
        fresh_scene,
        checked.surface_patches,
    )
    if (
        checked.scene_id != fresh_scene.scene_id
        or checked.scene_sha256 != expected_fresh_scene_sha256
        or frozen_scene.scene_id != fresh_scene.scene_id
    ):
        raise ValueError("capture-bound spawn map does not bind the fresh scene")
    frozen_scene_sha256 = _receptacle_scene_sha256(
        frozen_scene,
        checked.surface_patches,
    )
    source_sha256 = _receptacle_spawn_source_sha256(
        scene_id=checked.scene_id,
        subject_object_id=checked.subject_object_id,
        support_object_id=checked.support_object_id,
        native_subject_object_id=checked.native_subject_object_id,
        native_support_object_id=checked.native_support_object_id,
        runtime_identity=checked.runtime_identity,
        positions_sha256=checked.positions_sha256,
        scene_sha256=frozen_scene_sha256,
        surface_patches=checked.surface_patches,
    )
    return AI2ThorReceptacleSpawnMap(
        scene_id=checked.scene_id,
        subject_object_id=checked.subject_object_id,
        support_object_id=checked.support_object_id,
        native_subject_object_id=checked.native_subject_object_id,
        native_support_object_id=checked.native_support_object_id,
        runtime_identity=checked.runtime_identity,
        positions=checked.positions,
        positions_sha256=checked.positions_sha256,
        scene_sha256=frozen_scene_sha256,
        source_sha256=source_sha256,
        surface_patches=checked.surface_patches,
    )


def build_receptacle_support_position_region(
    scene: Scene,
    spawn_map: AI2ThorReceptacleSpawnMap,
) -> SubjectPositionRegion:
    """Build the fixed-pose subject-anchor locus from trigger-grid evidence."""

    checked = _strict_receptacle_spawn_map(spawn_map)
    subject = scene.object_by_id(checked.subject_object_id)
    scene_sha256 = _receptacle_scene_sha256(scene, checked.surface_patches)
    if (
        checked.scene_id != scene.scene_id
        or checked.scene_sha256 != scene_sha256
        or subject.support_object_id != checked.support_object_id
        or not checked.surface_patches
    ):
        raise ValueError("receptacle surface patches do not bind the scene subject")
    geometry = conservative_receptacle_position_geometry(
        surface_patch_bounds_xy=tuple(
            (item.x_min, item.z_min, item.x_max, item.z_max)
            for item in checked.surface_patches
        ),
        subject=subject,
    )
    return SubjectPositionRegion(
        region_id=(
            f"native-receptacle-trigger-grid:{subject.object_id}:"
            f"{checked.source_sha256[:16]}"
        ),
        subject_object_id=subject.object_id,
        source_kind="ai2thor-receptacle-trigger-grid-v1",
        source_sha256=checked.source_sha256,
        components=planar_polygon_payloads(geometry),
    )


def build_navigation_feasibility_map(
    scene: Scene,
    *,
    subject_object_id: str,
    room_polygon_xy: tuple[Vec2, ...],
    reachable_positions: tuple[AI2ThorNativePosition, ...],
    agent_radius_m: float,
    clearance_m: float,
) -> AI2ThorNavigationFeasibilityMap:
    """Deterministically bind native navigation evidence to one Scene."""
    if (
        type(agent_radius_m) not in (int, float)
        or not math.isfinite(float(agent_radius_m))
        or float(agent_radius_m) <= 0.0
        or type(clearance_m) not in (int, float)
        or not math.isfinite(float(clearance_m))
        or float(clearance_m) < 0.0
        or float(clearance_m) >= float(agent_radius_m)
    ):
        raise ValueError(
            "navigation radius must be positive and clearance non-negative "
            "and smaller than the radius"
        )
    subject = scene.object_by_id(subject_object_id)
    positions_payload = [
        {"x": item.x, "y": item.y, "z": item.z} for item in reachable_positions
    ]
    positions_bytes = json.dumps(
        positions_payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    positions_digest = sha256(positions_bytes).hexdigest()
    source_payload = {
        "agent_radius_m": float(agent_radius_m),
        "clearance_m": float(clearance_m),
        "method": "ai2thor-navigation-v1",
        "reachable_positions_sha256": positions_digest,
        "room_polygon_xy": [{"x": point.x, "y": point.y} for point in room_polygon_xy],
        "scene_id": scene.scene_id,
        "subject_object_id": subject.object_id,
        "subject_obb": subject.obb.model_dump(mode="json"),
        "subject_position": subject.position.model_dump(mode="json"),
    }
    source_digest = sha256(
        json.dumps(
            source_payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    geometry = conservative_navigation_position_geometry(
        room_polygon_xy=room_polygon_xy,
        reachable_positions_xy=tuple(
            Vec2(x=position.x, y=position.z) for position in reachable_positions
        ),
        subject=subject,
        agent_radius_m=float(agent_radius_m),
        clearance_m=float(clearance_m),
    )
    position_region = SubjectPositionRegion(
        region_id=f"native-navigation:{subject.object_id}:{source_digest[:16]}",
        subject_object_id=subject.object_id,
        source_kind="ai2thor-navigation-v1",
        source_sha256=source_digest,
        components=planar_polygon_payloads(geometry),
    )
    return AI2ThorNavigationFeasibilityMap(
        scene_id=scene.scene_id,
        subject_object_id=subject.object_id,
        agent_radius_m=float(agent_radius_m),
        clearance_m=float(clearance_m),
        reachable_positions=reachable_positions,
        reachable_positions_sha256=positions_digest,
        source_sha256=source_digest,
        position_region=position_region,
    )


def _domain_object_metadata(raw_objects: list[Any]) -> list[Any]:
    return [
        item
        for item in raw_objects
        if (
            not isinstance(item, dict)
            or item.get("objectType") not in _STRUCTURAL_OBJECT_TYPES
        )
    ]


def _validated_native_object_metadata(
    raw_objects: list[Any],
) -> list[dict[str, Any]]:
    validated_objects: list[dict[str, Any]] = []
    object_ids: set[str] = set()
    object_names: set[str] = set()
    for item in raw_objects:
        if type(item) is not dict:
            raise AI2ThorNativeReturnError(
                "AI2-THOR object collection entries must be exact dictionaries"
            )
        object_id = item.get("objectId")
        name = item.get("name")
        category = item.get("objectType")
        if any(
            type(value) is not str or not value.strip()
            for value in (object_id, name, category)
        ):
            raise AI2ThorNativeReturnError(
                "AI2-THOR object ID, name, and type must be non-empty text"
            )
        if object_id in object_ids or name in object_names:
            raise AI2ThorNativeReturnError(
                "AI2-THOR object IDs and names must be unique"
            )
        object_ids.add(object_id)
        object_names.add(name)
        if any(
            field in item and type(item[field]) is not bool
            for field in ("moveable", "pickupable")
        ):
            raise AI2ThorNativeReturnError(
                "AI2-THOR object mobility fields must be exact booleans"
            )
        parents = item.get("parentReceptacles")
        if parents is not None and (
            type(parents) is not list
            or any(type(parent) is not str or not parent.strip() for parent in parents)
        ):
            raise AI2ThorNativeReturnError(
                "AI2-THOR parent receptacles must be null or a list of "
                "non-empty text IDs"
            )
        validated_objects.append(item)

    for item in validated_objects:
        for parent in item.get("parentReceptacles") or []:
            if parent not in object_ids:
                raise AI2ThorNativeReturnError(
                    f"observed support {parent!r} has no stable object identity"
                )

    return [
        {
            **item,
            "parentReceptacles": list(item.get("parentReceptacles") or []),
        }
        for item in validated_objects
    ]


def _cyclic_domain_object_ids(
    graph: dict[str, tuple[str, ...]],
) -> frozenset[str]:
    """Return every member of every directed cycle, including self-loops."""

    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    active: set[str] = set()
    cyclic: set[str] = set()

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        active.add(node)
        for parent in graph.get(node, ()):
            if parent not in graph:
                continue
            if parent not in indices:
                visit(parent)
                lowlinks[node] = min(lowlinks[node], lowlinks[parent])
            elif parent in active:
                lowlinks[node] = min(lowlinks[node], indices[parent])
        if lowlinks[node] != indices[node]:
            return
        component: list[str] = []
        while True:
            member = stack.pop()
            active.remove(member)
            component.append(member)
            if member == node:
                break
        if len(component) > 1 or node in graph.get(node, ()):
            cyclic.update(component)

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return frozenset(cyclic)


def build_ai2thor_native_support_facts(
    scene: Scene,
    raw_objects: list[object],
) -> tuple[AI2ThorNativeSupportFact, ...]:
    """Validate and normalize native parent lineage without a platform action."""

    if type(scene) is not Scene:
        raise TypeError("native support scene must be an exact Scene")
    if type(raw_objects) is not list:
        raise TypeError("native support raw_objects must be an exact list")
    validated = _validated_native_object_metadata(raw_objects)
    raw_by_id = {
        item["objectId"]: (item["name"], item["objectType"]) for item in validated
    }
    scene_ids = tuple(item.object_id for item in scene.objects)
    scene_names = tuple(item.name for item in scene.objects)
    if len(scene_ids) != len(set(scene_ids)) or len(scene_names) != len(
        set(scene_names)
    ):
        raise AI2ThorNativeReturnError(
            "stable scene object IDs and names must be unique"
        )
    scene_by_name = {item.name: item for item in scene.objects}
    raw_domain_names = {
        item["name"]
        for item in validated
        if item["objectType"] not in _STRUCTURAL_OBJECT_TYPES
    }
    if set(scene_by_name) != raw_domain_names:
        raise AI2ThorNativeReturnError(
            "native and stable scene object name rosters must match"
        )
    native_by_name = {item["name"]: item for item in validated}

    normalized: dict[
        str,
        tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]],
    ] = {}
    for object_name, scene_object in scene_by_name.items():
        native = native_by_name[object_name]
        raw_parents = tuple(sorted(set(native["parentReceptacles"])))
        structural: list[str] = []
        domain: list[str] = []
        normalized_raw: list[str] = []
        for raw_parent in raw_parents:
            parent_name, parent_type = raw_by_id[raw_parent]
            if parent_type in _STRUCTURAL_OBJECT_TYPES:
                structural.append(raw_parent)
                normalized_raw.append(raw_parent)
            else:
                try:
                    stable_parent = scene_by_name[parent_name].object_id
                except KeyError as error:
                    raise AI2ThorNativeReturnError(
                        f"observed support {raw_parent!r} has no stable object identity"
                    ) from error
                domain.append(stable_parent)
                normalized_raw.append(stable_parent)
        normalized[scene_object.object_id] = (
            object_name,
            native["objectId"],
            tuple(sorted(set(normalized_raw))),
            tuple(sorted(set(structural))),
            tuple(sorted(set(domain))),
        )

    graph = {object_id: values[4] for object_id, values in normalized.items()}
    cyclic = _cyclic_domain_object_ids(graph)
    facts: list[AI2ThorNativeSupportFact] = []
    for object_id in sorted(normalized):
        object_name, native_object_id, raw, structural, domain = normalized[object_id]
        floor_parents = tuple(
            parent for parent in structural if raw_by_id[parent][1] == "Floor"
        )
        plausible_count = len(domain) + len(floor_parents)
        if object_id in cyclic:
            kind = AI2ThorNativeSupportKind.CYCLIC
            support_object_id = None
            floor_object_id = None
        elif plausible_count > 1:
            kind = AI2ThorNativeSupportKind.MULTIPLE_AMBIGUOUS
            support_object_id = None
            floor_object_id = None
        elif len(domain) == 1 and not structural:
            kind = AI2ThorNativeSupportKind.RECEPTACLE
            support_object_id = domain[0]
            floor_object_id = None
        elif len(floor_parents) == 1 and not domain and len(structural) == 1:
            kind = AI2ThorNativeSupportKind.FLOOR
            support_object_id = None
            floor_object_id = floor_parents[0]
        else:
            kind = AI2ThorNativeSupportKind.UNKNOWN
            support_object_id = None
            floor_object_id = None
            # Structural parents other than Floor are raw provenance, not usable
            # support. Keep them out of the structural support partition so the
            # UNKNOWN invariant remains explicit.
            structural = ()
        facts.append(
            AI2ThorNativeSupportFact(
                scene_id=scene.scene_id,
                object_id=object_id,
                object_name=object_name,
                native_object_id=native_object_id,
                raw_parent_object_ids=raw,
                structural_parent_object_ids=structural,
                domain_parent_object_ids=domain,
                support_kind=kind,
                support_object_id=support_object_id,
                floor_object_id=floor_object_id,
            )
        )
    return tuple(facts)


class AI2ThorSupportMixin:
    def capture_spawn_maps(
        self,
        facts: SourceCaptureFacts,
        *,
        subject_object_ids: tuple[str, ...],
    ) -> tuple[AdapterSpawnMap, ...]:
        """Capture maps for only the generation-selected source subjects."""

        if type(facts) is not SourceCaptureFacts:
            raise AdapterOperationError("spawn capture facts must be exact")
        if (
            type(subject_object_ids) is not tuple
            or any(type(item) is not str or not item for item in subject_object_ids)
            or len(set(subject_object_ids)) != len(subject_object_ids)
        ):
            raise AdapterOperationError(
                "spawn subject IDs must be unique exact strings"
            )
        try:
            maps = []
            for subject_object_id in subject_object_ids:
                subject = facts.scene.object_by_id(subject_object_id)
                if subject.support_object_id is None:
                    raise ValueError("selected subject has no support")
                native = self.receptacle_spawn_map(
                    facts.scene,
                    subject_object_id=subject_object_id,
                )
                region = (
                    build_receptacle_support_position_region(facts.scene, native)
                    if native.surface_patches
                    else None
                )
                maps.append(
                    adapter_spawn_map_from_native(
                        native,
                        binding=facts.binding,
                        position_region=region,
                    )
                )
            return tuple(maps)
        except (AI2ThorNativeReturnError, RuntimeError, ValueError, KeyError) as error:
            raise AdapterOperationError(str(error)) from error

    def native_support_facts(
        self,
        scene: Scene,
    ) -> tuple[AI2ThorNativeSupportFact, ...]:
        """Read support lineage from the exact current event without an action."""

        event = self._current_event_for_scene(scene)
        raw_objects = event.metadata.get("objects")
        if type(raw_objects) is not list:
            raise AI2ThorNativeReturnError("invalid AI2-THOR object collection")
        return build_ai2thor_native_support_facts(scene, raw_objects)

    def reachable_agent_positions(
        self,
        scene: Scene,
    ) -> tuple[AI2ThorNativePosition, ...]:
        """Return a deterministic, strictly validated native navigation grid."""
        controller = self._require_active()
        current_event = self._current_event_for_scene(scene)
        expected_native_scene_name = self._native_scene_name(current_event)
        expected_positions = {obj.name: obj.position for obj in scene.objects}
        expected_rotations = {
            obj.name: self._native_rotation_for(scene.scene_id, obj)
            for obj in scene.objects
        }
        try:
            event = self._step(
                controller,
                "GetReachablePositions",
                action="GetReachablePositions",
            )
            event = self._checked_scene_event(
                controller,
                event,
                "GetReachablePositions",
                scene.scene_id,
            )
            self._validate_native_scene_name_or_poison(
                event,
                expected_native_scene_name,
            )
            self._validate_returned_state(
                scene,
                event,
                expected_positions,
                expected_rotations,
            )
            action_return = event.metadata.get("actionReturn")
            if type(action_return) is not list:
                raise ValueError("GetReachablePositions actionReturn must be a list")
            keyed_positions: list[
                tuple[tuple[int, int, int], AI2ThorNativePosition]
            ] = []
            seen_keys: set[tuple[int, int, int]] = set()
            for index, raw_position in enumerate(action_return):
                position = self._native_position(
                    raw_position,
                    f"reachable position {index}",
                )
                quantized_key = tuple(
                    round(value / _REACHABLE_POSITION_QUANTIZATION_M)
                    for value in (position.x, position.z, position.y)
                )
                if quantized_key in seen_keys:
                    raise ValueError(
                        "GetReachablePositions returned a duplicate quantized position"
                    )
                seen_keys.add(quantized_key)
                keyed_positions.append((quantized_key, position))
        except BaseException:
            self._poison_scene_state()
            raise
        keyed_positions.sort(key=lambda item: item[0])
        self._event = event
        self._current_scene = scene
        return tuple(position for _, position in keyed_positions)

    def receptacle_spawn_map(
        self,
        scene: Scene,
        *,
        subject_object_id: str,
    ) -> AI2ThorReceptacleSpawnMap:
        """Capture deterministic native receptacle coordinates for one subject.

        Returned coordinates are source facts, not collision-free placements.
        The query is executed once against the exact current source event and
        cannot be used as a per-candidate platform search loop.
        """

        controller = self._require_active()
        current_event = self._current_event_for_scene(scene)
        subject = scene.object_by_id(subject_object_id)
        support_object_id = subject.support_object_id
        if support_object_id is None:
            raise ValueError("receptacle spawn subject has no declared support")
        support = scene.object_by_id(support_object_id)
        runtime_identity = self.runtime_identity()
        expected_native_scene_name = self._native_scene_name(current_event)
        expected_positions = {obj.name: obj.position for obj in scene.objects}
        expected_rotations = {
            obj.name: self._native_rotation_for(scene.scene_id, obj)
            for obj in scene.objects
        }
        native_subject_object_id = self._native_object_id_for_name(
            current_event,
            subject.name,
        )
        native_support_object_id = self._native_object_id_for_name(
            current_event,
            support.name,
        )
        try:
            event = self._step(
                controller,
                "GetSpawnCoordinatesAboveReceptacle",
                action="GetSpawnCoordinatesAboveReceptacle",
                objectId=native_support_object_id,
                anywhere=True,
            )
            event = self._checked_scene_event(
                controller,
                event,
                "GetSpawnCoordinatesAboveReceptacle",
                scene.scene_id,
            )
            self._validate_native_scene_name_or_poison(
                event,
                expected_native_scene_name,
            )
            self._validate_returned_state(
                scene,
                event,
                expected_positions,
                expected_rotations,
            )
            action_return = event.metadata.get("actionReturn")
            if type(action_return) is not list or not action_return:
                raise AI2ThorNativeReturnError(
                    "GetSpawnCoordinatesAboveReceptacle actionReturn must be "
                    "a non-empty list"
                )
            raw_positions: list[AI2ThorNativePosition] = []
            keyed_positions: list[
                tuple[tuple[int, int, int], AI2ThorNativePosition]
            ] = []
            seen_keys: set[tuple[int, int, int]] = set()
            for index, raw_position in enumerate(action_return):
                position = self._native_position(
                    raw_position,
                    f"receptacle spawn position {index}",
                    error_type=AI2ThorNativeReturnError,
                )
                raw_positions.append(position)
                quantized_key = tuple(
                    round(value / _REACHABLE_POSITION_QUANTIZATION_M)
                    for value in (position.x, position.z, position.y)
                )
                if quantized_key in seen_keys:
                    raise AI2ThorNativeReturnError(
                        "GetSpawnCoordinatesAboveReceptacle returned a duplicate "
                        "quantized position"
                    )
                seen_keys.add(quantized_key)
                keyed_positions.append((quantized_key, position))
        except BaseException:
            self._poison_scene_state()
            raise
        positions = tuple(
            sorted(
                (position for _, position in keyed_positions),
                key=lambda item: (item.x, item.z, item.y),
            )
        )
        surface_patches = build_ai2thor_receptacle_surface_patches(tuple(raw_positions))
        positions_sha256 = _native_positions_sha256(positions)
        scene_sha256 = _receptacle_scene_sha256(scene, surface_patches)
        source_sha256 = _receptacle_spawn_source_sha256(
            scene_id=scene.scene_id,
            subject_object_id=subject.object_id,
            support_object_id=support.object_id,
            native_subject_object_id=native_subject_object_id,
            native_support_object_id=native_support_object_id,
            runtime_identity=runtime_identity,
            positions_sha256=positions_sha256,
            scene_sha256=scene_sha256,
            surface_patches=surface_patches,
        )
        spawn_map = AI2ThorReceptacleSpawnMap(
            scene_id=scene.scene_id,
            subject_object_id=subject.object_id,
            support_object_id=support.object_id,
            native_subject_object_id=native_subject_object_id,
            native_support_object_id=native_support_object_id,
            runtime_identity=runtime_identity,
            positions=positions,
            positions_sha256=positions_sha256,
            scene_sha256=scene_sha256,
            source_sha256=source_sha256,
            surface_patches=surface_patches,
        )
        self._event = event
        self._current_scene = scene
        return spawn_map

    def conservative_floor_envelope(
        self,
        scene: Scene,
        clearance_m: float,
    ) -> AI2ThorFloorEnvelope:
        """Derive an inward-offset rectangle from the current native floor AABB."""
        event = self._current_event_for_scene(scene)
        if (
            isinstance(clearance_m, bool)
            or not isinstance(clearance_m, (int, float))
            or not math.isfinite(float(clearance_m))
            or clearance_m < 0.0
        ):
            raise ValueError("floor clearance must be finite and non-negative")
        raw_objects = event.metadata.get("objects")
        raw_objects_is_list = isinstance(raw_objects, list)
        if not raw_objects_is_list:
            raise ValueError("current event has no structural Floor collection")
        floors: list[dict[str, Any]] = []
        for item in raw_objects:
            if not isinstance(item, dict) or item.get("objectType") != "Floor":
                continue
            bounds = item.get("axisAlignedBoundingBox")
            size = bounds.get("size") if isinstance(bounds, dict) else None
            size_is_dict = isinstance(size, dict)
            if not size_is_dict:
                raise ValueError("structural Floor must have a finite positive AABB")
            try:
                native_size_x = float(size["x"])
                native_size_y = float(size["y"])
                native_size_z = float(size["z"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "structural Floor must have a finite positive AABB"
                ) from exc
            if (
                not all(
                    math.isfinite(value)
                    for value in (native_size_x, native_size_y, native_size_z)
                )
                or native_size_x <= 0.0
                or native_size_z <= 0.0
                or native_size_y < 0.0
            ):
                raise ValueError("structural Floor must have a finite positive AABB")
            if native_size_y == 0.0:
                continue
            floors.append(item)
        if len(floors) != 1:
            raise ValueError("current event must contain exactly one structural Floor")
        floor = floors[0]
        bounds = floor.get("axisAlignedBoundingBox")
        bounds_is_dict = isinstance(bounds, dict)
        if not bounds_is_dict:
            raise ValueError("structural Floor must have a finite positive AABB")
        try:
            center = ai2thor_position_to_world(Vec3(**bounds["center"]))
            size = bounds["size"]
            extent = Vec3(
                x=float(size["x"]),
                y=float(size["z"]),
                z=float(size["y"]),
            )
            floor_object_id = str(floor["objectId"])
            floor_name = str(floor["name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "structural Floor must have a finite positive AABB"
            ) from exc
        geometry_values = (
            center.x,
            center.y,
            center.z,
            extent.x,
            extent.y,
            extent.z,
        )
        if (
            not all(math.isfinite(value) for value in geometry_values)
            or min(extent.x, extent.y, extent.z) <= 0.0
            or not floor_object_id
            or not floor_name
        ):
            raise ValueError("structural Floor must have a finite positive AABB")
        source = self.procedural_scenes.get(scene.scene_id)
        effective_native_bounds: tuple[float, float, float, float] | None = None
        if source is not None:
            native_center = bounds.get("center")
            if not isinstance(native_center, dict):
                raise ValueError("structural Floor must have a finite positive AABB")
            try:
                native_center_x = float(native_center["x"])
                native_center_z = float(native_center["z"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "structural Floor must have a finite positive AABB"
                ) from error
            floor_aabb_bounds = (
                native_center_x - extent.x / 2.0,
                native_center_z - extent.y / 2.0,
                native_center_x + extent.x / 2.0,
                native_center_z + extent.y / 2.0,
            )
            if not all(
                math.isclose(expected, actual, rel_tol=0.0, abs_tol=1e-5)
                for expected, actual in zip(
                    source.floor_xz_bounds,
                    floor_aabb_bounds,
                    strict=True,
                )
            ):
                raise ValueError(
                    "procedural floorPolygon does not match structural Floor AABB"
                )
            effective_native_bounds = (
                max(source.floor_xz_bounds[0], floor_aabb_bounds[0]),
                max(source.floor_xz_bounds[1], floor_aabb_bounds[1]),
                min(source.floor_xz_bounds[2], floor_aabb_bounds[2]),
                min(source.floor_xz_bounds[3], floor_aabb_bounds[3]),
            )
        if effective_native_bounds is None:
            effective_native_bounds = (
                center.x - extent.x / 2.0,
                center.y - extent.y / 2.0,
                center.x + extent.x / 2.0,
                center.y + extent.y / 2.0,
            )
        minimum_x = effective_native_bounds[0] + float(clearance_m)
        minimum_y = effective_native_bounds[1] + float(clearance_m)
        maximum_x = effective_native_bounds[2] - float(clearance_m)
        maximum_y = effective_native_bounds[3] - float(clearance_m)
        if maximum_x <= minimum_x or maximum_y <= minimum_y:
            raise ValueError("floor clearance leaves no positive envelope")
        native_aabb = OBB(
            center=center,
            extent=extent,
            rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        return AI2ThorFloorEnvelope(
            scene_id=scene.scene_id,
            floor_object_id=floor_object_id,
            floor_name=floor_name,
            native_aabb=native_aabb,
            floor_top_z=center.z + extent.z / 2.0,
            clearance_m=float(clearance_m),
            polygon_xy=(
                Vec2(x=minimum_x, y=minimum_y),
                Vec2(x=maximum_x, y=minimum_y),
                Vec2(x=maximum_x, y=maximum_y),
                Vec2(x=minimum_x, y=maximum_y),
            ),
        )

    def conservative_collision_map(
        self,
        scene: Scene,
        *,
        subject_object_id: str,
        clearance_m: float,
    ) -> AI2ThorNativeFeasibilityMap:
        """Expand stationary native OBBs into view-independent obstacles."""
        self._current_event_for_scene(scene)
        if (
            isinstance(clearance_m, bool)
            or not isinstance(clearance_m, (int, float))
            or not math.isfinite(float(clearance_m))
            or float(clearance_m) <= 0.0
        ):
            raise ValueError("collision clearance must be finite and positive")
        subject = scene.object_by_id(subject_object_id)
        excluded_ids = {subject.object_id, subject.support_object_id}
        clearance = float(clearance_m)
        obstacles = tuple(
            CollisionObstacle(
                obstacle_id=f"native-clearance:{obj.object_id}",
                source_object_id=obj.object_id,
                clearance_m=clearance,
                obb=obj.obb,
            )
            for obj in sorted(scene.objects, key=lambda item: item.object_id)
            if obj.object_id not in excluded_ids
        )
        return AI2ThorNativeFeasibilityMap(
            scene_id=scene.scene_id,
            subject_object_id=subject.object_id,
            clearance_m=clearance,
            obstacles=obstacles,
        )

    def conservative_navigation_map(
        self,
        scene: Scene,
        *,
        subject_object_id: str,
        room_polygon_xy: tuple[Vec2, ...],
        agent_radius_m: float,
        clearance_m: float,
    ) -> AI2ThorNavigationFeasibilityMap:
        """Bind the current reachable grid to a conservative subject locus."""
        self._current_event_for_scene(scene)
        positions = self.reachable_agent_positions(scene)
        return build_navigation_feasibility_map(
            scene,
            subject_object_id=subject_object_id,
            room_polygon_xy=room_polygon_xy,
            reachable_positions=positions,
            agent_radius_m=agent_radius_m,
            clearance_m=clearance_m,
        )
