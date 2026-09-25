"""Exact, CPU-only rigid SE(3) mathematics for the M6 profile.

The functions here are deterministic rational predicates. They do not choose a
candidate, trust a backend, query native geometry, or claim swept-path safety.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction as F

from spatialcf.domain.rigid_se3 import (
    MAX_LEDGER_EVENTS,
    RigidSE3ContactFact,
    RigidSE3CountEvent,
    RigidSE3ExactSourceFacts,
    RigidSE3JointState,
    RigidSE3Ledger,
    RigidSE3Limits,
    RigidSE3ObjectivePolicy,
    RigidSE3Pose,
    RigidSE3RegionFact,
)

Vector = tuple[F, F, F]
Matrix = tuple[Vector, Vector, Vector]
I: Matrix = ((F(1), F(0), F(0)), (F(0), F(1), F(0)), (F(0), F(0), F(1)))
ZERO: Vector = (F(0), F(0), F(0))


class ExactKernelLimit(ValueError):
    def __init__(self, reason: str, item: str):
        super().__init__(f"{reason}: {item}")
        self.reason = reason
        self.item = item


class CheckKernelLimit(ExactKernelLimit):
    """The independent replay budget expired, irrespective of producer stage."""


@dataclass
class ExactBudget:
    """Precharge deterministic work; reject excessive rational width exactly."""

    limits: RigidSE3Limits
    stage: str
    operations: int = 0
    peak_numeric_bits: int = 0
    events: list[tuple[str, int]] = field(default_factory=list)
    audit: ExactBudget | None = None

    def __post_init__(self) -> None:
        if self.stage not in {"COMPILE", "SOLVE", "CHECK"}:
            raise ValueError("exact budget stage must be registered")

    def charge(self, kind: str, amount: int = 1) -> None:
        if amount < 1:
            raise ValueError("budget charge must be positive")
        if self.audit is not None:
            try:
                self.audit.charge(kind, amount)
            except ExactKernelLimit as error:
                raise CheckKernelLimit(error.reason, error.item) from error
        cap = min(
            self.limits.max_exact_operations,
            getattr(self.limits, f"{self.stage.lower()}_operations"),
        )
        if self.operations + amount > cap:
            raise ExactKernelLimit("RESOURCE_LIMIT", kind)
        if (not self.events or self.events[-1][0] != kind) and len(self.events) >= MAX_LEDGER_EVENTS:
            raise ExactKernelLimit("RESOURCE_LIMIT", kind)
        self.operations += amount
        if self.events and self.events[-1][0] == kind:
            previous_kind, previous_amount = self.events[-1]
            self.events[-1] = (previous_kind, previous_amount + amount)
        else:
            self.events.append((kind, amount))

    def as_ledger(self, *, completed_items: tuple[str, ...] = (),
                  first_unprocessed_item: str | None = None,
                  reason: str = "NONE") -> RigidSE3Ledger:
        cumulative = 0
        events = []
        for index, (kind, amount) in enumerate(self.events):
            cumulative += amount
            events.append(RigidSE3CountEvent(
                event_id=f"event:rigid-se3:{index:08d}", stage=self.stage,
                kind=kind, amount=amount, cumulative=cumulative,
            ))
        return RigidSE3Ledger(
            stage=self.stage, events=tuple(events), completed_items=completed_items,
            first_unprocessed_item=first_unprocessed_item,
            peak_numeric_bits=self.peak_numeric_bits, reason=reason,
        )

    def guard(self, *values: F) -> None:
        for value in values:
            if not isinstance(value, F):
                raise TypeError("exact kernel requires Fraction values")
            self._track_int(value.numerator, value.denominator)

    def _track_int(self, *values: int) -> None:
        if self.audit is not None:
            try:
                self.audit._track_int(*values)
            except ExactKernelLimit as error:
                raise CheckKernelLimit(error.reason, error.item) from error
        width = max((abs(value).bit_length() for value in values), default=0)
        self.peak_numeric_bits = max(self.peak_numeric_bits, width)
        if width > self.limits.numeric_bits:
            raise ExactKernelLimit("RESOURCE_LIMIT", "rational-width")

    def _product_fits(self, left: int, right: int) -> None:
        if not left or not right:
            return
        width = abs(left).bit_length() + abs(right).bit_length()
        cap = self.limits.numeric_bits
        if width <= cap:
            return
        if width - 1 > cap or abs(left) > ((1 << cap) - 1) // abs(right):
            raise ExactKernelLimit("RESOURCE_LIMIT", "rational-intermediate")

    def add(self, left: F, right: F) -> F:
        self.charge("EXACT_ARITHMETIC")
        self.guard(left, right)
        self._product_fits(left.numerator, right.denominator)
        self._product_fits(right.numerator, left.denominator)
        self._product_fits(left.denominator, right.denominator)
        left_num = left.numerator * right.denominator
        right_num = right.numerator * left.denominator
        self._track_int(left_num, right_num)
        if (left_num >= 0) == (right_num >= 0):
            cap = self.limits.numeric_bits
            if max(abs(left_num).bit_length(), abs(right_num).bit_length()) >= cap:
                maximum = (1 << cap) - 1
                if abs(left_num) > maximum - abs(right_num):
                    raise ExactKernelLimit("RESOURCE_LIMIT", "rational-intermediate")
        numerator = left_num + right_num
        denominator = left.denominator * right.denominator
        self._track_int(left_num, right_num, numerator, denominator)
        result = F(numerator, denominator)
        self.guard(result)
        return result

    def multiply(self, left: F, right: F) -> F:
        self.charge("EXACT_ARITHMETIC")
        self.guard(left, right)
        self._product_fits(left.numerator, right.numerator)
        self._product_fits(left.denominator, right.denominator)
        numerator = left.numerator * right.numerator
        denominator = left.denominator * right.denominator
        self._track_int(numerator, denominator)
        result = F(numerator, denominator)
        self.guard(result)
        return result

    def compare(self, left: F, right: F) -> int:
        self.charge("EXACT_ARITHMETIC")
        self.guard(left, right)
        self._product_fits(left.numerator, right.denominator)
        self._product_fits(right.numerator, left.denominator)
        left_num = left.numerator * right.denominator
        right_num = right.numerator * left.denominator
        self._track_int(left_num, right_num)
        return (left_num > right_num) - (left_num < right_num)

    def divide(self, left: F, right: F) -> F:
        if right == 0:
            raise ZeroDivisionError("exact rational denominator is zero")
        self.guard(left, right)
        return self.multiply(left, F(right.denominator, right.numerator))


def _charge(budget: ExactBudget | None, kind: str, *values: F) -> None:
    if budget is not None:
        budget.charge(kind)
        budget.guard(*values)


def _addf(left: F, right: F, budget: ExactBudget | None) -> F:
    return budget.add(left, right) if budget is not None else left + right


def _subf(left: F, right: F, budget: ExactBudget | None) -> F:
    return _addf(left, -right, budget)


def _mulf(left: F, right: F, budget: ExactBudget | None) -> F:
    return budget.multiply(left, right) if budget is not None else left * right


def _divf(left: F, right: F, budget: ExactBudget | None) -> F:
    return budget.divide(left, right) if budget is not None else left / right


def _cmpf(left: F, right: F, budget: ExactBudget | None) -> int:
    if budget is not None:
        return budget.compare(left, right)
    return (left > right) - (left < right)


def _equal_vec(left: Vector, right: Vector, budget: ExactBudget | None) -> bool:
    return all(_cmpf(a, b, budget) == 0 for a, b in zip(left, right, strict=True))


def dot(a: Vector, b: Vector, budget: ExactBudget | None = None) -> F:
    result = F()
    for left, right in zip(a, b, strict=True):
        result = _addf(result, _mulf(left, right, budget), budget)
    return result


def cross(a: Vector, b: Vector, budget: ExactBudget | None = None) -> Vector:
    return (
        _subf(_mulf(a[1], b[2], budget), _mulf(a[2], b[1], budget), budget),
        _subf(_mulf(a[2], b[0], budget), _mulf(a[0], b[2], budget), budget),
        _subf(_mulf(a[0], b[1], budget), _mulf(a[1], b[0], budget), budget),
    )


def add(a: Vector, b: Vector, budget: ExactBudget | None = None) -> Vector:
    result = tuple(_addf(x, y, budget) for x, y in zip(a, b, strict=True))
    return result  # type: ignore[return-value]


def subtract(a: Vector, b: Vector, budget: ExactBudget | None = None) -> Vector:
    result = tuple(_subf(x, y, budget) for x, y in zip(a, b, strict=True))
    return result  # type: ignore[return-value]


def scale(a: Vector, amount: F, budget: ExactBudget | None = None) -> Vector:
    result = tuple(_mulf(x, amount, budget) for x in a)
    return result  # type: ignore[return-value]


def mat_vec(matrix: Matrix, vector: Vector, budget: ExactBudget | None = None) -> Vector:
    return tuple(dot(row, vector, budget) for row in matrix)  # type: ignore[return-value]


def mat_mul(left: Matrix, right: Matrix, budget: ExactBudget | None = None) -> Matrix:
    columns = transpose(right)
    return tuple(tuple(dot(row, column, budget) for column in columns)
                 for row in left)  # type: ignore[return-value]


def transpose(matrix: Matrix) -> Matrix:
    return tuple(tuple(matrix[row][column] for row in range(3))
                 for column in range(3))  # type: ignore[return-value]


@dataclass(frozen=True)
class Transform:
    rotation: Matrix = I
    translation: Vector = ZERO

    @classmethod
    def from_pose(cls, pose: RigidSE3Pose) -> Transform:
        return cls(
            rotation=tuple(tuple(cell.as_fraction for cell in row)
                           for row in pose.rotation.rows),  # type: ignore[arg-type]
            translation=pose.translation_m.fractions,
        )


def apply_point(transform: Transform, point: Vector,
                budget: ExactBudget | None = None) -> Vector:
    return add(mat_vec(transform.rotation, point, budget), transform.translation, budget)


def compose(left: Transform, right: Transform,
            budget: ExactBudget | None = None) -> Transform:
    return Transform(
        rotation=mat_mul(left.rotation, right.rotation, budget),
        translation=apply_point(left, right.translation, budget),
    )


def inverse(transform: Transform, budget: ExactBudget | None = None) -> Transform:
    rotation = transpose(transform.rotation)
    return Transform(rotation=rotation, translation=scale(
        mat_vec(rotation, transform.translation, budget), F(-1), budget,
    ))


def delta(before: Transform, after: Transform,
          budget: ExactBudget | None = None) -> Transform:
    return compose(after, inverse(before, budget), budget)


def rotation_about_axis(axis: Vector, cosine: F, sine: F,
                        budget: ExactBudget | None = None) -> Matrix:
    circle_norm = _addf(_mulf(cosine, cosine, budget), _mulf(sine, sine, budget), budget)
    if (_cmpf(dot(axis, axis, budget), F(1), budget) != 0
            or _cmpf(circle_norm, F(1), budget) != 0):
        raise ValueError("Rodrigues requires an exact unit axis and circle point")
    one_minus = _subf(F(1), cosine, budget)
    skew = ((F(0), -axis[2], axis[1]),
            (axis[2], F(0), -axis[0]),
            (-axis[1], axis[0], F(0)))
    return tuple(tuple(_addf(
        _addf(cosine if i == j else F(0),
              _mulf(one_minus, _mulf(axis[i], axis[j], budget), budget), budget),
        _mulf(sine, skew[i][j], budget), budget,
    ) for j in range(3)) for i in range(3))  # type: ignore[return-value]


def forward_kinematics(source: RigidSE3ExactSourceFacts,
                       budget: ExactBudget | None = None) -> dict[str, Transform]:
    roots = {row.body_id: Transform.from_pose(row.world_pose) for row in source.roots}
    states = {row.joint_id: row for row in source.joint_states}
    children: dict[str, list] = {}
    for joint in source.joints:
        children.setdefault(joint.parent_body_id, []).append(joint)
    world = dict(roots)

    def visit(parent_id: str) -> None:
        for joint in sorted(children.get(parent_id, ()), key=lambda row: row.joint_id):
            _charge(budget, "KINEMATIC")
            if joint.kind == "FIXED":
                motion = Transform()
            elif joint.kind == "PRISMATIC":
                state = states[joint.joint_id]
                assert state.offset_m is not None and joint.axis is not None
                motion = Transform(translation=scale(
                    joint.axis.fractions, state.offset_m.as_fraction, budget,
                ))
            else:
                state = states[joint.joint_id]
                assert state.circle_point is not None and joint.axis is not None
                motion = Transform(rotation=rotation_about_axis(
                    joint.axis.fractions, state.circle_point.cosine.as_fraction,
                    state.circle_point.sine.as_fraction, budget,
                ))
            world[joint.child_body_id] = compose(
                compose(compose(world[parent_id], Transform.from_pose(joint.parent_attachment), budget),
                        motion, budget),
                Transform.from_pose(joint.child_attachment), budget,
            )
            visit(joint.child_body_id)

    for root_id in sorted(roots):
        visit(root_id)
    if set(world) != {body.body_id for body in source.bodies}:
        raise ValueError("kinematic forest did not cover every body")
    return dict(sorted(world.items()))


@dataclass(frozen=True)
class WorldBox:
    primitive_id: str
    body_id: str
    center: Vector
    axes: tuple[Vector, Vector, Vector]
    half_extents: Vector


def world_boxes(source: RigidSE3ExactSourceFacts,
                poses: Mapping[str, Transform],
                budget: ExactBudget | None = None) -> tuple[WorldBox, ...]:
    owners = {primitive: body.body_id for body in source.bodies
              for primitive in body.primitive_ids}
    result = []
    for primitive in source.boxes:
        owner = owners[primitive.primitive_id]
        transform = compose(poses[owner], Transform.from_pose(primitive.local_anchor), budget)
        result.append(WorldBox(
            primitive_id=primitive.primitive_id, body_id=owner,
            center=transform.translation, axes=transpose(transform.rotation),
            half_extents=primitive.half_extents_m.fractions,
        ))
    return tuple(result)


def _projection_radius(box: WorldBox, axis: Vector,
                       budget: ExactBudget | None = None) -> F:
    result = F(0)
    for half, direction in zip(box.half_extents, box.axes, strict=True):
        result = _addf(result, _mulf(half, abs(dot(axis, direction, budget)), budget), budget)
    return result


def obb_interiors_overlap(first: WorldBox, second: WorldBox,
                          budget: ExactBudget | None = None) -> bool:
    """Exact 15-axis SAT; a touching axis separates open interiors."""
    distance = subtract(second.center, first.center, budget)
    axes = (*first.axes, *second.axes,
            *(cross(a, b, budget) for a in first.axes for b in second.axes))
    for axis in axes:
        _charge(budget, "SAT_AXIS")
        if all(value.numerator == 0 for value in axis):
            continue
        separation = abs(dot(axis, distance, budget))
        radius = _addf(_projection_radius(first, axis, budget),
                       _projection_radius(second, axis, budget), budget)
        if _cmpf(separation, radius, budget) >= 0:
            return False
    return True


def collision_free(boxes: tuple[WorldBox, ...],
                   budget: ExactBudget | None = None) -> bool:
    clear = True
    for index, first in enumerate(boxes):
        for second in boxes[index + 1:]:
            if first.body_id == second.body_id:
                continue
            _charge(budget, "COLLISION_PAIR")
            if obb_interiors_overlap(first, second, budget):
                clear = False
    return clear


def box_corners(box: WorldBox, budget: ExactBudget | None = None) -> tuple[Vector, ...]:
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corner = box.center
                for sign, axis, half in zip(
                    (sx, sy, sz), box.axes, box.half_extents, strict=True,
                ):
                    corner = add(corner, scale(axis, _mulf(F(sign), half, budget), budget), budget)
                corners.append(corner)
    return tuple(corners)


def body_in_region(body_id: str, boxes: tuple[WorldBox, ...],
                   region: RigidSE3RegionFact,
                   budget: ExactBudget | None = None) -> bool:
    owned = tuple(box for box in boxes if box.body_id == body_id)
    if not owned:
        raise ValueError("body-region predicate requires body geometry")
    for box in owned:
        for corner in box_corners(box, budget):
            for halfspace in region.halfspaces:
                _charge(budget, "PREDICATE")
                if _cmpf(dot(halfspace.normal.fractions, corner, budget),
                         halfspace.upper_m.as_fraction, budget) > 0:
                    return False
    return True


def frame_offset_equals(first_body_id: str, second_body_id: str,
                        offset: Vector, poses: Mapping[str, Transform],
                        budget: ExactBudget | None = None) -> bool:
    _charge(budget, "PREDICATE")
    return _equal_vec(subtract(poses[second_body_id].translation,
                               poses[first_body_id].translation, budget), offset, budget)


_FACE = {"PX": (0, 1), "NX": (0, -1), "PY": (1, 1),
         "NY": (1, -1), "PZ": (2, 1), "NZ": (2, -1)}


def _face(box: WorldBox, label: str,
          budget: ExactBudget | None = None) -> tuple[Vector, Vector, tuple[Vector, Vector], tuple[F, F]]:
    index, sign = _FACE[label]
    normal = scale(box.axes[index], F(sign), budget)
    center = add(box.center, scale(normal, box.half_extents[index], budget), budget)
    tangents = tuple(box.axes[i] for i in range(3) if i != index)
    extents = tuple(box.half_extents[i] for i in range(3) if i != index)
    return center, normal, tangents, extents  # type: ignore[return-value]


def contact_satisfied(fact: RigidSE3ContactFact, mode: str,
                      boxes: Mapping[str, WorldBox],
                      budget: ExactBudget | None = None) -> bool:
    _charge(budget, "CONTACT_PAIR")
    first = boxes[fact.first_primitive_id]
    second = boxes[fact.second_primitive_id]
    if first.body_id != fact.first_body_id or second.body_id != fact.second_body_id:
        raise ValueError("contact primitive owner differs from immutable fact")
    first_center, normal, first_tangents, first_extents = _face(first, fact.first_face, budget)
    if mode == "RELEASED":
        minimum = _subf(dot(normal, second.center, budget),
                        _projection_radius(second, normal, budget), budget)
        plane = dot(normal, first_center, budget)
        return _cmpf(minimum, _addf(plane, fact.release_gap_m.as_fraction, budget), budget) >= 0
    if mode != "ENGAGED":
        raise ValueError("unknown contact mode")
    second_center, other_normal, second_tangents, second_extents = _face(
        second, fact.second_face, budget,
    )
    if not _equal_vec(other_normal, scale(normal, F(-1), budget), budget):
        return False
    displacement = subtract(second_center, first_center, budget)
    if _cmpf(dot(normal, displacement, budget), F(0), budget) != 0:
        return False
    for axis in (*first_tangents, *second_tangents):
        separation = abs(dot(axis, displacement, budget))
        radius = F(0)
        for half, tangent in (*zip(first_extents, first_tangents, strict=True),
                              *zip(second_extents, second_tangents, strict=True)):
            radius = _addf(radius, _mulf(half, abs(dot(axis, tangent, budget)), budget), budget)
        if _cmpf(separation, radius, budget) > 0:
            return False
    return True


def step_cost(before_roots: Mapping[str, Transform], after_roots: Mapping[str, Transform],
              before_joints: Mapping[str, RigidSE3JointState],
              after_joints: Mapping[str, RigidSE3JointState],
              before_contacts: Mapping[str, str], after_contacts: Mapping[str, str],
              policy: RigidSE3ObjectivePolicy,
              budget: ExactBudget | None = None) -> F:
    if (before_roots.keys() != after_roots.keys()
            or before_joints.keys() != after_joints.keys()
            or before_contacts.keys() != after_contacts.keys()):
        raise ValueError("objective states must have the same complete primary roster")
    total = policy.step_weight.as_fraction
    _charge(budget, "OBJECTIVE_TERM", total)
    for body_id in sorted(before_roots):
        before, after = before_roots[body_id], after_roots[body_id]
        translation = F(0)
        for difference in subtract(after.translation, before.translation, budget):
            translation = _addf(translation, _mulf(difference, difference, budget), budget)
        rotation = F(0)
        for i in range(3):
            for j in range(3):
                difference = _subf(after.rotation[i][j], before.rotation[i][j], budget)
                rotation = _addf(rotation, _mulf(difference, difference, budget), budget)
        if translation and not policy.translation_enabled:
            raise ValueError("translation changed with disabled objective freedom")
        if rotation and not policy.rotation_enabled:
            raise ValueError("rotation changed with disabled objective freedom")
        _charge(budget, "OBJECTIVE_TERM", translation, rotation)
        translation_term = _divf(
            _mulf(policy.translation_weight.as_fraction, translation, budget),
            _mulf(policy.translation_length_m.as_fraction,
                  policy.translation_length_m.as_fraction, budget), budget,
        )
        rotation_term = _mulf(policy.rotation_weight.as_fraction, rotation, budget)
        total = _addf(total, _addf(translation_term, rotation_term, budget), budget)
    for joint_id in sorted(before_joints):
        before, after = before_joints[joint_id], after_joints[joint_id]
        if before.kind != after.kind:
            raise ValueError("joint kind changed in objective states")
        _charge(budget, "OBJECTIVE_TERM")
        if before.kind == "PRISMATIC":
            assert before.offset_m is not None and after.offset_m is not None
            difference = _subf(after.offset_m.as_fraction, before.offset_m.as_fraction, budget)
            if difference and not policy.prismatic_enabled:
                raise ValueError("prismatic motion has disabled objective freedom")
            total = _addf(total, _divf(
                _mulf(policy.prismatic_weight.as_fraction,
                      _mulf(difference, difference, budget), budget),
                _mulf(policy.joint_length_m.as_fraction,
                      policy.joint_length_m.as_fraction, budget), budget,
            ), budget)
        else:
            assert before.circle_point is not None and after.circle_point is not None
            dc = _subf(after.circle_point.cosine.as_fraction,
                       before.circle_point.cosine.as_fraction, budget)
            ds = _subf(after.circle_point.sine.as_fraction,
                       before.circle_point.sine.as_fraction, budget)
            if (dc or ds) and not policy.revolute_enabled:
                raise ValueError("revolute motion has disabled objective freedom")
            total = _addf(total, _mulf(
                policy.revolute_weight.as_fraction,
                _addf(_mulf(dc, dc, budget), _mulf(ds, ds, budget), budget), budget,
            ), budget)
    for contact_id in sorted(before_contacts):
        _charge(budget, "OBJECTIVE_TERM")
        if before_contacts[contact_id] != after_contacts[contact_id]:
            total = _addf(total, policy.contact_weight.as_fraction, budget)
    if budget is not None:
        budget.guard(total)
    return total


def outward_float_bounds(value: F) -> tuple[float, float]:
    """Finite binary64 enclosure of an exact objective, or a typed numeric gap."""
    if value < 0:
        raise ValueError("objective cost must be nonnegative")
    try:
        rounded = float(value)
    except OverflowError as exc:
        raise ExactKernelLimit("NUMERIC_GAP", "objective-transport") from exc
    if not math.isfinite(rounded):
        raise ExactKernelLimit("NUMERIC_GAP", "objective-transport")
    exact = F.from_float(rounded)
    if exact == value:
        return rounded, rounded
    if exact < value:
        upper = math.nextafter(rounded, math.inf)
        if not math.isfinite(upper):
            raise ExactKernelLimit("NUMERIC_GAP", "objective-transport")
        return rounded, upper
    return math.nextafter(rounded, -math.inf), rounded
