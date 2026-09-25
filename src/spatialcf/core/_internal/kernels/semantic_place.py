"""Exact closed placement domains minus open box collisions.

Each axis is partitioned into singleton boundaries and open intervals. On each
Cartesian stratum every obstacle comparison has constant sign. The feasible
set is closed, so the closure of a feasible stratum is feasible; coordinate
clamping minimizes squared distance on that closure. All strata, including
lines and points, participate in the global minimum and empty-domain proof.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction as F
from functools import cmp_to_key
from itertools import product

from spatialcf.core._internal.kernels.rect import (
    ExactAxisAlignedRectV2, RectCoordinateSpaceV2,
    _fraction_to_float_floor, _fraction_to_float_ceil, RectKernelProjectionErrorV2,
)
from spatialcf.domain.semantic_place import (
    SemanticPlaceCoverage, SemanticPlaceExactBox, SemanticPlaceLedger,
    SemanticPlaceLimits, SemanticPlaceStratum, SemanticPlaceTarget, dyadic,
)


class PlacementLimit(ValueError):
    def __init__(self, reason: str, item: str, stage: str | None = None):
        super().__init__(reason + ":" + item)
        self.reason = reason
        self.item = item
        self.stage = stage


@dataclass
class Budget:
    """A deterministic ledger of exact predicates and numeric range checks.

    A charged operation is one bounded scalar/box predicate or numeric scan;
    it is never a claim to count Python instructions or wall-clock time.
    A stratum is committed only after its classification/minimum is complete.
    """
    limits: SemanticPlaceLimits
    stage: str = "SOLVE"
    operations: int = 0
    completed: list[str] = field(default_factory=list)
    current: str = "item:start"
    audit: Budget | None = None
    replay_operations: int = 0
    peak_numeric_bits: int = 0

    def charge(self, count: int = 1) -> None:
        self.replay_operations += count
        if self.audit is not None:
            self.audit.current = self.current
            self.audit.charge(count)
        cap = getattr(self.limits, self.stage.lower() + "_operations")
        if self.operations + count > cap:
            raise PlacementLimit("RESOURCE_LIMIT", self.current, self.stage)
        self.operations += count

    def numbers(self, *values: F) -> None:
        self.charge()
        for v in values:
            bits=max(abs(v.numerator).bit_length(), v.denominator.bit_length())
            if bits > self.limits.numeric_bits:
                raise PlacementLimit("NUMERIC_GAP", self.current, self.stage)
            # High-water mark of accepted representations; rejected numeric
            # work is located by the first-unprocessed source item instead.
            self.peak_numeric_bits=max(self.peak_numeric_bits,bits)
            if self.audit is not None:
                self.audit.peak_numeric_bits=max(self.audit.peak_numeric_bits,bits)

    def ledger(self, error: PlacementLimit | None = None) -> SemanticPlaceLedger:
        return SemanticPlaceLedger(stage=self.stage, operations=self.operations,
            replay_operations=self.replay_operations,peak_numeric_bits=self.peak_numeric_bits,
            completed_items=tuple(self.completed), first_unprocessed_item=error.item if error else None,
            reason=error.reason if error else "NONE")


def box(lower, upper) -> SemanticPlaceExactBox:
    return SemanticPlaceExactBox(lower=tuple(dyadic(F(x)) for x in lower),upper=tuple(dyadic(F(x)) for x in upper))


def bounds(b: SemanticPlaceExactBox):
    return tuple(v.as_fraction for v in b.lower), tuple(v.as_fraction for v in b.upper)


def translate(b: SemanticPlaceExactBox, xyz: tuple[F, F, F], budget: Budget) -> SemanticPlaceExactBox:
    low, high = bounds(b)
    result_low = tuple(a + p for a,p in zip(low,xyz,strict=True))
    result_high = tuple(a + p for a,p in zip(high,xyz,strict=True))
    budget.numbers(*low,*high,*xyz,*result_low,*result_high)
    return box(result_low,result_high)


def contains(outer: SemanticPlaceExactBox, inner: SemanticPlaceExactBox, budget: Budget) -> bool:
    budget.charge()
    a,b=bounds(outer); c,d=bounds(inner)
    return all(x <= y and z <= w for x,y,z,w in zip(a,c,d,b,strict=True))


def interiors_overlap(a: SemanticPlaceExactBox, b: SemanticPlaceExactBox, budget: Budget) -> bool:
    budget.charge()
    al,au=bounds(a); bl,bu=bounds(b)
    return all(max(x,y)<min(z,w) for x,y,z,w in zip(al,bl,au,bu,strict=True))


def compile_target(*, target_id, support_surface_id, subject, support, height,
                   authorized_xy, authorized_z, obstacles, cavity=None,
                   margins=(F(0),F(0),F(0)), budget=None) -> SemanticPlaceTarget:
    budget = budget or Budget(SemanticPlaceLimits(),"COMPILE")
    budget.current = "target:" + target_id
    lo,hi=bounds(subject)
    lx,ly,ux,uy=support
    ax,ay,bx,by=authorized_xy
    m,c,t=margins
    z=height-lo[2]
    x0=max(lx+m-lo[0],ax); y0=max(ly+m-lo[1],ay)
    x1=min(ux-m-hi[0],bx); y1=min(uy-m-hi[1],by)
    budget.numbers(*lo,*hi,*support,height,*authorized_xy,*authorized_z,*margins,z,x0,y0,x1,y1)
    reason="NONE"
    if not authorized_z[0] <= z <= authorized_z[1]:
        reason="Z_BOUNDS"
    if cavity is not None:
        cl,cu=bounds(cavity)
        if cl[2] != height:
            raise ValueError("cavity bottom must equal support height")
        x0=max(x0,cl[0]+c-lo[0]); y0=max(y0,cl[1]+c-lo[1])
        x1=min(x1,cu[0]-c-hi[0]); y1=min(y1,cu[1]-c-hi[1])
        budget.numbers(*cl,*cu,x0,y0,x1,y1,z+hi[2],cu[2]-t)
        if reason=="NONE" and z+hi[2] > cu[2]-t:
            reason="CAVITY_TOP"
    if reason=="NONE" and (x0>x1 or y0>y1):
        reason="XY_EMPTY"
    forbidden=[]; obstacle_ids=[]
    if reason=="NONE":
        # Reuse the established closed world-XY topology representation.
        ExactAxisAlignedRectV2.from_fraction_bounds(coordinate_space=RectCoordinateSpaceV2.WORLD_XY_M,
            min_x_m=x0,min_y_m=y0,max_x_m=x1,max_y_m=y1)
        for identifier,obstacle in sorted(obstacles,key=lambda row:row[0]):
            ol,ou=bounds(obstacle)
            budget.numbers(*ol,*ou,z+lo[2],z+hi[2])
            if max(z+lo[2],ol[2]) < min(z+hi[2],ou[2]):
                rectangle=(ol[0]-hi[0],ol[1]-hi[1],ou[0]-lo[0],ou[1]-lo[1])
                budget.numbers(*rectangle)
                forbidden.append(tuple(dyadic(v) for v in rectangle))
                obstacle_ids.append(identifier)
    return SemanticPlaceTarget(target_id=target_id,support_surface_id=support_surface_id,height=dyadic(z),
        xy_bounds=tuple(dyadic(v) for v in (x0,y0,x1,y1)) if reason=="NONE" else None,
        forbidden_open_rectangles=tuple(forbidden),obstacle_geometry_ids=tuple(obstacle_ids),infeasibility_reason=reason)


def _axis_parts(lower,upper,coordinates,budget):
    # Count filtering, ordering and deduplication before constructing strata.
    # Avoid a set: rational hash collisions must not hide unbounded comparisons.
    points=[lower,upper]
    for value in coordinates:
        budget.charge(2)
        if lower <= value <= upper:
            points.append(value)
    def compare(left,right):
        budget.charge(2)
        return (left > right) - (left < right)
    ordered=sorted(points,key=cmp_to_key(compare))
    points=[]
    for value in ordered:
        budget.charge()
        if not points or points[-1]!=value:
            points.append(value)
    return tuple(part for i,p in enumerate(points)
        for part in (((p,p),) if i==len(points)-1 else ((p,p),(p,points[i+1]))))


def arrange_target(target: SemanticPlaceTarget, budget: Budget):
    """Return lazy Cartesian strata, checking size before cross-product allocation."""
    if target.xy_bounds is None:
        return (), (), ()
    x0,y0,x1,y1=(v.as_fraction for v in target.xy_bounds)
    obstacles=tuple(tuple(v.as_fraction for v in row) for row in target.forbidden_open_rectangles)
    budget.numbers(x0,y0,x1,y1,*[v for r in obstacles for v in r])
    xs=_axis_parts(x0,x1,(v for r in obstacles for v in (r[0],r[2])),budget)
    ys=_axis_parts(y0,y1,(v for r in obstacles for v in (r[1],r[3])),budget)
    if len(xs)*len(ys) > budget.limits.max_strata:
        raise PlacementLimit("RESOURCE_LIMIT",budget.current)
    return xs,ys,obstacles


def _blocked(x,y,obstacles,budget):
    for i,(x0,y0,x1,y1) in enumerate(obstacles):
        budget.charge()
        if x0 < x < x1 and y0 < y < y1:
            return i
    return None


def solve_targets(targets, before, limits, *, stage="SOLVE", budget=None) -> SemanticPlaceCoverage:
    targets=tuple(sorted(targets,key=lambda target:target.target_id))
    if len({t.target_id for t in targets}) != len(targets):
        raise ValueError("duplicate target")
    budget=budget or Budget(limits,stage)
    strata=[]; winner=None; error=None
    try:
        budget.numbers(*before)
        for ti,target in enumerate(targets):
            budget.current="target:"+target.target_id
            if ti >= limits.max_targets:
                raise PlacementLimit("RESOURCE_LIMIT",budget.current)
            xs,ys,obstacles=arrange_target(target,budget)
            if len(strata)+len(xs)*len(ys)>limits.max_strata:
                raise PlacementLimit("RESOURCE_LIMIT",budget.current)
            if not xs:
                budget.completed.append(budget.current+":empty")
            for si,(xpart,ypart) in enumerate(product(xs,ys)):
                budget.current=f"stratum:{target.target_id}:{si}"
                x0,x1=xpart; y0,y1=ypart; z=target.height.as_fraction
                rx=(x0+x1)/2; ry=(y0+y1)/2
                budget.numbers(rx,ry,z)
                blocked=_blocked(rx,ry,obstacles,budget)
                minimum=None; cost=None
                if blocked is None:
                    x=max(x0,min(before[0],x1)); y=max(y0,min(before[1],y1))
                    dx=x-before[0]; dy=y-before[1]; dz=z-before[2]
                    cx=dx*dx; cy=dy*dy; cz=dz*dz
                    cost=cx+cy+cz
                    budget.numbers(x,y,dx,dy,dz,cx,cy,cz,cx+cy,cost)
                    if _blocked(x,y,obstacles,budget) is not None:
                        raise ValueError("feasible closure invariant violated")
                    minimum=(x,y,z)
                    candidate=(cost,target.target_id,x,y,z)
                    if winner is None or candidate<winner:
                        winner=candidate
                row=SemanticPlaceStratum(target_id=target.target_id,x=tuple(map(dyadic,xpart)),y=tuple(map(dyadic,ypart)),
                    dimension=int(x0!=x1)+int(y0!=y1),obstacle_index=blocked,
                    minimum_xyz=tuple(map(dyadic,minimum)) if minimum is not None else None,
                    minimum_cost=dyadic(cost) if cost is not None else None)
                strata.append(row)
                budget.completed.append(budget.current)
    except PlacementLimit as caught:
        if caught.stage == "CHECK" and budget.stage != "CHECK":
            raise
        error=caught
    # A prefix minimum is never presented as the global winner.
    if error is not None:
        winner=None
    return SemanticPlaceCoverage.seal(target_ids=tuple(t.target_id for t in targets),strata=tuple(strata),
        winner_target_id=winner[1] if winner else None,winner_xyz=tuple(map(dyadic,winner[2:])) if winner else None,
        winner_cost=dyadic(winner[0]) if winner else None,ledger=budget.ledger(error))


def transport_winner(coverage: SemanticPlaceCoverage):
    """Exact coordinate transport and finite outward cost bounds, or numeric gap."""
    if coverage.winner_xyz is None or coverage.winner_cost is None:
        return None
    try:
        xyz=tuple(float(v.as_fraction) for v in coverage.winner_xyz)
        if not all(math.isfinite(f) and F(f)==v.as_fraction for f,v in zip(xyz,coverage.winner_xyz,strict=True)):
            return None
        cost=coverage.winner_cost.as_fraction
        lower=_fraction_to_float_floor(cost); upper=_fraction_to_float_ceil(cost)
        if not (math.isfinite(lower) and math.isfinite(upper) and 0<=lower<=upper):
            return None
        return tuple(0.0 if x==0.0 else x for x in xyz),lower,upper
    except (OverflowError,RectKernelProjectionErrorV2):
        return None


def covered_rectangle(rectangle, coverings, budget: Budget) -> bool:
    """Closed top-face union coverage, including every boundary stratum."""
    x0,y0,x1,y1=rectangle
    budget.numbers(*rectangle,*[v for r in coverings for v in r])
    xs=_axis_parts(x0,x1,(v for r in coverings for v in (r[0],r[2])),budget)
    ys=_axis_parts(y0,y1,(v for r in coverings for v in (r[1],r[3])),budget)
    if len(xs)*len(ys)>budget.limits.max_strata:
        raise PlacementLimit("RESOURCE_LIMIT",budget.current)
    for (a,b),(c,d) in product(xs,ys):
        x=(a+b)/2; y=(c+d)/2
        budget.numbers(x,y)
        covered=False
        for l,r,u,v in coverings:
            budget.charge()
            if l<=x<=u and r<=y<=v:
                covered=True
                break
        if not covered:
            return False
    return True
