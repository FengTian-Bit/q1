#!/usr/bin/env python3
"""Certify separation of every q1 collision pair over its current joint ranges.

The certificate uses invariant world-Y projections wherever possible. Remaining
box/cylinder pairs use support projections and a continuous angular Lipschitz bound.
Same-link composite thigh shapes and calf/footmotor joint interfaces are
explicit permitted overlaps; thigh/calf pairs must pass geometric separation.
It reads all geometry and limits from the supplied
URDF. Unsupported kinematic assumptions or an inconclusive bound produce a nonzero
exit status, never PASS.
This is a geometric certificate, not a dynamics/contact simulation.

Run from any directory:
    python3 tools/check_collision_clearance.py
    python3 tools/check_collision_clearance.py --urdf PATH --output PATH

Dependencies: Python 3 and numpy.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

NUMERIC_GUARD_M = 1e-9


class Unsupported(ValueError):
    """The current model does not satisfy a proof premise."""


def vector(text: str | None, count: int, context: str, default: str | None = None):
    if text is None:
        text = default
    if text is None:
        raise Unsupported(f"{context}: missing vector")
    try:
        values = np.array([float(value) for value in text.split()], dtype=float)
    except ValueError as exc:
        raise Unsupported(f"{context}: invalid numbers") from exc
    if len(values) != count or not np.isfinite(values).all():
        raise Unsupported(f"{context}: expected {count} finite numbers")
    return values


def scalar(text: str | None, context: str, positive: bool = False):
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise Unsupported(f"{context}: missing/invalid scalar") from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise Unsupported(f"{context}: scalar must be finite" + (" and positive" if positive else ""))
    return value


def rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_rpy(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def transform(xyz, rotation=None):
    result = np.eye(4)
    result[:3, 3] = xyz
    if rotation is not None:
        result[:3, :3] = rotation
    return result


def apply(matrix, points):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def origin(element, context):
    item = element.find("origin")
    if item is None:
        return np.zeros(3), np.zeros(3)
    return (vector(item.get("xyz"), 3, context + " xyz", "0 0 0"),
            vector(item.get("rpy"), 3, context + " rpy", "0 0 0"))


def primitive_in_frame(collision, matrix):
    """Read the actual flat box/cylinder in a common rigid frame."""
    full = matrix @ collision["transform"]
    primitive = dict(shape=collision["shape"], center=full[:3, 3], rotation=full[:3, :3])
    if collision["shape"] == "box":
        primitive["half_size"] = collision["size"] / 2
    else:
        axis = full[:3, 2]
        primitive.update(axis=axis / np.linalg.norm(axis), radius=collision["radius"],
                         half_length=collision["length"] / 2)
    return primitive


def box_vertices(primitive):
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    return ((signs * primitive["half_size"]) @ primitive["rotation"].T
            + primitive["center"])


def support_extent(primitive, normals):
    """Exact centered support function of the finite primitive."""
    if primitive["shape"] == "box":
        return np.abs(normals @ primitive["rotation"]) @ primitive["half_size"]
    axial = normals @ primitive["axis"]
    radial = np.sqrt(np.maximum(0.0, np.sum(normals * normals, axis=-1) - axial * axial))
    return primitive["half_length"] * np.abs(axial) + primitive["radius"] * radial


def segment_distance(first, second):
    """Shortest distance between finite cylinder centerline segments."""
    p0 = first["center"] - first["half_length"] * first["axis"]
    p1 = first["center"] + first["half_length"] * first["axis"]
    q0 = second["center"] - second["half_length"] * second["axis"]
    q1 = second["center"] + second["half_length"] * second["axis"]
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c = float(u @ u), float(u @ v), float(v @ v)
    d, e = float(u @ w), float(v @ w)
    den = a * c - b * b
    s = (b * e - c * d) / den if den > 1e-20 else 0.0
    t = (a * e - b * d) / den if den > 1e-20 else 0.0
    s = min(1.0, max(0.0, s))
    t = (b * s + e) / c if c > 1e-20 else 0.0
    if t < 0.0:
        t = 0.0; s = min(1.0, max(0.0, -d / a if a > 1e-20 else 0.0))
    elif t > 1.0:
        t = 1.0; s = min(1.0, max(0.0, (b - d) / a if a > 1e-20 else 0.0))
    t = min(1.0, max(0.0, (b * s + e) / c if c > 1e-20 else 0.0))
    s = min(1.0, max(0.0, (b * t - d) / a if a > 1e-20 else 0.0))
    return float(np.linalg.norm((p0 + s * u) - (q0 + t * v)))


def finite_cylinder_axis_lower(first, second):
    if first["shape"] != "cylinder" or second["shape"] != "cylinder":
        return -math.inf
    return segment_distance(first, second) - first["radius"] - second["radius"]


def projection_separation_lower(first, second):
    """Valid distance lower bound from separating projection intervals.

    For two OBBs the face normals and edge cross products are the complete SAT
    axes. For a box/cylinder they remain valid separating directions, without
    claiming that this finite direction list is a complete collision test.
    A failure to find positive separation is therefore inconclusive, never PASS.
    """
    axes_first = first["rotation"].T if first["shape"] == "box" else first["axis"][None, :]
    axes_second = second["rotation"].T if second["shape"] == "box" else second["axis"][None, :]
    crosses = np.cross(axes_first[:, None, :], axes_second[None, :, :]).reshape(-1, 3)
    candidates = [axes_first, axes_second, crosses,
                  (second["center"] - first["center"])[None, :]]
    if {first["shape"], second["shape"]} == {"box", "cylinder"}:
        box = first if first["shape"] == "box" else second
        cylinder = second if first["shape"] == "box" else first
        vertices = box_vertices(box)
        t = np.clip((vertices - cylinder["center"]) @ cylinder["axis"],
                    -cylinder["half_length"], cylinder["half_length"])
        candidates.append(vertices - cylinder["center"] - t[:, None] * cylinder["axis"])
    if first["shape"] == "cylinder" and second["shape"] == "cylinder":
        # Additional finite-cylinder end-cap-to-axis normals. They provide
        # separating directions for skew/near-parallel cylinder pairs while
        # remaining valid support projections (failure is still inconclusive).
        for source, target in ((first, second), (second, first)):
            for endpoint in (source["center"] - source["half_length"] * source["axis"],
                             source["center"] + source["half_length"] * source["axis"]):
                t = np.clip(float((endpoint - target["center"]) @ target["axis"]),
                            -target["half_length"], target["half_length"])
                candidates.append((endpoint - target["center"] - t * target["axis"])[None, :])
        first_endpoints = (first["center"] - first["half_length"] * first["axis"],
                           first["center"] + first["half_length"] * first["axis"])
        second_endpoints = (second["center"] - second["half_length"] * second["axis"],
                            second["center"] + second["half_length"] * second["axis"])
        candidates.extend((p - q)[None, :] for p in first_endpoints for q in second_endpoints)
    normals = np.concatenate(candidates)
    norms = np.linalg.norm(normals, axis=1)
    normals = normals[norms > 1e-14] / norms[norms > 1e-14, None]
    if not len(normals):
        return 0.0
    gaps = (np.abs(normals @ (second["center"] - first["center"]))
            - support_extent(first, normals) - support_extent(second, normals))
    support_gap = float(np.max(gaps))
    if first["shape"] == "cylinder" and second["shape"] == "cylinder":
        support_gap = max(support_gap, finite_cylinder_axis_lower(first, second))
    return max(0.0, support_gap)


class Model:
    def __init__(self, path):
        self.path = path
        self.raw = path.read_bytes()
        root = ET.fromstring(self.raw)
        if root.tag != "robot":
            raise Unsupported("Root XML element must be robot")
        self.name = root.get("name")
        self.links = {}
        for link in root.findall("link"):
            name = link.get("name")
            if not name or name in self.links:
                raise Unsupported("Link names must be nonempty and unique")
            self.links[name] = link
        if not self.links:
            raise Unsupported("URDF has no links")
        self.joints, self.parent = {}, {}
        for item in root.findall("joint"):
            name, kind = item.get("name"), item.get("type")
            if not name or name in self.joints:
                raise Unsupported("Joint names must be nonempty and unique")
            if kind not in ("fixed", "revolute", "continuous"):
                raise Unsupported(f"{name}: only fixed, revolute or continuous joints are supported")
            parent_item, child_item = item.find("parent"), item.find("child")
            if parent_item is None or child_item is None:
                raise Unsupported(f"{name}: missing parent/child")
            parent, child = parent_item.get("link"), child_item.get("link")
            if parent not in self.links or child not in self.links or child in self.parent:
                raise Unsupported(f"{name}: invalid parent/child tree")
            xyz, rpy = origin(item, name)
            # Exact numerical zeros are a premise of invariant global-Y bounds.
            if np.any(rpy != 0):
                raise Unsupported(f"{name}: joint origin rpy must be exactly 0 0 0 for this certificate")
            axis, limits = None, None
            if kind != "fixed":
                axis_item = item.find("axis")
                axis = vector(axis_item.get("xyz") if axis_item is not None else None,
                              3, name + " axis", "1 0 0")
                if axis[0] != 0 or axis[2] != 0 or abs(axis[1]) != 1:
                    raise Unsupported(f"{name}: joint axis must be exactly 0 +/-1 0")
                if kind == "revolute":
                    limit_item = item.find("limit")
                    if limit_item is None:
                        raise Unsupported(f"{name}: revolute joint needs finite limits")
                    lower = scalar(limit_item.get("lower"), name + " lower")
                    upper = scalar(limit_item.get("upper"), name + " upper")
                    if lower > upper:
                        raise Unsupported(f"{name}: lower limit exceeds upper")
                    limits = (lower, upper)
            self.joints[name] = dict(name=name, type=kind, parent=parent, child=child,
                                     xyz=xyz, axis=axis, limits=limits)
            self.parent[child] = name
        roots = set(self.links) - set(self.parent)
        if len(roots) != 1:
            raise Unsupported("URDF must have exactly one root link")
        self.root = roots.pop()
        for link in self.links:
            self.path_to(self.root, link)  # Also rejects disconnected cycles.
        self.collisions = []
        for link, item in self.links.items():
            for index, collision in enumerate(item.findall("collision")):
                label = collision.get("name") or f"{link}_collision_{index}"
                key = f"{link}/{label}"
                xyz, rpy = origin(collision, key)
                rotation = rotation_rpy(rpy)
                geometry = collision.find("geometry")
                if geometry is None or len(geometry) != 1:
                    raise Unsupported(f"{key}: exactly one geometry element is required")
                shape = geometry[0]
                record = dict(id=key, name=label, link=link, shape=shape.tag,
                              xyz=xyz, rpy=rpy, transform=transform(xyz, rotation))
                if shape.tag == "box":
                    size = vector(shape.get("size"), 3, key + " size")
                    if np.any(size <= 0):
                        raise Unsupported(f"{key}: box dimensions must be positive")
                    record["size"] = size
                    extent_y = float(np.abs(rotation[1]) @ (size / 2))
                elif shape.tag == "cylinder":
                    radius = scalar(shape.get("radius"), key + " radius", True)
                    length = scalar(shape.get("length"), key + " length", True)
                    axis = rotation[:, 2]
                    record.update(radius=radius, length=length,
                                  endpoints=apply(record["transform"],
                                                  [[0, 0, -length / 2], [0, 0, length / 2]]))
                    # Support function of a finite cylinder along world Y.
                    radial_y = float(np.linalg.norm(rotation[1, :2]))
                    extent_y = length / 2 * abs(float(axis[1])) + radius * radial_y
                else:
                    raise Unsupported(f"{key}: only box and cylinder collisions are supported")
                center_world = apply(self.relative(self.root, link, {}), [xyz])[0]
                record["y_interval"] = (float(center_world[1] - extent_y),
                                        float(center_world[1] + extent_y))
                self.collisions.append(record)
        if not self.collisions:
            raise Unsupported("No collision geometry found")
        if len({c["id"] for c in self.collisions}) != len(self.collisions):
            raise Unsupported("Collision identifiers must be unique within each link")

    def path_to(self, ancestor, link):
        path, visited = [], set()
        while link != ancestor:
            if link in visited or link not in self.parent:
                raise Unsupported("Disconnected/cyclic tree or invalid ancestor")
            visited.add(link)
            name = self.parent[link]
            path.append(name)
            link = self.joints[name]["parent"]
        return list(reversed(path))

    def common_ancestor(self, first, second):
        ancestors = {first}
        while first in self.parent:
            first = self.joints[self.parent[first]]["parent"]
            ancestors.add(first)
        while second not in ancestors:
            if second not in self.parent:
                raise Unsupported("Links have no common ancestor")
            second = self.joints[self.parent[second]]["parent"]
        return second

    def relative(self, ancestor, link, angles):
        matrix = np.eye(4)
        for name in self.path_to(ancestor, link):
            joint = self.joints[name]
            matrix = matrix @ transform(joint["xyz"])
            if joint["type"] != "fixed":
                matrix = matrix @ transform(np.zeros(3), rotation_y(
                    float(joint["axis"][1]) * angles.get(name, 0.0)))
        return matrix

    def parameter_summary(self):
        result = []
        for collision in self.collisions:
            entry = {key: collision[key] for key in ("id", "link", "shape")}
            entry.update(origin_xyz_m=collision["xyz"].tolist(),
                         origin_rpy_rad=collision["rpy"].tolist(),
                         invariant_global_y_interval_m=list(collision["y_interval"]))
            if collision["shape"] == "box":
                entry["box_size_m"] = collision["size"].tolist()
            else:
                entry.update(radius_m=collision["radius"], length_m=collision["length"])
            result.append(entry)
        return result


def reference_overlap_reason(first, second):
    """Return the narrow, documented composite/joint overlap allowance, if any."""
    if first["link"] == second["link"]:
        return "same-link composite collision primitives"
    links = {first["link"], second["link"]}
    if len(links) != 2:
        return None
    side = first["link"][0]
    if side not in "LR" or second["link"][0] != side:
        return None
    if links == {f"{side}_calf", f"{side}_footmotor"}:
        return "same-side calf/footmotor connection overlap following the reference primitive logic"
    return None


def certify_pair(model, first, second, interval_error):
    result = dict(first=first["id"], second=second["id"])
    overlap_reason = reference_overlap_reason(first, second)
    if overlap_reason:
        result.update(method="permitted_reference_overlap", certified=True,
                      permitted_overlap=True,
                      conservative_gap_m=None,
                      domain=overlap_reason)
        return result
    a0, a1 = first["y_interval"]
    b0, b1 = second["y_interval"]
    y_gap = max(a0 - b1, b0 - a1)
    if y_gap > NUMERIC_GUARD_M:
        result.update(method="invariant_global_y_intervals", certified=True,
                      conservative_gap_m=y_gap - NUMERIC_GUARD_M,
                      raw_y_gap_m=y_gap,
                      first_y_interval_m=[a0, a1], second_y_interval_m=[b0, b1],
                      domain="all configured revolute limits and all continuous-joint angles")
        return result
    ancestor = model.common_ancestor(first["link"], second["link"])
    first_path = model.path_to(ancestor, first["link"])
    second_path = model.path_to(ancestor, second["link"])
    active = [name for name in first_path + second_path if model.joints[name]["type"] != "fixed"]
    first_zero = primitive_in_frame(first, model.relative(ancestor, first["link"], {}))
    second_zero = primitive_in_frame(second, model.relative(ancestor, second["link"], {}))
    result["common_ancestor"] = ancestor
    eliminated, symmetry_guard = [], 0.0
    # A coaxial finite circular cylinder is invariant under its wheel rotation.
    # Decimal pi/2 gives a tiny tilt in floating-point/URDF trigonometry. Bound
    # that deviation explicitly rather than silently treating it as exact zero.
    for name in list(active):
        joint = model.joints[name]
        if joint["type"] != "continuous":
            continue
        first_moves = name in first_path
        collision = first if first_moves else second
        path = first_path if first_moves else second_path
        after = path[path.index(name) + 1:]
        if collision["shape"] != "cylinder" or any(model.joints[j]["type"] != "fixed" for j in after):
            raise Unsupported(f"{name}: continuous angle elimination requires a downstream cylinder with only fixed transforms after it")
        primitive = first_zero if first_moves else second_zero
        prefix = model.relative(ancestor, joint["parent"], {}) @ transform(joint["xyz"])
        axis_origin, axis = prefix[:3, 3], prefix[:3, :3] @ joint["axis"]
        center_offset = primitive["center"] - axis_origin
        radial_offset = float(np.linalg.norm(np.cross(center_offset, axis)))
        sin_theta = float(np.linalg.norm(np.cross(primitive["axis"], axis)))
        abs_cos_theta = min(1.0, abs(float(primitive["axis"] @ axis)))
        if radial_offset > 1e-10 or sin_theta > 1e-10:
            raise Unsupported(f"{name}: cylinder is not coaxial with the continuous joint")
        # H(C(q), C(0)) <= 2*H(C(0), ideal coaxial C). Aligning the
        # cylinder to the ideal axis moves points by <= 2*R*sin(theta/2).
        sin_half_theta = sin_theta / math.sqrt(2 * (1 + abs_cos_theta))
        radius_of_bound = math.hypot(primitive["radius"], primitive["half_length"])
        error = 2 * (radial_offset + 2 * radius_of_bound * sin_half_theta)
        symmetry_guard += error
        eliminated.append(dict(joint=name, reason="coaxial circular-cylinder rotational symmetry",
                               center_radial_offset_m=radial_offset, axis_sin_tilt=sin_theta,
                               all_angle_hausdorff_guard_m=error))
        active.remove(name)
    result.update(eliminated_continuous_joint_symmetries=eliminated,
                  symmetry_guard_m=symmetry_guard)
    if len(active) > 1:
        raise Unsupported(f"{first['id']} / {second['id']}: more than one relative active joint remains after symmetry elimination")
    if not active:
        lower = projection_separation_lower(first_zero, second_zero) - symmetry_guard - NUMERIC_GUARD_M
        result.update(method="fixed_relative_support_projections", certified=lower > 0,
                      conservative_gap_m=lower,
                      domain="fixed relative transform after any coaxial wheel symmetries; all upstream joint angles cancel")
        return result
    name = active[0]
    joint = model.joints[name]
    if joint["type"] != "revolute" or joint["limits"] is None:
        raise Unsupported(f"{name}: remaining angular certificate requires a bounded revolute joint")
    first_moves = name in first_path
    moving = first if first_moves else second
    moving_zero = first_zero if first_moves else second_zero
    prefix = model.relative(ancestor, joint["parent"], {}) @ transform(joint["xyz"])
    axis_origin, axis_unit = prefix[:3, 3], prefix[:3, :3] @ joint["axis"]
    if moving_zero["shape"] == "box":
        # Radius to a fixed axis is convex: its maximum over a box occurs at a vertex.
        speed_bound = max(float(np.linalg.norm(np.cross(point - axis_origin, axis_unit)))
                          for point in box_vertices(moving_zero))
        speed_reason = "maximum box-vertex radius from the relative joint axis"
    else:
        center, direction = moving_zero["center"], moving_zero["axis"]
        endpoints = [center - moving_zero["half_length"] * direction,
                     center + moving_zero["half_length"] * direction]
        speed_bound = max(float(np.linalg.norm(np.cross(point - axis_origin, axis_unit)))
                          for point in endpoints) + moving_zero["radius"]
        speed_reason = "maximum cylinder-axis endpoint radius plus cylinder radius (conservative)"
    lower_q, upper_q = joint["limits"]
    width = upper_q - lower_q
    panel_count = max(1, math.ceil(speed_bound * width / (2 * interval_error)))
    if panel_count > 1_000_000:
        raise Unsupported(f"{name}: requested certificate needs over one million angular panels")
    panel_width = width / panel_count
    angular_penalty = speed_bound * panel_width / 2
    best_lower, worst_angle, midpoint_lower_at_worst = math.inf, None, None
    for index in range(panel_count):
        angle = lower_q + (index + 0.5) * panel_width
        angles = {name: angle}
        a = primitive_in_frame(first, model.relative(ancestor, first["link"], angles))
        b = primitive_in_frame(second, model.relative(ancestor, second["link"], angles))
        midpoint_lower = projection_separation_lower(a, b)
        bound = midpoint_lower - angular_penalty - symmetry_guard - NUMERIC_GUARD_M
        if bound < best_lower:
            best_lower, worst_angle, midpoint_lower_at_worst = bound, angle, midpoint_lower
    result.update(method="continuous_angular_lipschitz_support_projections",
                  certified=best_lower > 0, conservative_gap_m=best_lower,
                  relative_joint=name, relative_joint_range_rad=[lower_q, upper_q],
                  moving_collision=moving["id"], point_speed_bound_m_per_rad=speed_bound,
                  speed_bound_reason=speed_reason,
                  closed_cover_panel_count=panel_count, panel_width_rad=panel_width,
                  maximum_midpoint_to_panel_bound_m=angular_penalty,
                  worst_panel_midpoint_rad=worst_angle,
                  worst_panel_projection_lower_m=midpoint_lower_at_worst,
                  inequality="gap(q) >= support_projection_lower(midpoint) - speed_bound * panel_width / 2 - symmetry_guard - numeric_guard",
                  domain="closed panels cover the entire relative-joint interval; all upstream angles cancel")
    return result


def main():
    package = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=package / "urdf" / "双轮足机器人.urdf")
    parser.add_argument("--output", type=Path, default=package / "docs" / "q1_collision_clearance.json")
    parser.add_argument("--max-interval-error", type=float, default=0.00005,
                        help="maximum continuous-angle penalty in metres (default: 0.00005)")
    args = parser.parse_args()
    report = dict(generated_at=datetime.now().astimezone().isoformat(),
                  urdf=str(args.urdf.resolve()), output=str(args.output.resolve()),
                  not_dynamic_simulation=True,
                  purpose="Continuous geometric separation certificate for every collision-primitive pair",
                  numeric_guard_m=NUMERIC_GUARD_M,
                  proof_premises=["A connected rooted URDF tree",
                                  "Every joint origin rpy is exactly zero",
                                  "Every active joint axis is exactly positive or negative Y",
                                  "All collision primitives are finite positive boxes or cylinders",
                                  "Same-link composite shapes and same-side calf/footmotor interfaces are explicit permitted overlaps; all thigh/calf pairs and calf/wheel pairs require geometric separation certification",
                                  "After coaxial circular-wheel symmetry elimination, every unresolved Y-projection pair has at most one relative bounded revolute joint"],
                  pairs=[])
    code = 2
    try:
        if not math.isfinite(args.max_interval_error) or args.max_interval_error <= 0:
            raise Unsupported("--max-interval-error must be finite and positive")
        model = Model(args.urdf.resolve())
        report.update(robot_name=model.name, root_link=model.root,
                      urdf_sha256=hashlib.sha256(model.raw).hexdigest(),
                      link_count=len(model.links), collision_count=len(model.collisions),
                      required_pair_count=len(model.collisions) * (len(model.collisions) - 1) // 2,
                      collision_parameters=model.parameter_summary(),
                      active_joint_domains=[dict(name=joint["name"], type=joint["type"],
                                                 axis=joint["axis"].tolist(),
                                                 range_rad=list(joint["limits"]) if joint["limits"] is not None else None,
                                                 range_description="all real angles" if joint["type"] == "continuous" else "closed configured limit interval")
                                            for joint in model.joints.values() if joint["type"] != "fixed"],
                      maximum_requested_angular_penalty_m=args.max_interval_error)
        for first, second in itertools.combinations(model.collisions, 2):
            try:
                pair = certify_pair(model, first, second, args.max_interval_error)
            except Unsupported as exc:
                pair = dict(first=first["id"], second=second["id"], method="unsupported",
                            certified=False, conservative_gap_m=None, explanation=str(exc))
            report["pairs"].append(pair)
        report["checked_pair_count"] = len(report["pairs"])
        certified = (report["checked_pair_count"] == report["required_pair_count"]
                     and all(pair["certified"] for pair in report["pairs"]))
        permitted = sum(1 for pair in report["pairs"] if pair.get("permitted_overlap"))
        report["permitted_overlap_pair_count"] = permitted
        report["status"] = "passed_with_permitted_reference_overlaps" if certified and permitted else ("passed" if certified else "not_certified")
        report["certified_all_current_joint_ranges"] = certified
        measurable = [pair for pair in report["pairs"] if pair.get("conservative_gap_m") is not None]
        if measurable:
            minimum = min(measurable, key=lambda pair: pair["conservative_gap_m"])
            report["minimum_conservative_gap_m"] = minimum["conservative_gap_m"]
            report["minimum_pair"] = [minimum["first"], minimum["second"]]
        code = 0 if certified else 1
        if not certified:
            report["uncertified_pair_count"] = sum(not pair["certified"] for pair in report["pairs"])
            report["uncertified_pairs"] = [dict(first=pair["first"], second=pair["second"],
                                                method=pair["method"],
                                                explanation=pair.get("explanation"))
                                           for pair in report["pairs"] if not pair["certified"]]
            report["explanation"] = "At least one pair has a nonpositive lower bound or an unsupported proof premise. This is not a collision claim; separation has not been certified for those pairs."
    except (Unsupported, OSError, ET.ParseError) as exc:
        report.update(status="unsupported", certified_all_current_joint_ranges=False,
                      checked_pair_count=len(report["pairs"]), explanation=str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in
                      ("status", "collision_count", "checked_pair_count", "required_pair_count",
                       "minimum_conservative_gap_m", "minimum_pair", "explanation", "output")},
                     ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
