"""Static Q1 tree/inertia/mesh/initial-ground-clearance checks; no simulator.

python tools/check_q1_model.py --output docs/q1_model_check.json
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def vector(text, default="0 0 0"):
    return np.fromstring(text or default, sep=" ")


def rotation(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def origin(node):
    result = np.eye(4)
    element = node.find("origin")
    if element is not None:
        result[:3, :3] = rotation(vector(element.get("rpy")))
        result[:3, 3] = vector(element.get("xyz"))
    return result


def stl_vertices(path):
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + 50*count:
        raise ValueError("Expected binary STL: {}".format(path))
    dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    return np.frombuffer(raw, dtype=dtype, offset=84, count=count)["vertices"].reshape(-1, 3)


def check(urdf, mesh_root=None, *, initial_angles=None, initial_height=0.402):
    robot = ET.parse(urdf).getroot()
    links = {link.get("name"): link for link in robot.findall("link")}
    joints = robot.findall("joint")
    errors, warnings = [], []
    children = [joint.find("child").get("link") for joint in joints]
    roots = set(links) - set(children)
    if len(links) != len(robot.findall("link")) or len(set(children)) != len(children):
        errors.append("Duplicate links or multiply-parented child links")
    if len(roots) != 1:
        raise ValueError("Expected one connected root, got {}".format(roots))
    root = next(iter(roots))
    poses = {root: np.eye(4)}
    if not np.isfinite(initial_height):
        raise ValueError("Initial root height must be finite")
    poses[root][2, 3] = initial_height
    q = {name: angle for side in ("L", "R") for name, angle in (
        (side+"_thigh_joint", 0.6), (side+"_calf_joint", -1.2), (side+"_foot_joint", 0.0))}
    if initial_angles is not None:
        if set(initial_angles) != set(q) or not np.isfinite(list(initial_angles.values())).all():
            raise ValueError("Initial angles must contain Q1's six finite named joint positions")
        q = dict(initial_angles)
    pending = list(joints)
    joint_summary = []
    while pending:
        progressed = False
        for joint in list(pending):
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in links or child not in links:
                raise ValueError("Dangling joint reference")
            if parent not in poses:
                continue
            transform = origin(joint)
            if joint.get("type") != "fixed":
                if joint.find("axis").get("xyz") != "0 1 0":
                    raise ValueError("Q1 mapping requires +Y axes")
                angle = q[joint.get("name")]
                turning = np.eye(4)
                turning[:3, :3] = rotation([0, angle, 0])
                transform = transform @ turning
                limit = joint.find("limit")
                lower, upper = float(limit.get("lower")), float(limit.get("upper"))
                if not lower <= angle <= upper or float(limit.get("effort")) <= 0 or float(limit.get("velocity")) <= 0:
                    errors.append("Invalid limit: " + joint.get("name"))
                if joint.get("name").endswith("foot_joint") and (
                    joint.get("type") != "revolute" or (lower, upper) != (-99999, 99999)
                ):
                    errors.append("Wheel type/range convention: " + joint.get("name"))
                joint_summary.append(dict(name=joint.get("name"), axis="+Y", initial_rad=angle,
                                          lower=lower, upper=upper, effort=float(limit.get("effort")),
                                          velocity=float(limit.get("velocity"))))
            poses[child] = poses[parent] @ transform
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("Disconnected or cyclic joint tree")
    if set(poses) != set(links):
        errors.append("Not all links are reachable")
    inertia_rows, collisions, meshes = [], [], []
    total_mass, weighted_com = 0.0, np.zeros(3)
    for name, link in links.items():
        inertial = link.find("inertial")
        mass = float(inertial.find("mass").get("value"))
        values = inertial.find("inertia").attrib
        matrix = np.array([[float(values["ixx"]), float(values["ixy"]), float(values["ixz"])],
                           [float(values["ixy"]), float(values["iyy"]), float(values["iyz"])],
                           [float(values["ixz"]), float(values["iyz"]), float(values["izz"])]])
        principal = np.linalg.eigvalsh(matrix)
        if mass <= 0 or not np.isfinite(matrix).all() or principal[0] <= 0 or principal[-1] > principal[:2].sum() + 1e-10:
            errors.append("Nonphysical mass/inertia: " + name)
        com = origin(inertial)[:3, 3]
        com_world = poses[name][:3, :3] @ com + poses[name][:3, 3]
        total_mass += mass
        weighted_com += mass*com_world
        radii = np.sqrt(principal/mass)
        mesh_radius = None
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.get("filename")
            path = (mesh_root / Path(filename).name) if mesh_root else (urdf.parent / filename).resolve()
            vertices = stl_vertices(path) * vector(mesh.get("scale"), "1 1 1")
            t = origin(visual)
            vertices = vertices @ t[:3, :3].T + t[:3, 3]
            mesh_radius = float(np.linalg.norm(vertices-com, axis=1).max())
            if radii.max() > mesh_radius + 1e-6:
                warnings.append("Inertia exceeds visible mesh radius bound; confirm mesh completeness: " + name)
            meshes.append(dict(link=name, file=filename, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               dimensions_m=np.ptp(vertices, axis=0).tolist()))
        inertia_rows.append(dict(link=name, mass_kg=mass, principal_kg_m2=principal.tolist(),
                                 gyration_radius_m=radii.tolist(), mesh_radius_about_com_m=mesh_radius))
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            full = poses[name] @ origin(collision)
            if geometry.find("box") is not None:
                size = vector(geometry.find("box").get("size"))
                corners = np.array(list(itertools.product((-1., 1.), repeat=3))) * size/2
                bottom = float((corners @ full[:3, :3].T + full[:3, 3])[:, 2].min())
                shape, dimensions = "box", size.tolist()
            elif geometry.find("cylinder") is not None:
                element = geometry.find("cylinder")
                radius, length = float(element.get("radius")), float(element.get("length"))
                axis_z = full[2, 2]
                bottom = float(full[2, 3] - length/2*abs(axis_z) - radius*np.sqrt(max(0., 1-axis_z**2)))
                shape, dimensions = "cylinder", [radius, length]
            else:
                errors.append("Unsupported or mesh collision: " + name)
                continue
            if min(dimensions) <= 0 or bottom < -1e-6:
                errors.append("Invalid primitive or initial ground penetration: " + name)
            collisions.append(dict(link=name, name=collision.get("name"), shape=shape,
                                   dimensions_m=dimensions, initial_bottom_m=bottom))
    symmetry = []
    by_name = {row["link"]: row for row in inertia_rows}
    for suffix in ("thigh", "calf", "footmotor", "foot"):
        left, right = by_name["L_"+suffix], by_name["R_"+suffix]
        symmetry.append(dict(pair=suffix, mass_relative_difference=(right["mass_kg"]-left["mass_kg"])/left["mass_kg"],
                             principal_relative_difference=(np.array(right["principal_kg_m2"])/left["principal_kg_m2"]-1).tolist()))
    return dict(status="passed_static_checks" if not errors else "failed", errors=errors, warnings=warnings,
                link_count=len(links), joint_count=len(joints), active_joint_count=len(q), fixed_joint_count=len(joints)-len(q),
                total_mass_kg=total_mass, initial_base_z_m=initial_height,
                initial_total_com_m=(weighted_com/total_mass).tolist(),
                joints=joint_summary, inertias=inertia_rows, symmetry=symmetry, collisions=collisions, meshes=meshes,
                limitations=["Static primitive ground clearance only; no Isaac Gym import, contact simulation or viewer check.",
                             "Inertia small/large screening uses mesh dimensions, radius bounds and left/right principal comparisons; CAD mass distribution is unavailable, so a universal lower inertia bound cannot be certified.",
                             "No previous URDF version was supplied for a numerical inertia diff.",
                             "Full-range collision separation is not certified; see q1_collision_clearance.json."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=Path(__file__).resolve().parents[1]/"resources/robots/q1/urdf/q1.urdf")
    parser.add_argument("--mesh-root", type=Path)
    parser.add_argument("--thigh-angle", type=float, default=0.6)
    parser.add_argument("--calf-angle", type=float, default=-1.2)
    parser.add_argument("--initial-height", type=float, default=0.402)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    angles = {side+"_"+joint+"_joint": angle for side in ("L", "R")
              for joint, angle in (("thigh", args.thigh_angle), ("calf", args.calf_angle), ("foot", 0.0))}
    report = check(args.urdf, args.mesh_root, initial_angles=angles, initial_height=args.initial_height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "link_count", "active_joint_count", "total_mass_kg", "errors", "warnings")}, indent=2))
    raise SystemExit(bool(report["errors"]))
