import tomllib
from pathlib import Path
from typing import Any
from material import Material
from model.model import Model
from renderer import Renderer
from primitive import Sphere, Triangle, Box
import time
from bvh import BVHNode, QBVHNode
from utils.utils import quad_to_triangles
from numpy.typing import NDArray

import numpy as np


def load_scene(filepath: str) -> None:
    scene_data = _open_scene(filepath)

    renderer: Renderer = Renderer.get_instance()

    primitives = []

    print(scene_data)

    variables = scene_data.get("variables")

    _load_materials(scene_data["materials"], variables)
    _load_spheres(scene_data["spheres"], primitives, variables)
    return_data = _load_quads(scene_data["quads"], primitives)
    _load_meshes(scene_data.get("meshes"), primitives, variables, return_data[1], return_data[0])
    # _load_boxes(scene_data["boxes"], primitives, variables)

    start_building_bvh = time.perf_counter()

    root: BVHNode = BVHNode.build_tree(primitives)

    end_building_bvh = time.perf_counter()

    print("built base BVH:", end_building_bvh - start_building_bvh)
    # root.flatten()

    start_building_qbvh = time.perf_counter()

    qbvh_root: QBVHNode = QBVHNode.from_bvh(root)

    end_building_qbvh = time.perf_counter()

    print("collapsed to QBVH:", end_building_qbvh - start_building_qbvh)

    start_flattening_qbvh = time.perf_counter()

    qbvh_root.flatten_qbvh()

    end_flattening_qbvh = time.perf_counter()

    print("flattened the QBVH:", end_flattening_qbvh - start_flattening_qbvh)

    # print("QBVH", qbvh_root)

    start_loading_bvh = time.perf_counter()

    renderer.update_ssbo(renderer.bvh, qbvh_root.ogl_ssbo_data, 11)

    end_loading_bvh = time.perf_counter()

    print("Loaded BVH to VRAM:", end_loading_bvh - start_loading_bvh)

    print("BVH size:", qbvh_root.ogl_ssbo_data.nbytes)

    renderer.scene_min_coords = root.aabb.min
    renderer.scene_extent = root.aabb.max - root.aabb.min


def _load_materials(materials: list[dict], variables: dict) -> None:
    renderer: Renderer = Renderer.get_instance()

    material_data = None
    for material in materials:
        material_object = Material(
            color=_variable(material.get("color", [1.0, 1.0, 1.0]), variables),
            emission=_variable(material.get("emission", [0.0, 0.0, 0.0, 0.0]), variables),
            smoothness=_variable(material.get("smoothness", 0), variables),
            metallic=_variable(material.get("metallic", 0), variables),
            transmission=_variable(material.get("transmission", 0), variables),
            ior=_variable(material.get("ior", 1.0), variables),
        )

        print(material_object)

        if material_data is None:
            material_data = material_object.ogl_ssbo_data
        else:
            material_data = np.concatenate((material_data, material_object.ogl_ssbo_data))

    # result = np.concatenate(material_data)

    renderer.update_ssbo(renderer.materials, material_data, 9)


def _load_spheres(spheres: list[dict], primitives: list[dict], variables: dict) -> None:
    renderer: Renderer = Renderer.get_instance()

    sphere_data = []

    for [index, sphere] in enumerate(spheres):
        sphere_primitive = Sphere(
            center=_variable(sphere.get("center"), variables),
            radius=_variable(sphere.get("radius"), variables),
            material=_variable(sphere.get("material"), variables),
        )
        sphere_data.append(sphere_primitive.ogl_ssbo_data)

        primitives.append(
            {
                "primitive": sphere_primitive,
                "index": index,  # adjust as needed
                "primitive_type": "sphere",
            }
        )

    result = np.concatenate([sphere_data])
    renderer.update_ssbo(renderer.spheres, result, 5)


def _load_boxes(boxes: list[dict], primitives: list[dict], variables: dict) -> None:
    renderer: Renderer = Renderer.get_instance()

    box_data = []

    for [index, box] in enumerate(boxes):
        box_primitive = Box(
            p0=_variable(box.get("p0"), variables),
            p1=_variable(box.get("p1"), variables),
            material=_variable(box.get("material"), variables),
        )
        box_data.append(box_primitive.ogl_ssbo_data)

        primitives.append(
            {
                "primitive": box_data,
                "index": index,  # adjust as needed
                "primitive_type": "box",
            }
        )

    result = np.concatenate([box_data])
    renderer.update_ssbo(renderer.boxes, result, 7)


def _load_meshes(
    meshes: list[dict],
    primitives: list[dict],
    variables: dict,
    triangle_index: int,
    triangle_data_list,
):
    print("Triangle Index", triangle_index)
    renderer: Renderer = Renderer.get_instance()

    if meshes:
        for mesh in meshes:
            model = Model(str(mesh.get("path")))

            vertices = model.vertices
            # triangle_data: NDArray = None

            x_offset = 0
            y_offset = 0
            z_offset = 0
            material = int(mesh.get("material"))
            j = 0

            # Convert your flat list to a (num_triangles, 3, 3) array
            vertices_array = np.array(vertices, dtype=np.float32).reshape(-1, 3, 3)

            # Apply offsets in a vectorized way
            offsets = np.array([x_offset, y_offset, z_offset], dtype=np.float32)
            vertices_array += offsets  # broadcasts to all vertices

            for j, tri in enumerate(vertices_array):
                triangle = Triangle(v0=tri[0], v1=tri[1], v2=tri[2], material=material)
                primitives.append(
                    {
                        "primitive": triangle,
                        "index": j + triangle_index,  # adjust as needed
                        "primitive_type": "triangle",
                    }
                )
                triangle_data_list.append(triangle.ogl_ssbo_data)

    triangle_data = np.vstack(triangle_data_list)

    renderer.update_ssbo(renderer.triangles, triangle_data, 12)


def _load_quads(quads: list[dict], primitives: list[dict]) -> tuple:
    triangle_data = []

    max_index = 0

    for [index, quad] in enumerate(quads):
        triangles = quad_to_triangles(
            center=quad.get("center"),
            width=quad.get("width"),
            height=quad.get("height"),
            normal=quad.get("normal"),
        )

        triangle_1 = Triangle(
            v0=triangles[0][0],
            v1=triangles[0][1],
            v2=triangles[0][2],
            material=quad.get("material"),
        )

        print(triangle_1)

        triangle_2 = Triangle(
            v0=triangles[1][0],
            v1=triangles[1][1],
            v2=triangles[1][2],
            material=quad.get("material"),
        )

        print(triangle_2)

        primitives.append(
            {
                "primitive": triangle_1,
                "index": index * 2,  # adjust as needed
                "primitive_type": "triangle",
            }
        )

        primitives.append(
            {
                "primitive": triangle_2,
                "index": index * 2 + 1,  # adjust as needed
                "primitive_type": "triangle",
            }
        )

        triangle_data.append(triangle_1.ogl_ssbo_data)
        triangle_data.append(triangle_2.ogl_ssbo_data)

        max_index = index * 2 + 2

    return (triangle_data, max_index)
    # renderer.update_ssbo(renderer.triangles, result, 12)


def _open_scene(filepath: str) -> dict[str, Any]:
    with Path(filepath).open("rb") as f:
        return tomllib.load(f)


def _variable(value: Any, variables: dict) -> Any:
    if not isinstance(value, str):
        return value

    if value.startswith("{") and value.endswith("}"):
        key = value[1:-1]  # everything except first and last character
        return variables.get(key)

    return None
