import time

import glfw
import numpy as np
from numpy.typing import NDArray

from aabb.AABB import AABB
from bvh.BVHNode import BVHNode
from controller.Controller import Controller
from data.Data import Data
from model.Model import Model
from primitive import Plane, Sphere, Triangle
from renderer import Renderer
from window.window import Window

# import win_precise_time as wpt


def main():
    window = Window.get_instance()
    controller = Controller.getInstance()
    data = Data.getInstance()
    renderer = Renderer.get_instance()

    #                  color (r,g,b)  emission(rgb)  e_p shine albedo(r,g,b)  albedo
    data.load_material(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0)  # 0 white opaqu
    data.load_material(
        1.0,
        0.2,
        0.2,
        0.0,
        0.0,
        0.0,
        1.0,
        0.2,
        1.0,
        0.2,
        0.2,
        1.0,
    )  # 1 red 0.2 shine
    data.load_material(
        0.2,
        1.0,
        0.2,
        0.0,
        0.0,
        0.0,
        1.0,
        0.4,
        0.2,
        1.0,
        0.2,
        1.0,
    )  # 2 green 0.4 shine
    data.load_material(
        0.2,
        0.2,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.6,
        0.2,
        0.2,
        1.0,
        1.0,
    )  # 3 blue 0.6 shine
    data.load_material(
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        2.0,
        0.8,
        1.0,
        1.0,
        1.0,
        1.0,
    )  # 4 white 0.8 shine
    data.load_material(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )  # 5 white 1.0 shine
    data.load_material(
        1.0,
        0.2,
        0.2,
        0.0,
        0.0,
        0.0,
        3.0,
        0.5,
        1.0,
        1.0,
        1.0,
        0.2,
    )  # 6 red shiny albedo 20%
    data.load_material(
        0.2,
        1.0,
        0.2,
        0.0,
        0.0,
        0.0,
        3.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.5,
    )  # 7 green shiny albedo 50%
    data.load_material(
        0.2,
        0.2,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )  # 8 blue shiny albedo 100%
    data.load_material(1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 0.5, 1.0, 1.0, 1.0, 0.0)  # 9 emissive red
    data.load_material(
        1.0,
        1.0,
        1.0,
        0.2,
        1.0,
        0.2,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
    )  # 10 emissive blue
    data.load_material(1.0, 0.2, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0)  # 11 red
    data.load_material(0.2, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0)  # 12 blue
    data.load_material(0.2, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0)  # 13 green
    data.load_material(
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        5.0,
        0.5,
        1.0,
        1.0,
        1.0,
        0.0,
    )  # 14 emissive white
    data.load_material(
        124 / 255,
        112 / 255,
        208 / 255,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
    )  # 15 white opaque
    # rgb(124, 112, 208)

    gally = Model("../models/gally.obj")
    tree = Model("../models/tree.obj")
    box = Model("../models/box.obj")

    tree.scale(0.04, 0.04, 0.04)
    tree.move(0, 0, 0)
    gally.move(0, 0.5, 0)
    gally.scale(0.1, 0.1, 0.1)
    box.scale(0.4, 0.4, 0.4)
    box.move(-0.5, 2.30, 0)

    # data.load_model(gally, 15)
    # data.load_model(tree, 10)
    # data.load_model(box, 14)
    primitives = []
    vertices = gally.vertices
    triangle_data: NDArray = None
    
    # triangle_data = Triangle([0.2, 0.5, 1.0], [1.0, 0.5, 0.2], [0.5, 1.0, 0.2], 1).ogl_ssbo_data
    
    for i in range(0, len(vertices), 9):
        v0 = np.array([vertices[i], vertices[i + 1], vertices[i + 2]], dtype=np.float32)
        v1 = np.array([vertices[i + 3], vertices[i + 4], vertices[i + 5]], dtype=np.float32)
        v2 = np.array([vertices[i + 6], vertices[i + 7], vertices[i + 8]], dtype=np.float32)
            
        triangle: Triangle = Triangle(v0=v0, v1=v1, v2=v2, material=14)
        print(triangle)
        primitives.append({"primitive": triangle, "index": int(i / 9), "primitive_type": "triangle"})
        
        if triangle_data is None:
            triangle_data = triangle.ogl_ssbo_data
        else:
            triangle_data = np.concatenate((triangle_data, triangle.ogl_ssbo_data))
            
    renderer.update_ssbo(renderer.triangles, triangle_data, 12)

    # data.set_color(0, 1.0, 1.0, 1.0, 1.0)
    # data.set_color(1, 1.0, 1.0, 0.5, 1.0)
    # data.set_color(2, 1.0, 1.0, 1.0, 1.0)

    planes_data_array: NDArray = Plane(
        point=[0.0, 30.0, 0.0], normal=[0.0, -1.0, 0.0], material=14
    ).ogl_ssbo_data

    planes_data_array = np.concatenate(
        (planes_data_array, Plane(point=[0.0, -30.0, 0.0], normal=[0.0, 1.0, 0.0], material=1).ogl_ssbo_data)
    )
    planes_data_array = np.concatenate(
        (planes_data_array, Plane(point=[-30.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0], material=2).ogl_ssbo_data)
    )
    planes_data_array = np.concatenate(
        (planes_data_array, Plane(point=[30.0, 0.0, 0.0], normal=[-1.0, 0.0, 0.0], material=3).ogl_ssbo_data)
    )
    planes_data_array = np.concatenate(
        (planes_data_array, Plane(point=[0.0, 0.0, 30.0], normal=[0.0, 0.0, -1.0]).ogl_ssbo_data)
    )
    planes_data_array = np.concatenate(
        (planes_data_array, Plane(point=[0.0, 0.0, -30.0], normal=[0.0, 0.0, 1.0]).ogl_ssbo_data)
    )

    renderer.update_ssbo(renderer.planes, planes_data_array, 6)

    # data.load_sphere(0.0, 7.0, 0.0, 0.5, 14)

    

    data_array = None

    # for i in range(1000):
    #     sphere: Sphere = Sphere(
    #         [random.random() * 10 - 5, random.random() * 10 - 5, random.random() * 10 - 5],
    #         random.random() * 0.5 + 0.1,
    #         random.randint(0, 13),
    #     )
    #     # index = data.load_primitive(sphere)
    #     primitives.append({"sphere": sphere, "index": i})

    #     if data_array is None:
    #         data_array = sphere.ogl_ssbo_data
    #     else:
    #         data_array = np.concatenate((data_array, sphere.ogl_ssbo_data))

    sphere: Sphere = Sphere([0.0, 15.0, 0.0], 5.0, 14)
    # primitives.append({"primitive": sphere, "index": 0, "primitive_type": "sphere"})
    # data_array = np.concatenate((data_array, sphere.ogl_ssbo_data))
    data_array = sphere.ogl_ssbo_data

    renderer.update_ssbo(renderer.spheres, data_array, 5)

    # triangle: Triangle = Triangle(v0=[0, 0, 0], v1=[2, 0, 0], v2=[1, 2, 1])
    # primitives.append({"primitive": triangle, "index": 0, "primitive_type": "triangle"})

    # renderer.update_ssbo(renderer.triangles, triangle.ogl_ssbo_data, 12)

    def build_bvh(primitives: list[dict[str, Sphere | int | str]], depth=0):
        """Function for building a BVH"""
        # default case to end the recursion (only 1 primitive left in the list)
        if len(primitives) == 1:
            return BVHNode(
                primitives[0]["primitive"].aabb,
                primitive=primitives[0]["index"],
                primitive_type=primitives[0]["primitive_type"],
            )

        # Every recursion, the axis in which we split changes
        axis = depth % 3

        # sort the primitives available based on the minimum coordinate of their bounding box on the selected axis
        primitives.sort(key=lambda p: p["primitive"].aabb.min[axis])

        # find the middle point of the primitives sorted from left to right
        mid = len(primitives) // 2

        # call the building function recursively for the left and right sides passing the part of the list that correspond and increasing the depth
        left = build_bvh(primitives[:mid], depth + 1)
        right = build_bvh(primitives[mid:], depth + 1)

        # build the root node based on the bounding box of the child nodes
        aabb_min = np.minimum(left.aabb.min, right.aabb.min)
        aabb_max = np.maximum(left.aabb.max, right.aabb.max)

        return BVHNode(AABB(aabb_min, aabb_max), left, right)

    root: BVHNode = build_bvh(primitives)

    def flatten_bvh(root):
        flat_nodes = []

        def recurse(node: BVHNode):
            current_index = len(flat_nodes)
            # Placeholder for this node, append dummy to fix index now
            flat_nodes.append(None)

            # Leaf or internal node?
            if node.primitive is not None:
                # Leaf node: no children, store primitive index
                left_idx = -1
                right_idx = -1
                primitive = node.primitive
                primitive_type = 0 if node.primitive_type == "triangle" else 1
            else:
                # Internal node: recurse children
                left_idx = recurse(node.left) if node.left else -1
                right_idx = recurse(node.right) if node.right else -1
                primitive = -1
                primitive_type = -1

            # Pack your AABB: (min_x, min_y, min_z), (max_x, max_y, max_z)
            # This depends on your AABB representation
            aabb_min = node.aabb.min  # assuming .min is vec3 or tuple
            aabb_max = node.aabb.max

            # Create a flat node data structure
            flat_node = {
                "aabb_min": aabb_min,
                "aabb_max": aabb_max,
                "left": left_idx,
                "right": right_idx,
                "primitive": primitive,
                "primitive_type": primitive_type,
            }

            # Replace placeholder with actual data
            flat_nodes[current_index] = flat_node
            return current_index

        recurse(root)

        return flat_nodes

    flattened_bvh: list = flatten_bvh(root)

    # print(flattened_bvh)

    data.load_bvh(flattened_bvh)

    # print(root)

    dt = 0

    while not glfw.window_should_close(window.window):
        start = time.time()

        controller.update(dt)
        data.update()
        renderer.render()

        glfw.swap_buffers(window.window)
        glfw.poll_events()

        end = time.time()
        dt = end - start
        print(f"Total: {dt * 1000}")
        print(f"FPS: {1 / (dt if dt != 0 else 0.000001)}")
        print(f"Render: {renderer.render_time * 1000}")
        print(f"Samples: {renderer.rendered_frames}")
        print(f"Bounces: {renderer.bounces}")

    glfw.terminate()


if __name__ == "__main__":
    main()
