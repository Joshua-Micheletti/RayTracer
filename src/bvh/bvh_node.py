"""bvhnode module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict, cast

import numpy as np

from aabb import AABB
from core import SSBOData
from utils.utils import pack_data_28_2_2

if TYPE_CHECKING:
    from primitive import Primitive


class PrimitiveData(TypedDict):
    """Input data for bulding the BVH."""

    primitive: Primitive
    index: int
    primitive_type: Literal["sphere", "triangle", "box"]


class BVHNode(SSBOData):
    """Node of a BVH."""

    def __init__(
        self,
        aabb: AABB,
        left: BVHNode | None = None,
        right: BVHNode | None = None,
        split_axis: int = 3,
        primitive_index: int = -1,
        primitive_type: str = "",
    ) -> None:
        """
        Initialization method.

        Args:
            aabb (AABB): Bounding box of the node
            left (BVHNode | None, optional): Pointer to the left node. Defaults to None.
            right (BVHNode | None, optional): Pointer to the right node. Defaults to None.
            primitive_index (int, optional): Index of the primitive. Defaults to -1.
            primitive_type (str, optional): Type of the primitive. Defaults to "".

        """
        self._dtype = np.dtype(
            [
                ("aabb", np.uint32, 3),  # vec3 + padding
                ("primitive", np.uint32),
                ("right_child_offset", np.int32),
                ("pad", np.uint32, 3),
            ],
            align=True,
        )

        super().__init__()

        self.aabb = aabb
        self.left = left
        self.right = right
        self.primitive_index = primitive_index  # Only for leaf nodes
        self.primitive_type = primitive_type
        self.split_axis = split_axis

    def __str__(self) -> str:
        """
        String representation of the BVH.

        Returns:
            str: Formatted string.

        """
        output: str = "BVHNode:\n"

        output += f"\tAABB: {self.aabb}\n"
        output += f"\tPrimitive: {self.primitive_index}\n"
        output += f"\tPrimitive Type: {self.primitive_type}\n"
        output += f"\tLeft: \n{self.left}\n"
        output += f"\tRight: \n{self.right}"

        return output

    def __repr__(self) -> str:
        """
        String representation of the BVH.

        Returns:
            str: Formatted string.

        """
        output: str = "BVHNode:\n"

        output += f"\tAABB: {self.aabb}\n"
        output += f"\tPrimitive: {self.primitive_index}\n"
        output += f"\tPrimitive Type: {self.primitive_type}\n"
        output += f"\tLeft: \n{self.left}\n"
        output += f"\tRight: \n{self.right}"

        return output

    @classmethod
    def build_tree(
        cls,
        primitives: list[PrimitiveData],
        depth: int = 0,
        bucket_count: int = 12,
        min_leaf_size: int = 1,
    ) -> BVHNode:
        """
        Optimized method for building a BVH tree from primitives using SAH.

        Args:
            primitives (list[PrimitiveData]): List of primitives to build the BVH out of.
            depth (int, optional): Depth of the tree. Defaults to 0.
            bucket_count (int, optional): Number of SAH buckets. Defaults to 12.
            min_leaf_size (int, optional): Minimum primitives per leaf. Defaults to 1.

        Returns:
            BVHNode: Root of the tree.

        """
        n = len(primitives)
        if n <= min_leaf_size:
            # Create leaf node
            primitive = primitives[0]["primitive"]
            return cls(
                primitive.aabb,
                primitive_index=primitives[0]["index"],
                primitive_type=primitives[0]["primitive_type"],
                split_axis=3,
            )

        axis = depth % 3

        # Precompute AABB mins/maxs and centroids
        aabb_mins = np.array([p["primitive"].aabb.min for p in primitives])
        aabb_maxs = np.array([p["primitive"].aabb.max for p in primitives])
        centroids = (aabb_mins + aabb_maxs) * 0.5

        global_min = np.min(aabb_mins, axis=0)
        global_max = np.max(aabb_maxs, axis=0)
        global_aabb = AABB(global_min, global_max)

        axis_min = global_min[axis]
        axis_max = global_max[axis]
        axis_range = axis_max - axis_min if axis_max > axis_min else 1e-8

        # Initialize buckets
        buckets = [{"count": 0, "bounds": None} for _ in range(bucket_count)]
        for i, p in enumerate(primitives):
            b_idx = int(((centroids[i, axis] - axis_min) / axis_range) * (bucket_count - 1))
            bucket = buckets[b_idx]
            bucket["count"] += 1
            bucket["bounds"] = (
                p["primitive"].aabb
                if bucket["bounds"] is None
                else bucket["bounds"].union(p["primitive"].aabb)
            )

        # Prefix computation
        prefix_bounds = [None] * bucket_count
        prefix_counts = [0] * bucket_count
        for i, b in enumerate(buckets):
            if b["count"] == 0:
                prefix_bounds[i] = prefix_bounds[i - 1] if i > 0 else None
                prefix_counts[i] = prefix_counts[i - 1] if i > 0 else 0
            else:
                prefix_bounds[i] = (
                    b["bounds"]
                    if i == 0 or prefix_bounds[i - 1] is None
                    else prefix_bounds[i - 1].union(b["bounds"])
                )
                prefix_counts[i] = b["count"] + (prefix_counts[i - 1] if i > 0 else 0)

        # Suffix computation
        suffix_bounds = [None] * bucket_count
        suffix_counts = [0] * bucket_count
        for i in reversed(range(bucket_count)):
            b = buckets[i]
            if b["count"] == 0:
                suffix_bounds[i] = suffix_bounds[i + 1] if i < bucket_count - 1 else None
                suffix_counts[i] = suffix_counts[i + 1] if i < bucket_count - 1 else 0
            else:
                suffix_bounds[i] = (
                    b["bounds"]
                    if i == bucket_count - 1 or suffix_bounds[i + 1] is None
                    else suffix_bounds[i + 1].union(b["bounds"])
                )
                suffix_counts[i] = b["count"] + (
                    suffix_counts[i + 1] if i < bucket_count - 1 else 0
                )

        # Find best SAH split
        best_cost = float("inf")
        best_split = None
        sa_p = global_aabb.surface_area()

        for i in range(1, bucket_count):
            left_count, right_count = prefix_counts[i - 1], suffix_counts[i]
            if left_count == 0 or right_count == 0:
                continue

            cost = (prefix_bounds[i - 1].surface_area() / sa_p) * left_count + (
                suffix_bounds[i].surface_area() / sa_p
            ) * right_count
            if cost < best_cost:
                best_cost = cost
                best_split = i

        # Partition primitives
        if best_split is None:
            # fallback: split in the middle along axis
            primitives.sort(key=lambda p: p["primitive"].aabb.min[axis])
            mid = n // 2
        else:
            # sort by centroid along axis and split by best bucket
            primitives.sort(
                key=lambda p: 0.5 * (p["primitive"].aabb.min[axis] + p["primitive"].aabb.max[axis])
            )
            mid = sum(b["count"] for b in buckets[:best_split])

        left = cls.build_tree(primitives[:mid], depth + 1, bucket_count, min_leaf_size)
        right = cls.build_tree(primitives[mid:], depth + 1, bucket_count, min_leaf_size)

        node_aabb_min = np.minimum(left.aabb.min, right.aabb.min)
        node_aabb_max = np.maximum(left.aabb.max, right.aabb.max)

        return cls(AABB(node_aabb_min, node_aabb_max), left, right, split_axis=axis)

    def flatten(self) -> list:
        """Flattens this BVHNode tree into a linear array for GPU traversal."""
        flat_nodes = []
        scene_aabb_min = self.aabb.min
        scene_extent = self.aabb.max - self.aabb.min

        def _flatten_recursive(node: BVHNode | None) -> int:
            if node is None:
                return -1

            current_index = len(flat_nodes)
            flat_nodes.append(None)  # placeholder

            if node.primitive_index != -1:
                primitive_type = -1
                if node.primitive_type == "triangle":
                    primitive_type = 0
                elif node.primitive_type == "sphere":
                    primitive_type = 1
                elif node.primitive_type == "box":
                    primitive_type = 2

                norm_min = (node.aabb.min - scene_aabb_min) / scene_extent
                norm_max = (node.aabb.max - scene_aabb_min) / scene_extent

                packed_min = np.round(norm_min * 65535).astype(np.uint16)
                packed_max = np.round(norm_max * 65535).astype(np.uint16)

                # Leaf node
                flat_node = {
                    "aabb_min": packed_min,
                    "aabb_max": packed_max,
                    "primitive_index": node.primitive_index,
                    "right_child_offset": 0,  # no children
                    "primitive_type": primitive_type,
                    "split_axis": node.split_axis,
                }
                flat_nodes[current_index] = flat_node
                return current_index

            # Internal node
            _flatten_recursive(node.left)
            right_child_index = _flatten_recursive(node.right)

            norm_min = (node.aabb.min - scene_aabb_min) / scene_extent
            norm_max = (node.aabb.max - scene_aabb_min) / scene_extent

            packed_min = np.round(norm_min * 65535).astype(np.uint16)
            packed_max = np.round(norm_max * 65535).astype(np.uint16)

            flat_node = {
                "aabb_min": packed_min,
                "aabb_max": packed_max,
                "primitive_index": -1,  # internal node
                "right_child_offset": right_child_index,
                "primitive_type": -1,
                "split_axis": node.split_axis,
            }
            flat_nodes[current_index] = flat_node
            return current_index

        _flatten_recursive(self)

        self._flat_nodes = flat_nodes
        self._calculate_ogl_ssbo_array()

        return flat_nodes

    def _calculate_ogl_ssbo_array(self) -> None:
        n = len(self._flat_nodes)
        self._ogl_ssbo_data = np.zeros(n, dtype=self._dtype)

        for i, node in enumerate(self._flat_nodes):
            packed_uint32_x = (node["aabb_max"][0] << 16) | node["aabb_min"][0]
            packed_uint32_y = (node["aabb_max"][1] << 16) | node["aabb_min"][1]
            packed_uint32_z = (node["aabb_max"][2] << 16) | node["aabb_min"][2]
            # aabb_min and aabb_max need 4 floats (vec3 + padding)
            # So pack with a zero at the end for padding
            self._ogl_ssbo_data[i]["aabb"] = [packed_uint32_x, packed_uint32_y, packed_uint32_z]
            # np_bvh[i]["aabb_min"][3] = 0.0  # padding

            # self._ogl_ssbo_data[i]["aabb_max"] = node["aabb_max"]
            # np_bvh[i]["aabb_max"][3] = 0.0  # padding
            self._ogl_ssbo_data[i]["primitive"] = pack_data_28_2_2(
                node["primitive_index"], node["primitive_type"], node["split_axis"]
            )
            # self._ogl_ssbo_data[i]["primitive_index"] = node["primitive_index"]
            # self._ogl_ssbo_data[i]["primitive_type"] = node["primitive_type"]
            self._ogl_ssbo_data[i]["right_child_offset"] = node["right_child_offset"]
            # self._ogl_ssbo_data[i]["_padding"] = [0, 0, 0]  # keep zero


class QBVHNode(SSBOData):
    """Node of a QBVH."""

    def __init__(
        self,
        aabb: AABB,
        children: list[BVHNode | None],
        primitive_index: list[int] | None = None,
        primitive_type: list[str] | None = None,
    ):
        u4 = np.dtype("<u4")  # little-endian uint32

        # Each field is 4 uints (16 bytes), so offsets are multiples of 16
        self._dtype = np.dtype(
            {
                "names": ["aabb_x", "aabb_y", "aabb_z", "metadata"],
                "formats": [(u4, 4), (u4, 4), (u4, 4), (u4, 4)],
                "offsets": [0, 16, 32, 48],
                "itemsize": 64,  # 4 fields * 16 bytes each = 64 bytes total
            }
        )

        if primitive_index is None:
            primitive_index = [-1, -1, -1, -1]
        if primitive_type is None:
            primitive_type = ["", "", "", ""]

        super().__init__()

        self.aabb = aabb
        self.children = children
        self.primitive_index = primitive_index  # Only for leaf nodes
        self.primitive_type = primitive_type

    def __str__(self) -> str:
        output = "QBVHNode:\n"
        output += f"AABB: {self.aabb}\n"
        output += f"Children: {self.children}\n"
        output += f"Primitive Types: {self.primitive_type}\n"
        output += f"Primitive Index: {self.primitive_index}\n"
        return output

    def __repr__(self) -> str:
        output = "QBVHNode:\n"
        output += f"AABB: {self.aabb}\n"
        output += f"Children: {self.children}\n"
        output += f"Primitive Types: {self.primitive_type}\n"
        output += f"Primitive Index: {self.primitive_index}\n"
        return output

    @classmethod
    def from_bvh(cls, node: BVHNode):
        # If it's a leaf in the binary BVH, make it a leaf QBVH node.
        if node.left is None and node.right is None:
            return cls(
                aabb=node.aabb,
                children=[None, None, None, None],
                primitive_index=node.primitive_index,
                primitive_type=node.primitive_type,
            )

        # Otherwise, collect up to 4 children
        candidates = []
        for child in (node.left, node.right):
            if child is None:
                continue
            # If binary child is leaf, just wrap it
            if child.left is None and child.right is None:
                candidates.append(cls.from_bvh(child))
            else:
                # Non-leaf: grab its two children if possible
                if child.left is not None:
                    candidates.append(cls.from_bvh(child.left))
                if child.right is not None:
                    candidates.append(cls.from_bvh(child.right))

        # Pad to 4 children
        while len(candidates) < 4:
            candidates.append(None)

        # Build this QBVH node
        return cls(aabb=node.aabb, children=candidates[:4])

    def flatten_qbvh(self):
        flat_nodes = []
        # mapping from QBVHNode (object) to index in flat array
        node_to_index = {}
        primitive_count = 0
        scene_aabb_min = self.aabb.min
        scene_extent = self.aabb.max - self.aabb.min

        def assign_index(node: QBVHNode):
            nonlocal primitive_count
            if node is None:
                return -1
            if node in node_to_index:
                return node_to_index[node]

            idx = len(flat_nodes)
            node_to_index[node] = idx

            # Reserve space for this node
            flat_nodes.append(None)

            # Recurse on children
            child_indices = []
            is_leaf = []
            prim_index = []
            prim_type = []

            for child in node.children:
                if child is None:
                    child_indices.append(-1)
                    is_leaf.append(1)  # mark as leaf so traversal ignores them
                    prim_index.append(9999999)  # invalid primitive
                    prim_type.append(3)  # unknown
                    continue

                if all(c is None for c in child.children):  # leaf
                    primitive_count += 1
                    child_indices.append(-1)
                    is_leaf.append(1)
                    prim_index.append(child.primitive_index)
                    prim_type.append(child.primitive_type)  # or however many primitives you pack
                else:  # internal
                    child_indices.append(assign_index(cast("QBVHNode", child)))
                    is_leaf.append(0)
                    prim_index.append(99999999)
                    prim_type.append("")

            for i in range(len(prim_type)):
                if prim_type[i] == "triangle":
                    prim_type[i] = 0
                elif prim_type[i] == "sphere":
                    prim_type[i] = 1
                elif prim_type[i] == "box":
                    prim_type[i] = 2
                else:
                    prim_type[i] = 3

            aabb_mins = []
            aabb_maxs = []

            for child in node.children:
                if child is None:
                    norm_min = (
                        np.array([1e30, 1e30, 1e30], dtype=np.float32) - scene_aabb_min
                    ) / scene_extent
                    norm_max = (
                        np.array([-1e30, -1e30, -1e30], dtype=np.float32) - scene_aabb_min
                    ) / scene_extent
                else:
                    norm_min = (child.aabb.min - scene_aabb_min) / scene_extent
                    norm_max = (child.aabb.max - scene_aabb_min) / scene_extent

                packed_min = np.round(norm_min * 65535).astype(np.uint16)
                packed_max = np.round(norm_max * 65535).astype(np.uint16)

                aabb_mins.append(packed_min)
                aabb_maxs.append(packed_max)

            # Build the flattened GPU node
            gpu_node = {
                "aabb_min": aabb_mins,
                "aabb_max": aabb_maxs,
                "childIndex": child_indices,
                "isLeaf": is_leaf,
                "primIndex": prim_index,
                "primType": prim_type,
            }

            flat_nodes[idx] = gpu_node
            return idx

        assign_index(self)

        print("PRIMITIVE COUNT AFTER FLATTENING", primitive_count)

        self._flat_nodes = flat_nodes
        self._calculate_ogl_ssbo_array()

    def _calculate_ogl_ssbo_array(self):
        n = len(self._flat_nodes)
        self._ogl_ssbo_data = np.zeros(n, dtype=self._dtype)

        for i, node in enumerate(self._flat_nodes):
            # aabb_x = []
            # aabb_y = []
            # aabb_z = []

            # for j in range(4):
            #     aabb_x.append(node["aabb_max"][j][0] << 16 | node["aabb_min"][j][0])
            #     aabb_y.append(node["aabb_max"][j][1] << 16 | node["aabb_min"][j][1])
            #     aabb_z.append(node["aabb_max"][j][2] << 16 | node["aabb_min"][j][2])

            # self._ogl_ssbo_data[i]["aabb_x"] = np.array(aabb_x, dtype=np.uint32)
            # self._ogl_ssbo_data[i]["aabb_y"] = np.array(aabb_y, dtype=np.uint32)
            # self._ogl_ssbo_data[i]["aabb_z"] = np.array(aabb_z, dtype=np.uint32)

            self._ogl_ssbo_data["aabb_x"][i, :] = [
                (np.uint32(node["aabb_max"][j][0]) << 16) | np.uint32(node["aabb_min"][j][0])
                for j in range(4)
            ]

            self._ogl_ssbo_data["aabb_y"][i, :] = [
                (np.uint32(node["aabb_max"][j][1]) << 16) | np.uint32(node["aabb_min"][j][1])
                for j in range(4)
            ]

            self._ogl_ssbo_data["aabb_z"][i, :] = [
                (np.uint32(node["aabb_max"][j][2]) << 16) | np.uint32(node["aabb_min"][j][2])
                for j in range(4)
            ]

            index = []

            for k in range(4):
                if node["primType"][k] == 3:
                    index.append(
                        node["childIndex"][k] if node["childIndex"][k] != -1 else -1 & 0xFFFFFFFF
                    )
                else:
                    index.append(
                        node["primIndex"][k] if node["primIndex"][k] != -1 else -1 & 0xFFFFFFFF
                    )

            indexArray = np.array(index, dtype=np.uint32)  # 4 elements
            primType = np.array(node["primType"], dtype=np.uint32)  # 4 elements

            # Mask and shift
            packed = (indexArray & 0x3FFFFFFF) | ((primType & 0x3) << 30)

            self._ogl_ssbo_data[i]["metadata"] = packed.astype(np.uint32)
