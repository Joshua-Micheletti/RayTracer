from __future__ import annotations

from aabb.AABB import AABB


class BVHNode:
    def __init__(self, aabb: AABB, left: BVHNode = None, right: BVHNode = None, primitive=None, primitive_type=''):
        self.aabb = aabb
        self.left = left
        self.right = right
        self.primitive = primitive  # Only for leaf nodes
        self.primitive_type = primitive_type

    def __str__(self) -> str:
        output: str = "BVHNode:\n"

        output += f"\tAABB: {self.aabb}\n"
        output += f"\tPrimitive: {self.primitive}\n"
        output += f"\tLeft: {self.left}\n"
        output += f"\tRight: {self.right}"

        return output

    def __repr__(self) -> str:
        output: str = "BVHNode:\n"

        output += f"\tAABB: {self.aabb}\n"
        output += f"\tPrimitive: {self.primitive}\n"
        output += f"\tLeft: {self.left}\n"
        output += f"\tRight: {self.right}"

        return output
