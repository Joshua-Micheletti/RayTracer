from PIL import Image
import numpy as np

class Renderer:
    
    __instance = None
    
    @staticmethod
    def getInstance():
        if Renderer.__instance == None:
            Renderer()
        return Renderer.__instance
    
    def __init__(self):
        if Renderer.__instance != None:
            raise Exception("Window already exists!")
        
        Renderer.__instance = self
    
    
    def render(self):
        width = 400
        height = 300
        
        ratio = float(width) / height
        max_depth = 3
        
        img = Image.new(mode = "RGB", size = (width, height), color = (77, 77, 77))


        camera = np.array([0, 0, 1])
        
        objects = [
            { 'center': np.array([-0.2, 0, -1]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 0, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
            { 'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
            { 'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]), 'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
            { 'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 }
        ]
        
        light = { 'position': np.array([5, 5, 5]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }
        
        for y in range(height):
            for x in range(width):
                u = map_range(x, 0, width,  0, 1)
                v = map_range(y, 0, height, 0, 1)
                
                u = u - 0.5
                v = v - 0.5
                u *= 2
                v *= 2
                
                v /= ratio
                
                pixel = np.array([u, v, 0])
                
                
                origin = camera
                direction = normalize(pixel - origin)
                
                color = np.zeros((3))
                reflection = 1
                
                for k in range(max_depth):
                    nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
                    if nearest_object is None:
                        continue

                    # compute intersection point between ray and nearest object
                    intersection = origin + min_distance * direction

                    normal_to_surface = normalize(intersection - nearest_object['center'])
                    shifted_point = intersection + 1e-5 * normal_to_surface
                    intersection_to_light = normalize(light['position'] - shifted_point)

                    _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
                    intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
                    is_shadowed = min_distance < intersection_to_light_distance

                    if is_shadowed:
                        continue
                    
                    # RGB
                    illumination = np.zeros((3))

                    # ambiant
                    illumination += nearest_object['ambient'] * light['ambient']

                    # diffuse
                    illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface)

                    # specular
                    intersection_to_camera = normalize(camera - intersection)
                    H = normalize(intersection_to_light + intersection_to_camera)
                    illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4)
                
                    # reflection
                    color += reflection * illumination
                    reflection *= nearest_object['reflection']
                    
                    origin = shifted_point
                    direction = reflected(direction, normal_to_surface)
                

                # print(illumination)

                # r = int(map_range(u, 0, 1, 0, 255))
                # g = int(map_range(v, 0, 1, 0, 255))
                r = int(map_range(color[0], 0, 1, 0, 255))
                g = int(map_range(color[1], 0, 1, 0, 255))
                b = int(map_range(color[2], 0, 1, 0, 255))

                img.putpixel((x, (height - 1) - y), (r, g, b))
        
        img.save("render.png")
        
        
def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;


def normalize(vector):
    return vector / np.linalg.norm(vector)


def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None


def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance


def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis
