
#
#   Blender 3.0.1.
#

import bpy
from bpy import data
from bpy import context
import bmesh
from bmesh import geometry
import mathutils
from mathutils.bvhtree import BVHTree

import numpy as np
from numpy.random import default_rng

import math

################################################################################################
#
# COMMON; HELPERS.
#
################################################################################################

# https://blender.stackexchange.com/questions/220072/check-using-name-if-a-collection-exists-in-blend-is-linked-to-scene
def create_collection_if_not_exists(collection_name):
    if collection_name not in bpy.data.collections:
        new_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(new_collection) #Creates a new collection

def select_activate_only(objects=[]):
    for obj in bpy.data.objects:
        obj.select_set(False)
    bpy.context.view_layer.objects.active = None 
    for obj in objects:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

# Given a list of lists, find index of list with smallest number of elements
def get_idx_min_list(list_of_lists):
    found_idx = 0
    min_elements = len(list_of_lists[0])
    for i in range(len(list_of_lists)):
        if min_elements > len(curr_list):
            min_elements = len(curr_list)
            found_idx = i
    return found_idx

# triangulate using bmesh.
def triangulate(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces, quad_method="BEAUTY", ngon_method="BEAUTY")
    bm.to_mesh(obj.data)
    bm.free()

# https://graphics.pixar.com/library/OrthonormalB/paper.pdf
def pixar_onb(n):
    t = mathutils.Vector((0,0,0))
    b = mathutils.Vector((0,0,0))
    if(n[2] < 0.0):
        a = 1.0 / (1.0 - n[2])
        b = n[0] * n[1] * a
        t = mathutils.Vector((1.0 - n[0] * n[0] * a, -b, n[0]))
        b = mathutils.Vector((b, n[1] * n[1] * a - 1.0, -n[1]))
    else:
        a = 1.0 / (1.0 + n[2])
        b = -n[0] * n[1] * a
        t = mathutils.Vector((1.0 - n[0] * n[0] * a, b, -n[0]))
        b = mathutils.Vector((b, 1 - n[1] * n[1] * a, -n[1]))
    return t, b

# https://stackoverflow.com/questions/33757842/how-to-calculate-coordinate-system-from-one-vector
def coordsys_from_vector(x):
    tmp = mathutils.Vector((1, 0, 0))
    if mathutils.Vector(x-tmp).length < 1e-3:
        tmp = mathutils.Vector((0, 1, 0))
    if mathutils.Vector(x-tmp).length < 1e-3:
        tmp = mathutils.Vector((0, 0, 1))
    z = x.cross(tmp)
    y = z.cross(x)
    return mathutils.Vector(y), mathutils.Vector(z)

def draw_vector(center, v):
    for i in range(5):
        pos_v = center + v * i
        bpy.ops.mesh.primitive_cube_add(size=0.5, location=pos_v, scale=(1, 1, 1))

def draw_coord_axis(center, t,b,n):
    for i in range(5):
        pos_t = center + t * i
        pos_b = center + b * i
        pos_n = center + line_dir * i
        bpy.ops.mesh.primitive_cube_add(size=0.5, location=pos_t, scale=(1, 1, 1))
        bpy.ops.mesh.primitive_cube_add(size=0.5, location=pos_b, scale=(1, 1, 1))
        bpy.ops.mesh.primitive_cube_add(size=0.7, location=pos_n, scale=(1, 1, 1))

################################################################################################
#
# MESH POINT SAMPLERS.
#
################################################################################################

# Return positions of object vertices.
# position = (p, n, w, tbn)
def get_object_vertices_positions(base_obj):
    positions = []
    bm = bmesh.new()
    bm.from_mesh(base_obj.data)
    for v in bm.verts:
        position = []
        p = mathutils.Vector(v.co)
        n = mathutils.Vector(v.normal)
        w = 0.0 # TODO
        t, b = pixar_onb(n) # or t,b = coordsys_from_vector(line_dir)
        tbn = mathutils.Matrix((t, b, n)) # NOTE: using pixar_onb()?
        tbn = tbn.transposed() # TODO: why transposing?
        tbn.resize_4x4()
        position.append(p)
        position.append(n)
        position.append(w)
        position.append(tbn)
        positions.append(position)
    bm.free()
    return positions

# https://stackoverflow.com/questions/19045971/random-rounding-to-integer-in-python
def probabilistic_round(x):
    return int(math.floor(x + mathutils.noise.random()))

# https://github.com/blender/blender/blob/master/source/blender/nodes/geometry/nodes/node_geo_distribute_points_on_faces.cc
# base_obj - MUST BE TRIANGULATED!
# returns: list of touples: (p, N, w, tbn)
def mesh_uniform_weighted_sampling(base_obj, n_samples, base_density=1.0, use_weight_paint=False):
    rng = default_rng()
    samples = [] # (p, N, w, tbn)
    samples_all = []
    for polygon in base_obj.data.polygons: # must be triangulated mesh!
        # Extract triangle vertices and their weights.
        triangle_vertices = []
        triangle_vertex_weights = []
        for v_idx in polygon.vertices:
            v = base_obj.data.vertices[v_idx]
            triangle_vertices.append(v.co)
            if len(v.groups) < 1:
                triangle_vertex_weights.append(0.0)
            else:
                triangle_vertex_weights.append(v.groups[0].weight) # TODO: only one group? Investigate! float in [0, 1], default 0.0
        # Create samples.
        polygon_density = 1
        if use_weight_paint:
            polygon_density = (triangle_vertex_weights[0] + triangle_vertex_weights[1] + triangle_vertex_weights[2]) / 3.0
        point_amount = probabilistic_round(polygon.area * polygon_density * base_density)
        for i in range(point_amount):
            a = mathutils.noise.random()
            b = mathutils.noise.random()
            c = mathutils.noise.random()
            s = a + b + c
            un = (a / s)
            vn = (b / s)
            wn = (c / s)
            p = un * triangle_vertices[0] + vn * triangle_vertices[1] + wn * triangle_vertices[2]
            w = un * triangle_vertex_weights[0] + vn * triangle_vertex_weights[1] + wn * triangle_vertex_weights[2] # interpolate weight
            n = polygon.normal # TODO: vertex normals?
            # Calc BTN. 
            t = mathutils.Vector(triangle_vertices[0] - p) # NOTE: use most distant point from barycentric coord to evade problems with 0
            t = t.normalized()
            bt = n.cross(t)
            bt = bt.normalized()
            tbn = mathutils.Matrix((t, bt, n)) # NOTE: using pixar_onb()?
            tbn = tbn.transposed() # TODO: why transposing?
            tbn.resize_4x4()
            samples_all.append([p,n,w,tbn])
    print(len(samples_all), n_samples)
    random_sample_indices = rng.integers(len(samples_all), size=n_samples)
    for i in random_sample_indices:
        samples.append(samples_all[i])
    return samples

################################################################################################
#
# PATH BUILDERS.
#
################################################################################################


# https://en.wikipedia.org/wiki/B%C3%A9zier_curve
# linear bezier or lerp
def linear_bezier(p0, p1, t):
    return (1 - t) * p0 + t * p1

# https://en.wikipedia.org/wiki/B%C3%A9zier_curve
def quadratic_bezier(p0, p1, p2, t):
    return linear_bezier(linear_bezier(p0, p1, t), linear_bezier(p1, p2, t), t)

# origin, destination - only Vector3
def create_bezier_from_two_points(origin, destination, point_density, control_avg=-2.0, control_std=1.0):
    bezier_line_points = []
    p0 = origin
    p2 = destination
    p1_x = linear_bezier(p0[0], p2[0], 0.5) # TODO: random interp factor?
    p1_y = linear_bezier(p0[1], p2[1], 0.5)
    control_point = (control_avg - control_std) * mathutils.noise.random() + control_std
    p1_z = min(p0[2], p2[2]) + control_avg #  mathutils.noise.random() * 10
    p1 = mathutils.Vector(( p1_x, p1_y, p1_z )) # TODO: randomize P1 control point
    dist_p0_p2 = mathutils.Vector(p0-p2).length
    point_amount = probabilistic_round(dist_p0_p2 * point_density)
    for t in np.linspace(0, 1, point_amount):
        bezier_line_point = quadratic_bezier(p0, p1, p2, t)
        bezier_line_points.append(bezier_line_point)
    return bezier_line_points

# For now it simulates gravity.
# Create bezier lines points between two sets of starting and ending points
# origin_samples, destination_samples - lists of tuples (p,n,w,tbn)
def create_bezier_line_points_from_two_sets(origin_samples, destination_samples, point_density):
    bezier_lines_points = []
    for i in range(len(origin_samples)):
        bezier_line_points = create_bezier_from_two_points(origin_samples[i][0], destination_samples[i][0], point_density)
        bezier_lines_points.append(bezier_line_points)
    return bezier_lines_points

# Create bezier lines points for given path key points
# path_key_points - list of (p,n,w,tbn)
def create_bezier_line_points_from_path(path_key_points, point_density):
    bezier_path_points = []
    prev_path_key_point = path_key_points[0]
    for i in range(1, len(path_key_points)):
        curr_path_key_point = path_key_points[i]
        path_points = create_bezier_from_two_points(prev_path_key_point, curr_path_key_point, point_density)
        bezier_path_points.extend(path_points)
        prev_path_key_point = path_key_points[i]
    return bezier_path_points

# Generates hemisphere samples in tangent space where surface normal points in z.
# https://learnopengl.com/Advanced-Lighting/SSAO
def generate_hemisphere_samples_tangent_space(n_samples):
    samples = []
    for i in range(n_samples):
        sample = mathutils.Vector((
            mathutils.noise.random() * 2.0 - 1.0,
            mathutils.noise.random() * 2.0 - 1.0,
            mathutils.noise.random()))
        sample.normalize()
        #sample = sample * mathutils.noise.random() # ???
        samples.append(sample)
    return samples

def generate_sphere_samples_tangent_space(n_samples):
    samples = []
    for i in range(n_samples):
        sample = mathutils.Vector((
            mathutils.noise.random() * 2.0 - 1.0,
            mathutils.noise.random() * 2.0 - 1.0,
            mathutils.noise.random() * 2.0 - 1.0,))
        sample.normalize()
        #sample = sample * mathutils.noise.random() # ???
        samples.append(sample)
    return samples

# intersectable_object - [(obj, "<normal_direction>"), ...]
# Returns (p,n,w,tbn) or all None if hit is not found
# TODO: in corners of cube small distance raycastings are found. They can be terminated.
def raycast(intersectable_object, origin, direction):
    bm = bmesh.new()
    bm.from_mesh(intersectable_object[0].data)
    bvhtree = BVHTree.FromBMesh(bm)
    ray_cast_context = bvhtree.ray_cast(origin, direction) # returns: (Vector location, Vector normal, int index, float distance)
    if ray_cast_context[0] == None: # values will all be None if no hit is found
        bm.free()
        return (None, None, None, None) 
    else:
        p = mathutils.Vector(ray_cast_context[0])
        n = mathutils.Vector(ray_cast_context[1])
        if intersectable_object[1] == "normal_inside":
            n = -n # NOTE: correct tbn is calculate on this normal later!
        w = 0.0 # TODO: use original point w!
        bm.faces.ensure_lookup_table()
        intersected_face = bm.faces[ray_cast_context[2]]
        t = mathutils.Vector(intersected_face.verts[0].co - p)
        t.normalize()
        bt = n.cross(t)
        bt = bt.normalized()
        tbn = mathutils.Matrix((t, bt, n)) # NOTE: using pixar_onb()?
        tbn = tbn.transposed() # TODO: why transposing?
        tbn.resize_4x4()
        bm.free()
        p = p + n * 0.1 # NOTE: IMPORANT: add small offset from surface so that intersection is not done on that surface again!
        return (p, n, w, tbn)

# starting_p - (p,n,w,tbn)
# TODO: add multiple base objects that are in the scene! NOTE: N outside vs N inside!
def raytrace(intersectable_objects, starting_point, hemisphere_sample_tangent_space, n_steps_depth, path=[]):
    p = starting_point[0]
    n = starting_point[1]
    #draw_vector(p,n)
    tnb = starting_point[3]
    path.append(mathutils.Vector(p))
    hemisphere_sample = p + tnb @ hemisphere_sample_tangent_space * 1.0 # 1.0 - radius
    ray = hemisphere_sample - p
    ray.normalize()
    #draw_vector(p, ray)
    # Perform raycasting for all intersectable objects. Take first intersection (if exists)
    intersection_point = (None, None, None, None)
    for intersectable_object in intersectable_objects:
        intersection_point = raycast(intersectable_object, p, ray) 
        if intersection_point[0] != None:
            break
    if intersection_point[0] == None or n_steps_depth <= 0:
        return path
    else:
        tangent_hemisphere_samples = generate_hemisphere_samples_tangent_space(1)
        raytrace(intersectable_objects, intersection_point, tangent_hemisphere_samples[0], n_steps_depth-1, path)
        return path

# Generate n_main_samples on hemispere oriented in normal direction of starting_point. For each sample
# raytrace path with n_steps_depth. Take in accont all objects in list intersectable_objects.
# starting_point - (p,n,w,tbn)
# Result is list of lists of paths. Number of lists is n_main_samples.
def hemisphere_raycasting(intersectable_objects, starting_point, n_main_samples, n_steps_depth, sphere_samples=False):
    n = starting_point[1]
    p = starting_point[0] + n * 0.1 # NOTE: IMPORANT: add small offset from surface so that intersection is not done on same surface again!
    w = starting_point[2]
    tbn = starting_point[3]
    paths = [] # list of path lists
    samples_tangent_space = []
    if sphere_samples:
        print("SPHERE SAMPLES!")
        samples_tangent_space = generate_sphere_samples_tangent_space(n_main_samples)
    else:
        samples_tangent_space = generate_hemisphere_samples_tangent_space(n_main_samples)
    for hemisphere_sample_tangent_space in samples_tangent_space:
        path = raytrace(intersectable_objects, (p,n,w,tbn), hemisphere_sample_tangent_space, n_steps_depth, [])
        paths.append(path)
    return paths

################################################################################################
#
# PATH DECORATORS.
#
################################################################################################

# bmesh
def create_edges_from_line_points(line_points, obj_name="line", col_name=None):
    bm = bmesh.new()
    prev_line_point = line_points[0]
    for i in range(1, len(line_points)):
        curr_bm_vert = bm.verts.new(line_points[i])
        prev_bm_vert = bm.verts.new(prev_line_point)
        bm.edges.new((curr_bm_vert, prev_bm_vert))
        prev_line_point = line_points[i]
    mesh = bpy.data.meshes.new(obj_name+"_mesh")
    bm.to_mesh(mesh)
    obj = bpy.data.objects.new(obj_name, mesh)
    if col_name == None:
        bpy.context.collection.objects.link(obj) # use active/selected collection
    else:
        create_collection_if_not_exists(col_name)
        bpy.data.collections[col_name].objects.link(obj)
    bm.free()

def create_triangles_from_line_points(line_points, obj_name="triangle_line", col_name=None):
    bm = bmesh.new()
    prev_line_point = line_points[0]
    for i in range(1, len(line_points)):
        curr_line_point = line_points[i]
        curr_bm_vert = bm.verts.new(curr_line_point)
        prev_bm_vert = bm.verts.new(prev_line_point)
        middle_vert = linear_bezier(prev_line_point, curr_line_point, 0.5)
        middle_vert[2] = min(prev_line_point[2], curr_line_point[2]) + 3.0
        middle_bm_vert = bm.verts.new(middle_vert)
        bm.faces.new((curr_bm_vert, middle_bm_vert, prev_bm_vert))
        prev_line_point = line_points[i]
    mesh = bpy.data.meshes.new(obj_name+"_mesh")
    bm.to_mesh(mesh)
    obj = bpy.data.objects.new(obj_name, mesh)
    if col_name == None:
        bpy.context.collection.objects.link(obj) # use active/selected collection
    else:
        create_collection_if_not_exists(col_name)
        bpy.data.collections[col_name].objects.link(obj)
    bm.free()

# https://en.wikipedia.org/wiki/Spring_(mathematics)
# u = [0, 2 * pi * n], n from R
# v = [0, 2pi]
# TODO: figure out how to draw...
def spring(R, r, P, u, v):
    x = (R + r * math.cos(v)) * math.cos(u)
    y = (R + r * math.cos(v)) * math.sin(u)
    z = r * math.sin(v) + ((P * u) / math.pi)
    return mathutils.Vector((x,y,z))

# https://en.wikipedia.org/wiki/Helix
# Helix in x-y growing in z.
# t - parameter
# r - radius
# f - phase
# P - amount of squashness
def helix(t, r, P, f):
    x = r * math.cos(t + f)
    y = r * math.sin(t + f)
    z = P * t
    return mathutils.Vector((x,y,z))

# length - float
# resolution - int
def create_helix_points(r, P, f, length, resolution):
    points = []
    for t in np.linspace(0, length, resolution):
        point = helix(t, r, P, f)
        points.append(point)
    return points

def create_transform_helix_from_two_points(r, P, f, resolution, line_point_a, line_point_b):
    helix_segment_points = []
    # Create helix points locally: in x-y moving into z.
    line_vec = mathutils.Vector(line_point_b - line_point_a)
    line_length = line_vec.length
    helix_points = create_helix_points(r, P, f, line_length, resolution)
    # Transform helix points to line.
    line_dir = line_vec.normalized()
    t, b = pixar_onb(line_dir) # or t,b = coordsys_from_vector(line_dir)
    tbn = mathutils.Matrix((t, b, line_dir)) # NOTE: using pixar_onb()?
    tbn = tbn.transposed() # TODO: why transposing?
    tbn.resize_4x4()
    mat_trans = mathutils.Matrix.Translation(line_point_a)
    for point in helix_points:
        point_transformed = mat_trans @ tbn @ point 
        helix_segment_points.append(mathutils.Vector(point_transformed))
    return helix_segment_points

def create_helix_on_path(path, r, P, f, resolution):
    helix_path_points = []
    prev_path_point = path[0]
    for i in range(1, len(path)):
        curr_path_point = path[i]
        helix_segment_points = create_transform_helix_from_two_points(r, P, f, resolution, prev_path_point, curr_path_point)
        helix_path_points.extend(helix_segment_points)
        prev_path_point = path[i]
    return helix_path_points

# https://behreajj.medium.com/scripting-curves-in-blender-with-python-c487097efd13
def append_spline_curve(curve_data=None,
                        resolution=12,
                        count=2,
                        handle_type='FREE',
                        origin=mathutils.Vector((-1.0, 0.0, 0.0)),
                        destination=mathutils.Vector((1.0, 0.0, 0.0))):
    spline = curve_data.splines.new(type='BEZIER')
    spline.use_cyclic_u = False
    spline.resolution_u = resolution
    knots = spline.bezier_points
    knots.add(count=count - 1)
    knots_range = range(0, count, 1)

    # Convert number of points in the array to a percent.
    to_percent = 1.0 / (count - 1)
    one_third = 1.0 / 3.0

    # Loop through bezier points.
    for i in knots_range:

        # Cache shortcut to current bezier point.
        knot = knots[i]

        # Calculate bezier point coordinate.
        step = i * to_percent
        knot.co = origin.lerp(destination, step)

        # Calculate left handle by subtracting 1/3 from step.
        step = (i - one_third) * to_percent
        knot.handle_left = origin.lerp(destination, step)

        # Calculate right handle by adding 1/3 to step.
        step = (i + one_third) * to_percent
        knot.handle_right = origin.lerp(destination, step)

        # Assign handle types: ['FREE', 'VECTOR', 'ALIGNED', 'AUTO']
        knot.handle_left_type = handle_type
        knot.handle_right_type = handle_type

    return spline

# path - a list of 3d points [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3), etc.]
def create_display_curve_from_path(path=[], name="bezier_curve", bevel=0.05, collection_name=None):
    # Types: ['CURVE', 'SURFACE', 'FONT']
    curve_data = data.curves.new(name=name, type='CURVE')
    # Dimensions: ['2D', '3D']
    curve_data.dimensions = "3D"
    # Fill curve data.
    if len(path) > 0:
        prev_path_point = path[0]
        for i in range(1, len(path)):
            append_spline_curve(curve_data=curve_data, origin=mathutils.Vector(prev_path_point), destination=mathutils.Vector(path[i]))
            prev_path_point = path[i]
    # Add curve properties.
    curve_data.bevel_depth = bevel
    # Create object from the data.
    curve_object = data.objects.new(name=name, object_data=curve_data)
    # Store to collection.
    if collection_name == None:
        bpy.context.collection.objects.link(curve_object)
    else:
        create_collection_if_not_exists(collection_name)
        bpy.data.collections[collection_name].objects.link(curve_object)
    return curve_object


################################################################################################
#
# BASE ELEMENTS AND INSTANCING.
#
################################################################################################
def create_instance(base_obj,
                    translate=mathutils.Vector((0,0,0)), 
                    scale=1.0,
                    rotate=("Z", 0.0),
                    basis=mathutils.Matrix.Identity(4),
                    tbn=mathutils.Matrix.Identity(4),
                    collection_name=None):
    # Create instance.
    inst_obj = bpy.data.objects.new(base_obj.name+"_inst", base_obj.data)
    # Perform translation, rotation, scaling and moving to target coord system for instance.
    mat_rot = mathutils.Matrix.Rotation(math.radians(rotate[1]), 4, rotate[0])
    mat_trans = mathutils.Matrix.Translation(translate)
    mat_sca = mathutils.Matrix.Scale(scale, 4) # TODO: figure out how to scale in given vector direction
    # TODO: If I am using `tbn` as basis then it sould go last, If I use `matrix_basis` as basis then it should go first.
    # `tbn` matrix is usually constructed for samples on base geometry using triangle normal. Therefore, it only contains
    # information about rotation.
    inst_obj.matrix_basis = basis @ mat_trans @ mat_rot @ mat_sca @ tbn  # TODO: is matrix_basis correct to be used for this?
    # Store to collection.
    if collection_name == None:
        bpy.context.collection.objects.link(inst_obj)
    else:
        create_collection_if_not_exists(collection_name)
        bpy.data.collections[collection_name].objects.link(inst_obj)
    return inst_obj

def instance_on_path(base_obj, path=[]):
    prev_point = path[0]
    for i in range(1, len(path)):
        # Calculate transformations.
        curr_point = path[i]
        n = mathutils.Vector(curr_point - prev_point)
        n.normalize()
        t, b = pixar_onb(n)
        tbn = mathutils.Matrix((t, b, n)) # NOTE: using pixar_onb()?
        tbn = tbn.transposed() # TODO: why transposing?
        tbn.resize_4x4()
        # instance with transform.
        scale = (0.9 - 0.4) * mathutils.noise.random() + 0.5
        create_instance(base_obj,
                    translate=prev_point, 
                    scale=scale,
                    rotate=("Z", 0.0),
                    basis=mathutils.Matrix.Identity(4),
                    tbn=tbn,
                    collection_name=None)
        prev_point = path[i]

def create_icosphere(radius=1.0):
    bm = bmesh.new()
    # Create icosphere.
    # https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.create_icosphere
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1, matrix=mathutils.Matrix.Identity(4), calc_uvs=False)
    object_mesh = bpy.data.meshes.new("ico_sphere_mesh")
    bm.to_mesh(object_mesh)
    obj = bpy.data.objects.new("ico_sphere_obj", object_mesh)
    bpy.context.collection.objects.link(obj)
    bm.free()
    return obj

def create_penta_sphere(radius=1.0, location=mathutils.Vector((0,0,0)), name="penta_sphere"):
    bm = bmesh.new()
    # Create icosphere.
    # https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.create_icosphere
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1, matrix=mathutils.Matrix.Identity(4), calc_uvs=False)
    # From icosphere create pentasphere.
    # https://blender.stackexchange.com/a/780
    # https://en.wikipedia.org/wiki/Dual_polyhedron
    # For icosphere of radius=1, edges must be beveled in range [0.29,0.3] so we obtain pentasphere!
    bmesh.ops.bevel(bm, geom=(bm.edges), offset=0.29, affect="EDGES")
    # Obtain "clean" pentasphere while bevel introduces additional vertices!
    #bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.05)
    object_mesh = bpy.data.meshes.new(name + "_mesh")
    bm.to_mesh(object_mesh)
    bm.free()
    obj = bpy.data.objects.new(name + "_obj", object_mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    return obj

def boolean_difference(with_object):
    bpy.ops.object.modifier_add(type="BOOLEAN")
    bpy.context.object.modifiers["Boolean"].operation = "DIFFERENCE"
    bpy.context.object.modifiers["Boolean"].object = with_object
    bpy.context.object.modifiers["Boolean"].solver = "EXACT" # TODO: approx?
    bpy.context.object.modifiers["Boolean"].use_self = False
    bpy.ops.object.modifier_apply(modifier="Boolean")

# https://blender.stackexchange.com/questions/115397/extrude-in-python
# https://docs.blender.org/api/current/bmesh.ops.html
# hollow_size [0,1]
def create_penta_sphere_hollow2(location=mathutils.Vector((0,0,0)), name="penta_sphere_hollow2", hole_size=0.5, hole_scale=0.1):
    # NOTE: calculations of geometry are done in (0,0,0), no rotation, with scale 1! Later, object is transformed.
    # Create base pentasphere.
    base_obj=create_penta_sphere(radius=1.0, location=mathutils.Vector((0,0,0)), name=name)
    #  Create detail pentasphere.
    bm = bmesh.new()
    # Create icosphere.
    # https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.create_icosphere
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1, matrix=mathutils.Matrix.Identity(4), calc_uvs=False)
    # From icosphere create pentasphere.
    # https://blender.stackexchange.com/a/780
    # https://en.wikipedia.org/wiki/Dual_polyhedron
    # For icosphere of radius=1, edges must be beveled in range [0.29,0.3] so we obtain pentasphere!
    bmesh.ops.bevel(bm, geom=(bm.edges), offset=0.3, affect="EDGES")
    # Obtain "clean" pentasphere while bevel introduces additional vertices!
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.05)
    # Extrude faces in normal direction
    efaces = bmesh.ops.extrude_discrete_faces(bm, faces=bm.faces)
    for eface in efaces["faces"]:
        bmesh.ops.translate(bm,verts=eface.verts,vec=eface.normal*6.0)
    # Scale faces in place.
    # https://blender.stackexchange.com/questions/121123/using-python-and-bmesh-to-scale-resize-a-face-in-place
    """
    for face in bm.faces:
        face_center = face.calc_center_median()
        for v in face.verts:
            #v.co = face_center + hole_scale * (v.co - face_center)
            v_offset_dir = (v.co - face_center)
            v_offset_dir.normalize()
            v.co = v.co + hole_scale * v_offset_dir
    """
    # Create mesh object from bmesh.
    detail_mesh = bpy.data.meshes.new("detail_mesh")
    bm.to_mesh(detail_mesh)
    detail_obj = bpy.data.objects.new("detail_obj", detail_mesh)
    detail_obj.scale = mathutils.Vector((hole_size,hole_size,hole_size)) # [0.1, 0.8] - see the extrude length
    bpy.context.collection.objects.link(detail_obj)
    bm.free()
    # Perform boolean difference.
    select_activate_only([base_obj])
    boolean_difference(detail_obj)
    bpy.data.objects.remove(detail_obj, do_unlink=True)
    base_obj.location = location
    return base_obj

# Create shader for given material.
# https://vividfax.github.io/2021/01/14/blender-materials.html
# shader_type = {"glossy", "diffuse", "glass"}
def create_shader(dest_mat, shader_type="diffuse", color=(0.8, 0.54519, 0.224999, 1), roughness=0.2, ior=1.45):

    # Obtain shader nodes and links.
    nodes = dest_mat.node_tree.nodes
    links = dest_mat.node_tree.links
    # Create output (surface) node.
    output = nodes.new(type='ShaderNodeOutputMaterial')

    # Create BSDF.
    if shader_type == "glossy":
        shader = nodes.new(type='ShaderNodeBsdfGlossy')
        nodes["Glossy BSDF"].inputs[0].default_value = color
        nodes["Glossy BSDF"].inputs[1].default_value = roughness
    elif shader_type == "diffuse":
        shader = nodes.new(type='ShaderNodeBsdfDiffuse')
        nodes["Diffuse BSDF"].inputs[0].default_value = color
    elif shader_type == "glass":
        shader = nodes.new(type='ShaderNodeBsdfGlass')
        nodes["Glass BSDF"].inputs[0].default_value = color
        nodes["Glass BSDF"].inputs[1].default_value = roughness
        nodes["Glass BSDF"].inputs[2].default_value = ior

    # Create links.
    links.new(shader.outputs[0], output.inputs[0])

# Create material, create shader and assign it to the object.
def assign_new_material(base_obj, shader_type="diffuse", color=(0.8, 0.54519, 0.224999, 1), roughness=0.2, ior=1.45, mat_name="mat"):
    # Create new material.
    mat = bpy.data.materials.new(name=mat_name)
    # Create Shader.
    mat.use_nodes = True
    if mat.node_tree:
        mat.node_tree.links.clear()
        mat.node_tree.nodes.clear()
    create_shader(mat, shader_type, color, roughness, ior)
    # Assign material to object.
    base_obj.data.materials.append(mat)

def create_point_light(location=mathutils.Vector((0,0,0)), color=mathutils.Vector((0.823102, 0.285278, 0.118767)), intensity=20.0):
    # Create new light datablock.
    light_data = bpy.data.lights.new(name="point_light_data", type='POINT')
    # Create new object with our light datablock.
    light_object = bpy.data.objects.new(name="point_light_object", object_data=light_data)
    # Link to the scene.
    bpy.context.collection.objects.link(light_object)
    # Place light to a specified location.
    light_object.location = location
    # Specify color
    light_object.data.color = color
    # Specify intensity
    light_object.data.energy = intensity
    return light_object




#
# Main.
#
def main():


    ##
    ## Lines between sampled mesh and mesh vertices.
    ##
    """
    # Sample triangulated mesh.
    base_obj = bpy.context.selected_objects[0]
    triangulate(base_obj) # we work with triangulated meshes
    mesh_face_samples = mesh_uniform_weighted_sampling(base_obj=base_obj, n_samples=20, base_density=3.0, use_weight_paint=False)
    # get_object_vertices.
    mesh_vertices = get_object_vertices_positions(bpy.context.scene.objects["vert_sampling_coll"])
    # Set.
    sets = []
    sets.append(mesh_face_samples)
    sets.append(mesh_vertices)
    sets.sort(key=lambda x:len(x))
    # Base object for instancing.
    penta_sphere_hollow = create_penta_sphere_hollow2()
    # Create bezier
    bezier_lines_points = create_bezier_line_points_from_two_sets(sets[0], sets[1], 0.2)
    for bezier_line_points in bezier_lines_points:
        if len(bezier_line_points) < 1:
            print("1 or less bezier point! Ignore the path")
            continue
        helix_path_points1 = create_helix_on_path(path=bezier_line_points, r=1, P=1, f=0.0, resolution=10)
        create_display_curve_from_path(path=helix_path_points1, bevel=0.01)
        for point in helix_path_points1:
                scale = (0.4 - 0.2) * mathutils.noise.random() + 0.2
                create_instance(penta_sphere_hollow,
                        translate=point,
                        scale=scale,
                        rotate=("Z", 0.0),
                        basis=mathutils.Matrix.Identity(4),
                        tbn=mathutils.Matrix.Identity(4),
                        collection_name=None)
    """


    ##
    ## Lines between two sampled meshes
    ##
    # Gather samples.
    """
    samples_all = []
    for i in range(len(bpy.context.selected_objects)):
        base_obj = bpy.context.selected_objects[i]
        triangulate(base_obj) # we work with triangulated meshes
        samples = mesh_uniform_weighted_sampling(base_obj=base_obj, n_samples=30, base_density=5, use_weight_paint=True)
        samples_all.append(samples)
    # Base object.
    penta_sphere_hollow = create_penta_sphere_hollow2()
    # Build bezier lines between samples.
    samples_all.sort(key=lambda x:len(x)) # NOTE: sometimes number of samples won't be the same!
    bezier_lines_points = create_bezier_line_points_from_two_sets(samples_all[0], samples_all[1], 0.1)
    # Add display elements.
    for bezier_line_points in bezier_lines_points:
        #create_edges_from_line_points(bezier_line_points)
        if len(bezier_line_points) < 2:
            print("Not enough bezier points! Ignore!")
            continue
        helix_path_points1 = create_helix_on_path(path=bezier_line_points, r=1, P=1, f=0.0, resolution=40)
        helix_path_points2 = create_helix_on_path(path=bezier_line_points, r=1, P=1, f=4.0, resolution=40)
        create_display_curve_from_path(path=helix_path_points1)
        create_display_curve_from_path(path=helix_path_points2)
        for point in helix_path_points1:
                scale = (0.5 - 0.2) * mathutils.noise.random() + 0.2
                create_instance(penta_sphere_hollow,
                        translate=point,
                        scale=scale,
                        rotate=("Z", 0.0),
                        basis=mathutils.Matrix.Identity(4),
                        tbn=mathutils.Matrix.Identity(4),
                        collection_name=None)
    """

    ##
    ## Raytracing paths.
    ##
    # Obtain intersectable objects and mark where the intersection should take place.
    # https://blenderartists.org/t/how-to-select-all-objects-of-a-known-collection-with-python/1195742
    intersectable_objects = [] # (obj, <normal_dir>)
    for obj in bpy.data.collections['normal_inside'].all_objects:
        intersectable_objects.append((obj, "normal_inside"))
    for obj in bpy.data.collections['normal_outside'].all_objects:
        intersectable_objects.append((obj, "normal_outside"))
    # Create starting samples.
    samples_all = []
    # Samples from intersectable objects.
    if False:
        for intersectable_object in intersectable_objects:
            triangulate(intersectable_object[0])
            samples = mesh_uniform_weighted_sampling(base_obj=intersectable_object[0], n_samples=20, base_density=2)
            for sample in samples:
                if intersectable_object[1] == "normal_inside":
                    # Invert normal!
                    sample[1] = -sample[1]
                    # Recalculate tnb with new normal orientation!
                    t, b = pixar_onb(sample[1])
                    tbn = mathutils.Matrix((t, b, sample[1] )) # NOTE: using pixar_onb()?
                    tbn = tbn.transposed() # TODO: why transposing?
                    tbn.resize_4x4()
                    sample[3] = tbn  
                samples_all.append(sample)
    # Samples from special objects.
    special_objects = []
    for obj in bpy.data.collections['special_objects'].all_objects:
        triangulate(obj)
        samples = mesh_uniform_weighted_sampling(base_obj=obj, n_samples=1, base_density=2)
        samples_all.extend(samples)

    # Create base object for instancing.
    penta_sphere = create_penta_sphere()
    penta_sphere_hollow = create_penta_sphere_hollow2()
    # Path-tracing for all samples, path construction, instancing...
    for sample in samples_all:
        print("SAMPLE", sample[0])
        path_tracing_paths = hemisphere_raycasting(intersectable_objects=intersectable_objects, starting_point=sample, n_main_samples=40, n_steps_depth=1, sphere_samples=True) # list of lists!
        bezier_paths_points = []
        for path_tracing_path in path_tracing_paths:
            if len(path_tracing_path) < 2: # In some cases path can not be built!
                print("Raytracing path has points < 2! Removing this path!")
                continue
            bezier_path_points = create_bezier_line_points_from_path(path_tracing_path, 0.1)
            bezier_paths_points.append(bezier_path_points)
        for bezier_path_points in bezier_paths_points:
            print("BEZIER PATH", bezier_path_points)
            print("\n")
            if len(bezier_path_points) <= 1:
                print("Empty or Bezier path with only one point! Ignore.")
                continue
            helix_path_points1 = create_helix_on_path(path=bezier_path_points, r=1, P=1, f=0.0, resolution=40)
            helix_path_points2 = create_helix_on_path(path=bezier_path_points, r=1, P=1, f=4.0, resolution=40)
            helix_path_points3 = create_helix_on_path(path=bezier_path_points, r=1, P=1, f=8.0, resolution=40)
            create_display_curve_from_path(path=helix_path_points1)
            create_display_curve_from_path(path=helix_path_points2)
            create_display_curve_from_path(path=helix_path_points3)
            all_helix_paths = []
            all_helix_paths.append(helix_path_points1)
            all_helix_paths.append(helix_path_points2)
            all_helix_paths.append(helix_path_points3)
            for helix_path in all_helix_paths:
                for point in helix_path:
                    scale = (0.9 - 0.4) * mathutils.noise.random() + 0.4
                    create_instance(penta_sphere_hollow,
                            translate=point,
                            scale=scale,
                            rotate=("Z", 0.0),
                            basis=mathutils.Matrix.Identity(4),
                            tbn=mathutils.Matrix.Identity(4),
                            collection_name=None)

#
# Script entry point.
#
if __name__ == "__main__":
    main()
