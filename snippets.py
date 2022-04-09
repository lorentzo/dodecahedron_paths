
def jitter_around_triangle_sample(triangle_vertices, base_sample, n_epsilon, epsilon, triangle_area=0.0):
    jitter_samples = []
    while True:
        if len(jitter_samples) >= n_epsilon:
            break
        u = mathutils.noise.random()
        v = mathutils.noise.random()
        w = mathutils.noise.random()
        s = u + v + w
        un = (u / s)
        vn = (v / s)
        wn = (w / s)
        p = un * triangle_vertices[0] + vn * triangle_vertices[1] + wn * triangle_vertices[2]
        base_jitter = base_sample - p
        if base_jitter.length < epsilon * triangle_area: # TODO: enhance useage of triangle area!
            # NOTE: samples are defined by their barycentric factors.
            jitter_samples.append(list((un,vn,wn)))
    return jitter_samples

#
# barycentric sampling of triangulated mesh using object.data (mesh)
# object.data (mesh) has extensive info on vertices (e.g. weight, color, etc.)
#
# `n_epsilon` - number of additional samples per sample in `epsilon` neighbourhood.
# `epsilon` - depends on triangle size and later size of growth elements.
#
def vertex_weighted_barycentric_sampling_jitter(base_obj, n_samples, n_epsilon=0, epsilon=0.0):
    samples = [] # (p, N, w, tbn)
    for polygon in base_obj.data.polygons: # must be triangulated mesh!
        triangle_vertices = []
        triangle_vertex_weights = []
        for v_idx in polygon.vertices:
            v = base_obj.data.vertices[v_idx]
            triangle_vertices.append(v.co)
            if len(v.groups) < 1:
                triangle_vertex_weights.append(0.0)
            else:
                triangle_vertex_weights.append(v.groups[0].weight) # TODO: only one group? Investigate! float in [0, 1], default 0.0
        for i in range(n_samples):
            # Find sample using barycentric sampling.
            a = mathutils.noise.random()
            b = mathutils.noise.random()
            c = mathutils.noise.random()
            s = a + b + c
            un = (a / s)
            vn = (b / s)
            wn = (c / s)
            p_original = un * triangle_vertices[0] + vn * triangle_vertices[1] + wn * triangle_vertices[2]
            # NOTE: we later use w_original to calculate w of other neighbourhood samples. This way neigh samples are not 
            # discarded due to mesh weight in that point. Other possibility is to calculate w for each neighbourhood using
            # interpolation.
            w_original = un * triangle_vertex_weights[0] + vn * triangle_vertex_weights[1] + wn * triangle_vertex_weights[2] # interpolate weight
            if w_original > 0.01: # probably an error otherwise
                # Find more samples in neighbourhood of the found one.
                all_samples = jitter_around_triangle_sample(triangle_vertices, p_original, n_epsilon, epsilon, polygon.area)
                all_samples.append(list((un,vn,wn)))
                for sample in all_samples:
                    p = sample[0] * triangle_vertices[0] + sample[1] * triangle_vertices[1] + sample[2] * triangle_vertices[2]
                    #w = sample[0] * triangle_vertex_weights[0] + sample[1] * triangle_vertex_weights[1] + sample[2] * triangle_vertex_weights[2] # interpolate weight
                    w = w_original * (1.0 - 0.5) * mathutils.noise.random() + 0.5
                    n = polygon.normal # TODO: vertex normals?
                    # Calc tangent. NOTE: use most distant point from barycentric coord to evade problems with 0
                    t = mathutils.Vector(triangle_vertices[0] - p)
                    t = t.normalized()
                    bt = n.cross(t)
                    bt = bt.normalized()
                    tbn = mathutils.Matrix((t, bt, n)) # NOTE: using pixar_onb()?
                    tbn = tbn.transposed() # TODO: why transposing?
                    tbn.resize_4x4()
                    samples.append((p,n,w,tbn))
    return samples 

# Create cube.
bm = bmesh.new()
bmesh.ops.create_cube(bm, size=1.0, matrix=mathutils.Matrix.Identity(4), calc_uvs=False)
# Scale in local space.
bmesh.ops.scale(bm, 
                vec=mathutils.Vector((1,1,5)), 
                space=mathutils.Matrix.Identity(4), 
                verts=bm.verts)
#bmesh.ops.subdivide_edges(bm, edges=bm.edges, fractal=1.5, smooth=0.2, cuts=3)
object_mesh = bpy.data.meshes.new("cube_mesh")
bm.to_mesh(object_mesh)
bm.free()
obj = bpy.data.objects.new("cube_obj", object_mesh)
#mat_sca = mathutils.Matrix.Scale(4, 4, mathutils.Vector((2,1,1)))
#mat_rot = mathutils.Matrix.Rotation(math.radians(rotate[1]), 4, rotate[0])
mat_trans = mathutils.Matrix.Translation()
obj.matrix_basis = mat_trans @ tbn #  @ mat_rot @  @ tbn
bpy.context.collection.objects.link(obj)
bm.free()

# Create edges using blender mesh: https://www.youtube.com/watch?v=mljWBuj0Gho&t=94s
# TODO: one mesh for all lines?
# Params:
#  origin_samples - list of tuples (p, N, w, tbn) resulting from `vertex_weighted_barycentric_sampling_jitter()` function
#    Note: len(origin_samples) < len(destination_samples)
#  destination_samples - list of tuples (p, N, w, tbn) resulting from `vertex_weighted_barycentric_sampling_jitter()` function
def create_lines_object(origin_samples, destination_samples, obj_name="edges_obj", col_name=None):
    verts = []
    edges = []
    faces = []
    for i in range(len(origin_samples)):
        verts.append(origin_samples[i][0])        # idx: i
        verts.append(destination_samples[i][0])   # idx: i+1
        edges.append([i, i+1])
    mesh = bpy.data.meshes.new(obj_name+"_mesh")
    mesh.from_pydata(verts, edges, faces)
    obj = bpy.data.objects.new(obj_name, mesh)
    if col_name == None:
        bpy.context.collection.objects.link(obj) # use active/selected collection
    else:
        create_collection_if_not_exists(col_name)
        bpy.data.collections[collection_name].objects.link(obj)
    return obj

# Create edges using blender bmesh: https://blender.stackexchange.com/questions/414/how-to-use-bmesh-to-add-verts-faces-and-edges-to-existing-geometry
def create_line_objects(origin_samples, destination_samples, n_cuts=1, obj_name="edges_obj", col_name=None):
    edge_objects = []
    for i in range(len(origin_samples)):
        bm = bmesh.new()
        bm_vert1 = bm.verts.new(origin_samples[i][0])
        bm_vert2 = bm.verts.new(destination_samples[i][0])
        bm.edges.new((bm_vert1, bm_vert2))
        bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=n_cuts)
        mesh = bpy.data.meshes.new(obj_name+"_mesh")
        bm.to_mesh(mesh)
        obj = bpy.data.objects.new(obj_name, mesh)
        edge_objects.append(obj)
        if col_name == None:
            bpy.context.collection.objects.link(obj) # use active/selected collection
        else:
            create_collection_if_not_exists(col_name)
            bpy.data.collections[col_name].objects.link(obj)
        bm.free()
    return edge_objects

    
# Params:
#  origin_samples - list of tuples (p, N, w, tbn) resulting from `vertex_weighted_barycentric_sampling_jitter()` function
#    Note: len(origin_samples) < len(destination_samples)
#  destination_samples - list of tuples (p, N, w, tbn) resulting from `vertex_weighted_barycentric_sampling_jitter()` function
def create_line_sample_points(origin_samples, destination_samples, n_samples):
    lines_samples = [] # list of list of line samples
    # For every pair of original_sample and destination_sample create a line of samples
    for i in range(len(origin_samples)):
        line_samples = [] # list of line samples
        orig_dest_vec = destination_samples[i][0] - origin_samples[i][0]
        orig_dest_vec_norm = orig_dest_vec.normalized()
        orig_dest_len = orig_dest_vec.length
        step_delta = orig_dest_len / (n_samples + 1)
        line_samples.append(origin_samples[i][0])
        current_sample_point = origin_samples[i][0]
        for j in range(n_samples+1): # we want n_samples between starting and ending point!
            current_sample_point = current_sample_point + orig_dest_vec_norm * step_delta
            line_samples.append(current_sample_point)
        #line_samples.append(destination_samples[i][0])
        lines_samples.append(line_samples)
    return lines_samples


# Assume gravity is in -z. TODO: add approx gravity vector
def approx_gravity_between_two_nodes(edge_objects):
    for edge_object in edge_objects:
        bm = bmesh.new()   # create an empty BMesh
        bm.from_mesh(edge_object.data)   # fill it in from a Mesh
        bm.verts.ensure_lookup_table()
        vert_first = bm.verts[0]
        vert_last = bm.verts[-1]
        middle_idx = int(len(bm.verts)/2)
        middle_point_height = min(bm.verts[0].co[2], bm.verts[1].co[2]) - 1.0
        middle_point_position = mathutils.Vector((bm.verts[middle_idx].co[0], bm.verts[middle_idx].co[1], middle_point_height))
        bm.verts[middle_idx].co = middle_point_position
        #max_t = bm.verts[0].co - bm.verts[middle_idx].co
        #max_t = max_t.length
        max_t = bm.verts[0].co[2] - bm.verts[middle_idx].co[2]
        print("max_t", max_t)
        # TODO: bmesh.subdivide: original vertices are 0 and 1, subdivided (new) are 2,3,4 and so on!
        for i in range(0, middle_idx):
            #print("***")
            #print(bm.verts[i].co)
            bpy.ops.mesh.primitive_cube_add(size=0.2, location=bm.verts[0].co, scale=(1, 1, 1))
            t = bm.verts[i].co[2] - bm.verts[middle_idx].co[2]
            t = t / max_t
            print("t",t)
            new_height = t * bm.verts[0].co[2] + (1-t) * bm.verts[middle_idx].co[2]
            new_point_position = mathutils.Vector((bm.verts[i].co[0], bm.verts[i].co[1], new_height))
            bm.verts[i].co = new_point_position
        bm.to_mesh(edge_object.data)
        bm.free()