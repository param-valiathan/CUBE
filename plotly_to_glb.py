#!/usr/bin/env python3
r"""
plotly_to_glb.py
================
Converts Plotly 3D HTML plots into high-vibrancy, single self-contained GLB files
optimized for Microsoft PowerPoint 3D presentations.

Features:
- High-saturation HSV color boosting to prevent washed-out colors under PowerPoint 3D lighting.
- Pure opaque PBR materials with metallicFactor=0.0 & roughnessFactor=0.3 for rich satin color rendering.
- Double-sided textured 3D floating badges for centroids and flow values.
- Auto-scaling 3D floor grid stage and UMAP 1, UMAP 2, UMAP 3 labels.

Usage:
    python plotly_to_glb.py <path_to_html_file> [output_glb_path]
"""

import os
import sys
import json
import base64
import math
import argparse
import colorsys
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

def boost_color_vibrancy(hex_str, alpha=1.0):
    """
    Boost color saturation and brightness using HSV color space so colors
    never appear muted or washed-out under PowerPoint 3D lighting.
    """
    if not isinstance(hex_str, str):
        return [255, 0, 128, int(alpha * 255)]
        
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 6:
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
    else:
        r, g, b = 0.5, 0.5, 0.5

    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    # Boost saturation and value for maximum pop
    s = min(1.0, s * 1.45)
    v = min(1.0, v * 1.30)
    
    # Ensure strong minimum saturation and brightness
    s = max(s, 0.82)
    v = max(v, 0.90)
    
    r_new, g_new, b_new = colorsys.hsv_to_rgb(h, s, v)
    return [int(r_new * 255), int(g_new * 255), int(b_new * 255), int(alpha * 255)]

def decode_arr(val):
    """ Decode base64 float32/float64 binary buffers or standard Python lists. """
    if isinstance(val, dict) and 'bdata' in val:
        raw = base64.b64decode(val['bdata'])
        dtype = val.get('dtype', 'f4')
        if dtype == 'f4':
            return np.frombuffer(raw, dtype=np.float32)
        elif dtype == 'f8':
            return np.frombuffer(raw, dtype=np.float64)
        elif dtype == 'i4':
            return np.frombuffer(raw, dtype=np.int32)
        else:
            return np.array([])
    elif isinstance(val, list):
        clean = [v for v in val if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return np.array(clean)
    return np.array([])

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the 4x4 rotation matrix that rotates vec1 to vec2. """
    a = (vec1 / np.linalg.norm(vec1)).reshape(3)
    b = (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(4) if c > 0 else np.diag([-1, -1, 1, 1])
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    return matrix

def make_badge_texture(text, bg_hex="#4E79A7", width=256, height=128, font_size=56, is_dark_bg=False):
    """ Render high-contrast RGBA text badge. """
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    vibrant_rgba = boost_color_vibrancy(bg_hex)
    r, g, b = vibrant_rgba[0], vibrant_rgba[1], vibrant_rgba[2]
        
    pad = 6
    if is_dark_bg:
        fill_color = (18, 18, 28, 240)
        outline_color = (r, g, b, 255)
        text_fill = (255, 255, 255, 255)
    else:
        fill_color = (r, g, b, 245)
        outline_color = (255, 255, 255, 240)
        text_fill = (255, 255, 255, 255)
        
    draw.rounded_rectangle([pad, pad, width - pad, height - pad], radius=24, fill=fill_color, outline=outline_color, width=5)
    
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
            
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (width - tw) / 2
    ty = (height - th) / 2 - bbox[1]
    
    draw.text((tx, ty), text, font=font, fill=text_fill)
    return img

def create_double_sided_badge(center_pos, text, bg_hex, quad_w=0.65, quad_h=0.32, is_dark_bg=False):
    """ Create double-sided 3D textured quad badge mesh. """
    img = make_badge_texture(text, bg_hex, is_dark_bg=is_dark_bg)
    half_w = quad_w / 2.0
    half_h = quad_h / 2.0
    
    local_verts = np.array([
        [-half_w, 0, -half_h],
        [ half_w, 0, -half_h],
        [ half_w, 0,  half_h],
        [-half_w, 0,  half_h]
    ], dtype=np.float32)
    
    rot_x = trimesh.transformations.rotation_matrix(np.radians(25), [1, 0, 0])
    local_verts = trimesh.transformations.transform_points(local_verts, rot_x)
    
    verts = local_verts + np.array(center_pos)
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [0, 2, 1], [0, 3, 2]
    ], dtype=np.int32)
    
    uvs = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ], dtype=np.float32)
    
    material = trimesh.visual.texture.SimpleMaterial(image=img)
    visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=visual)
    return mesh

def convert_plotly_html_to_glb(html_path, glb_path=None):
    """ Converts Plotly 3D HTML plot to vibrant, PowerPoint-optimized GLB. """
    if not os.path.exists(html_path):
        raise FileNotFoundError(f"Input HTML file not found: '{html_path}'")
        
    if glb_path is None:
        glb_path = os.path.splitext(html_path)[0] + ".glb"
        
    print(f"Reading Plotly figure from '{html_path}'...")
    with open(html_path, "r", encoding="utf-8") as f:
        text = f.read()

    call_idx = text.find("Plotly.newPlot(")
    if call_idx == -1:
        raise ValueError(f"Could not find 'Plotly.newPlot' call in '{html_path}'")
        
    sub = text[call_idx:]

    b_start = sub.find("[")
    depth = 0
    in_string = False
    escape = False
    end_idx = -1
    for i, c in enumerate(sub[b_start:], start=b_start):
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if not in_string:
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break

    data = json.loads(sub[b_start:end_idx])
    scene_objects = []

    # High quality icosphere primitives
    point_sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.065)
    centroid_sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.20)

    # Standard non-reflective PBR material for crisp colors
    pbr_mat = trimesh.visual.material.PBRMaterial(metallicFactor=0.0, roughnessFactor=0.35)

    # Calculate spatial bounds ignoring NaNs
    all_x, all_y, all_z = [], [], []
    for t in data:
        x_raw = decode_arr(t.get('x'))
        y_raw = decode_arr(t.get('y'))
        z_raw = decode_arr(t.get('z'))
        if len(x_raw) > 0:
            valid_mask = ~np.isnan(x_raw) & ~np.isnan(y_raw) & ~np.isnan(z_raw)
            if np.any(valid_mask):
                all_x.extend(x_raw[valid_mask])
                all_y.extend(y_raw[valid_mask])
                all_z.extend(z_raw[valid_mask])

    if len(all_x) == 0:
        raise ValueError("No valid 3D data points found in Plotly JSON data.")

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_z = np.array(all_z)
    
    x_min, x_max = all_x.min() - 0.5, all_x.max() + 0.5
    y_min, y_max = all_y.min() - 0.5, all_y.max() + 0.5
    z_min, z_max = all_z.min() - 0.5, all_z.max() + 0.5

    # 1. Scatter Points (Clusters) with 100% full opacity & vibrant color boost
    for i, t in enumerate(data):
        t_type = t.get('type')
        name = str(t.get('name', ''))
        mode = str(t.get('mode', ''))
        
        x = decode_arr(t.get('x'))
        y = decode_arr(t.get('y'))
        z = decode_arr(t.get('z'))
        
        if len(x) == 0:
            continue
            
        if t_type == 'scatter3d' and ('markers' in mode) and name != 'Centroids':
            marker_color = t.get('marker', {}).get('color', '#4E79A7')
            if isinstance(marker_color, str):
                color_hex = marker_color
            elif isinstance(marker_color, list) and len(marker_color) > 0:
                color_hex = marker_color[0]
            else:
                color_hex = '#4E79A7'
                
            # Use full 100% alpha (255) to prevent lighting wash-out
            rgba = boost_color_vibrancy(color_hex, alpha=1.0)
            
            vertices_list = []
            faces_list = []
            v_offset = 0
            
            base_verts = point_sphere.vertices
            base_faces = point_sphere.faces
            
            for px, py, pz in zip(x, y, z):
                if np.isnan(px) or np.isnan(py) or np.isnan(pz):
                    continue
                vertices_list.append(base_verts + np.array([px, py, pz]))
                faces_list.append(base_faces + v_offset)
                v_offset += len(base_verts)
                
            if len(vertices_list) > 0:
                all_verts = np.vstack(vertices_list)
                all_faces = np.vstack(faces_list)
                
                mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
                mesh.visual.vertex_colors = np.tile(rgba, (len(all_verts), 1))
                mesh.visual.material = pbr_mat
                mesh.metadata['name'] = f"Cluster_{name}"
                scene_objects.append(mesh)

    # 2. Centroids & Floating Text Badges
    for t in data:
        if t.get('type') == 'scatter3d' and t.get('name') == 'Centroids':
            x = decode_arr(t.get('x'))
            y = decode_arr(t.get('y'))
            z = decode_arr(t.get('z'))
            colors = t.get('marker', {}).get('color', [])
            labels = t.get('text', [])
            
            for idx, (px, py, pz) in enumerate(zip(x, y, z)):
                if np.isnan(px) or np.isnan(py) or np.isnan(pz):
                    continue
                c_hex = colors[idx] if idx < len(colors) and isinstance(colors, list) else '#FFFFFF'
                c_label = labels[idx] if idx < len(labels) and isinstance(labels, list) else f"C{idx}"
                rgba = boost_color_vibrancy(c_hex, alpha=1.0)
                
                c_mesh = centroid_sphere.copy()
                c_mesh.apply_translation([px, py, pz])
                c_mesh.visual.vertex_colors = np.tile(rgba, (len(c_mesh.vertices), 1))
                c_mesh.visual.material = pbr_mat
                c_mesh.metadata['name'] = f"Centroid_{c_label}"
                scene_objects.append(c_mesh)
                
                badge_pos = [px, py - 0.05, pz + 0.40]
                badge_mesh = create_double_sided_badge(badge_pos, str(c_label), c_hex, quad_w=0.68, quad_h=0.34)
                badge_mesh.metadata['name'] = f"Badge_{c_label}"
                scene_objects.append(badge_mesh)

    # 3. Transition Lines (3D Tubes)
    for i, t in enumerate(data):
        t_type = t.get('type')
        mode = str(t.get('mode', ''))
        x = decode_arr(t.get('x'))
        y = decode_arr(t.get('y'))
        z = decode_arr(t.get('z'))
        
        if len(x) == 0:
            continue
            
        if t_type == 'scatter3d' and mode == 'lines':
            color_hex = t.get('line', {}).get('color', '#9C27B0')
            line_w = t.get('line', {}).get('width', 6)
            rgba = boost_color_vibrancy(color_hex, alpha=1.0)
            
            if len(x) >= 2 and not np.isnan(x[0]) and not np.isnan(x[1]):
                p1 = np.array([x[0], y[0], z[0]])
                p2 = np.array([x[1], y[1], z[1]])
                vec = p2 - p1
                dist = np.linalg.norm(vec)
                if dist > 1e-4:
                    tube_r = 0.025 + (line_w / 10.0) * 0.03
                    cyl = trimesh.creation.cylinder(radius=tube_r, height=dist, sections=12)
                    T = rotation_matrix_from_vectors(np.array([0, 0, 1]), vec)
                    T[:3, 3] = (p1 + p2) / 2.0
                    cyl.apply_transform(T)
                    cyl.visual.vertex_colors = np.tile(rgba, (len(cyl.vertices), 1))
                    cyl.visual.material = pbr_mat
                    cyl.metadata['name'] = f"Transition_Tube_{i}"
                    scene_objects.append(cyl)

        # 4. Cones (Arrowheads)
        elif t_type == 'cone':
            color_scale = t.get('colorscale', [[0, '#9C27B0']])
            color_hex = color_scale[0][1] if isinstance(color_scale, list) and len(color_scale) > 0 else '#9C27B0'
            rgba = boost_color_vibrancy(color_hex, alpha=1.0)
            
            u = decode_arr(t.get('u'))
            v = decode_arr(t.get('v'))
            w = decode_arr(t.get('w'))
            
            if len(x) > 0 and len(u) > 0:
                tip_x, tip_y, tip_z = x[0], y[0], z[0]
                dir_vec = np.array([u[0], v[0], w[0]])
                dir_norm = np.linalg.norm(dir_vec)
                
                if dir_norm > 1e-4:
                    dir_unit = dir_vec / dir_norm
                    cone_h, cone_r = 0.42, 0.18
                    cone = trimesh.creation.cone(radius=cone_r, height=cone_h, sections=16)
                    T = rotation_matrix_from_vectors(np.array([0, 0, 1]), dir_unit)
                    center_pos = np.array([tip_x, tip_y, tip_z]) - dir_unit * (cone_h / 2.0)
                    T[:3, 3] = center_pos
                    cone.apply_transform(T)
                    cone.visual.vertex_colors = np.tile(rgba, (len(cone.vertices), 1))
                    cone.visual.material = pbr_mat
                    cone.metadata['name'] = f"Arrowhead_{i}"
                    scene_objects.append(cone)

        # 5. Transition Value Text Badges
        elif t_type == 'scatter3d' and mode == 'text' and 'text' in t:
            text_vals = t.get('text', [])
            if len(text_vals) > 0 and len(x) > 0 and not np.isnan(x[0]):
                val_str = str(text_vals[0])
                pos = [x[0], y[0], z[0] + 0.15]
                badge = create_double_sided_badge(pos, val_str, "#FF007F", quad_w=0.55, quad_h=0.28, is_dark_bg=True)
                badge.metadata['name'] = f"Transition_Value_{val_str}"
                scene_objects.append(badge)

    # 6. Floor Grid Stage & Axis Badges
    z_floor = z_min - 0.2
    grid_rgba = [140, 150, 175, 255]
    
    step_x = max(1.0, math.ceil((x_max - x_min) / 10.0))
    step_y = max(1.0, math.ceil((y_max - y_min) / 10.0))

    for gx in np.arange(x_min, x_max + 0.1, step_x):
        p1 = np.array([gx, y_min, z_floor])
        p2 = np.array([gx, y_max, z_floor])
        vec = p2 - p1
        d = np.linalg.norm(vec)
        cyl = trimesh.creation.cylinder(radius=0.015, height=d, sections=8)
        T = rotation_matrix_from_vectors(np.array([0, 0, 1]), vec)
        T[:3, 3] = (p1 + p2) / 2.0
        cyl.apply_transform(T)
        cyl.visual.vertex_colors = np.tile(grid_rgba, (len(cyl.vertices), 1))
        cyl.visual.material = pbr_mat
        scene_objects.append(cyl)

    for gy in np.arange(y_min, y_max + 0.1, step_y):
        p1 = np.array([x_min, gy, z_floor])
        p2 = np.array([x_max, gy, z_floor])
        vec = p2 - p1
        d = np.linalg.norm(vec)
        cyl = trimesh.creation.cylinder(radius=0.015, height=d, sections=8)
        T = rotation_matrix_from_vectors(np.array([0, 0, 1]), vec)
        T[:3, 3] = (p1 + p2) / 2.0
        cyl.apply_transform(T)
        cyl.visual.vertex_colors = np.tile(grid_rgba, (len(cyl.vertices), 1))
        cyl.visual.material = pbr_mat
        scene_objects.append(cyl)

    axis_label_1 = create_double_sided_badge([(x_min + x_max)/2, y_min - 0.5, z_floor], "UMAP 1", "#1E1E2E", quad_w=1.2, quad_h=0.4, is_dark_bg=True)
    axis_label_2 = create_double_sided_badge([x_min - 0.6, (y_min + y_max)/2, z_floor], "UMAP 2", "#1E1E2E", quad_w=1.2, quad_h=0.4, is_dark_bg=True)
    axis_label_3 = create_double_sided_badge([x_min - 0.6, y_min - 0.5, (z_min + z_max)/2], "UMAP 3", "#1E1E2E", quad_w=1.2, quad_h=0.4, is_dark_bg=True)
    scene_objects.extend([axis_label_1, axis_label_2, axis_label_3])

    scene = trimesh.Scene(scene_objects)
    glb_bytes = scene.export(file_type='glb')
    with open(glb_path, "wb") as f:
        f.write(glb_bytes)
        
    print(f"SUCCESS: Saved vibrant single GLB ({os.path.getsize(glb_path)/1024:.1f} KB) -> '{glb_path}'")
    return glb_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert Plotly 3D HTML figure to high-vibrancy PowerPoint-ready GLB model.")
    parser.add_argument("html_path", help="Path to input .html file")
    parser.add_argument("glb_path", nargs="?", default=None, help="Optional output .glb file path")
    
    args = parser.parse_args()
    convert_plotly_html_to_glb(args.html_path, args.glb_path)
