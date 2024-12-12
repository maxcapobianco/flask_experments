import os
import numpy as np
from datetime import datetime
from flask import Flask, Response, render_template, jsonify, request, send_from_directory, redirect, url_for
import matplotlib.pyplot as plt
import json
import open3d as o3d

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def process_point_cloud_with_voxel_coloring(file_path, voxel_size=1.5, threshold=10):
    """
    Process a point cloud file, apply voxel grid filtering, and color the voxels
    based on the number of points inside each voxel.
    """
    try:
        # Read point cloud
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors(
        ) else np.ones((len(points), 3))

        # Convert points to voxel grid coordinates
        voxel_coords = (points / voxel_size).astype(int)

        # Count points in each voxel
        voxel_dict = {}
        for i, coord in enumerate(voxel_coords):
            coord_tuple = tuple(coord)
            if coord_tuple not in voxel_dict:
                voxel_dict[coord_tuple] = {
                    'count': 0, 'points': []
                }
            voxel_dict[coord_tuple]['count'] += 1
            voxel_dict[coord_tuple]['points'].append(points[i])

        # Filter voxels based on threshold
        filtered_voxels = []
        max_count = 0
        for voxel, data in voxel_dict.items():
            if data['count'] >= threshold:
                # Compute the voxel center (average of points in the voxel)
                avg_point = np.mean(data['points'], axis=0)
                filtered_voxels.append((avg_point, data['count']))
                max_count = max(max_count, data['count'])

        # Normalize colors based on voxel counts
        voxel_points = [voxel[0] for voxel in filtered_voxels]
        voxel_counts = [voxel[1] for voxel in filtered_voxels]

        # Map counts to colors
        # Normalize counts between 0 and 1
        norm_counts = np.array(voxel_counts) / max_count
        cmap = plt.get_cmap('viridis')  # Use a colormap (e.g., viridis)
        # Map normalized values to RGB
        voxel_colors = [cmap(norm)[:3] for norm in norm_counts]

        return {
            'points': np.array(voxel_points).tolist(),
            'colors': np.array(voxel_colors).tolist()
        }
    except Exception as e:
        print(f"Error processing point cloud: {e}")
        return None


def process_point_cloud(file_path, voxel_size=0.2):
    """
    Process a point cloud file and convert to JSON with voxel grid filtering.
    """
    try:
        # Read point cloud
        pcd = o3d.io.read_point_cloud(file_path)

        # Downsample using voxel grid
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        # Convert to numpy arrays
        points = np.asarray(downsampled_pcd.points)
        colors = np.asarray(downsampled_pcd.colors) if pcd.has_colors(
        ) else np.ones((len(points), 3))

        return {
            'points': points.tolist(),
            'colors': colors.tolist()
        }
    except Exception as e:
        print(f"Error processing point cloud: {e}")
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_pointcloud():
    """
    Handle point cloud file upload
    """
    if 'pointcloud' not in request.files:
        return redirect(url_for('index'))

    file = request.files['pointcloud']

    if file.filename == '':
        return redirect(url_for('index'))

    if file and file.filename.endswith('.pcd'):
        # Save the file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Redirect to visualization with the uploaded file
        return render_template('pointcloud.html',
                               data_route='load_uploaded_pointcloud',
                               filename=file.filename)

    return redirect(url_for('index'))


@app.route('/load_uploaded_pointcloud', methods=['POST'])
def load_uploaded_pointcloud():
    start_time = datetime.now()
    print(f"Start time: {start_time}")

    filename = request.form.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'})

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'})

    print(f"File path check complete: {datetime.now() - start_time}")

    processed_data = process_point_cloud_with_voxel_coloring(file_path)

    print(f"Point cloud processing complete: {datetime.now() - start_time}")

    if processed_data:
        def generate():
            yield json.dumps(processed_data, separators=(',', ':'))

        response = Response(generate(), content_type='application/json')
    else:
        response = jsonify({'error': 'Could not process point cloud'})

    print(f"Total time: {datetime.now() - start_time}")
    return response


if __name__ == '__main__':
    app.run(debug=True)
