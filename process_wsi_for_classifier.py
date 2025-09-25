import os
import numpy as np
import skimage.io
import openslide
import geojson
import pandas as pd
import ast
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser(description='Process WSI for classifier training')
parser.add_argument('--batch', type=str, required=True, help='Batch name (e.g., batch1, batch2)')
parser.add_argument('--slide_path', type=str, required=True, help='Path to the WSI file')
parser.add_argument('--annotation_file', type=str, required=True, help='Path to the annotation geojson file')
parser.add_argument('--tiles_dir', type=str, required=True, help='Directory to save the extracted tiles')
parser.add_argument('--labels_dir', type=str, required=True, help='Directory to save the generated labels')
args = parser.parse_args()

def change_classification(current_class, label):
    if label == 'good':
        return current_class
    elif label == 'tum_to_str':
        return 'Connective'
    elif label == 'tumepi_to_imm':
        return 'Inflammatory'

def create_polygon_mask_instance(shape, points, instance):
    return skimage.draw.polygon2mask(shape, points) * instance

def create_polygon_mask_class(shape, points, classification):
    label = 0
    if classification == 'Neoplastic':
        label = 1
    if classification == 'Connective':
        label = 2
    if classification == 'Inflammatory':
        label = 3
    if classification == 'Dead':
        label = 4
    if classification == 'Epithelial':
        label = 5
    return skimage.draw.polygon2mask(shape, points) * label, label

def extract_tiles(annotation_data):
    tiles = {}
    columns = ['tile_index', 'start_x', 'end_x', 'start_y', 'end_y']
    tile_counter = 0
    for i in range(len(annotation_data['features'])):
        current_tile = annotation_data['features'][i]
        start_coords = np.min(np.array(current_tile['geometry']['coordinates'][0]), axis=0)
        end_coords = np.max(np.array(current_tile['geometry']['coordinates'][0]), axis=0)
        for j in range(start_coords[0], end_coords[0], 256):
            for k in range(start_coords[1], end_coords[1], 256):
                tile_counter += 1
                tiles[tile_counter] = [tile_counter, j, j+256, k, k+256]
    tiles_df = pd.DataFrame.from_dict(tiles, orient='index', columns=columns)
    return tiles_df

def set_tile_index(row, tiles_df):
    detection_x = row['detection_x']
    detection_y = row['detection_y']
    # Vectorized search for tile index
    mask = (
        (tiles_df['start_x'] <= detection_x) & (tiles_df['end_x'] >= detection_x) &
        (tiles_df['start_y'] <= detection_y) & (tiles_df['end_y'] >= detection_y)
    )
    matches = tiles_df[mask]
    if not matches.empty:
        return matches.iloc[0]['tile_index']
    return np.nan

def process_one_file(annotation_file, batch):
    image_code = os.path.basename(annotation_file).split("_")[0]
    try:
        with open(annotation_file, "r") as f:
            annotation_data = geojson.load(f)
    except Exception as e:
        print(f"Error reading annotation file: {e}")
        return None

    detection_file = f"cellvit_pp_outputs/{batch}/{image_code}_cell_detection.geojson"
    seg_file = f"cellvit_pp_outputs/{batch}/{image_code}_cells.geojson"

    try:
        with open(detection_file, "r") as f:
            detection_data = geojson.load(f)
        with open(seg_file, "r") as f:
            seg_data = geojson.load(f)
    except Exception as e:
        print(f"Error reading detection/segmentation files: {e}")
        return None

    rois = {}
    columns = ['start_x', 'end_x', 'start_y', 'end_y', 'label']
    for i, feature in enumerate(annotation_data['features']):
        start_coords = np.min(np.array(feature['geometry']['coordinates'][0]), axis=0)
        end_coords = np.max(np.array(feature['geometry']['coordinates'][0]), axis=0)
        label = feature['properties']['classification']['name']
        rois[i] = [start_coords[0], end_coords[0], start_coords[1], end_coords[1], label]
    rois_df = pd.DataFrame.from_dict(rois, orient='index', columns=columns)

    annotated_nuclei = {}
    columns = ['detection_x', 'detection_y', 'contour', 'Classification']
    index = 0
    for i, current_points in tqdm(enumerate(detection_data), total=len(detection_data), desc="Processing nuclei annotations"):
        current_class = current_points['properties']['classification']['name']
        for j, current_point in enumerate(current_points['geometry']['coordinates']):
            current_x, current_y = current_point
            mask = (
                (rois_df['start_x'] <= current_x) & (rois_df['end_x'] >= current_x) &
                (rois_df['start_y'] <= current_y) & (rois_df['end_y'] >= current_y)
            )
            matches = rois_df[mask]
            if not matches.empty:
                current_roi = matches.iloc[0]
                new_class = change_classification(current_class, current_roi['label'])
                annotated_nuclei[index] = [current_x, current_y, seg_data[i]['geometry']['coordinates'][j], new_class]
                index += 1

    annotated_nuclei_df = pd.DataFrame.from_dict(annotated_nuclei, orient='index', columns=columns)
    annotated_nuclei_df.to_csv(f"processed_annotations/{batch}/{image_code}_processed_annotations.csv", index=False)
    return annotated_nuclei_df

def save_tile(slide, row, image_code, tiles_dir):
    try:
        tile = slide.read_region((row['start_x'], row['start_y']), 0, (256, 256)).convert("RGB")
        tile_path = os.path.join(tiles_dir, f"{image_code}_tile_{str(row['tile_index']).zfill(4)}.png")
        tile.save(tile_path)
    except Exception as e:
        print(f"Error saving tile {row['tile_index']}: {e}")

def save_label(row, processed_annotations, image_code, labels_dir):
    tile_index = row['tile_index']
    tile_annotations = processed_annotations[processed_annotations['tile_index'] == tile_index]
    if len(tile_annotations) == 0:
        return
    instance_map = np.zeros((256, 256), dtype=np.uint16)
    type_map = np.zeros((256, 256), dtype=np.uint8)
    for j, annotation in tile_annotations.iterrows():
        polygon_points = np.array(annotation['contour'][0])
        adjusted_points = polygon_points - np.array([row['start_x'], row['start_y']])
        poly_instance_mask = create_polygon_mask_instance((256, 256), np.rint(adjusted_points), j + 1)
        poly_class_mask, class_label = create_polygon_mask_class((256, 256), np.rint(adjusted_points), annotation['Classification'])
        instance_map = np.maximum(instance_map, poly_instance_mask)
        type_map = np.maximum(type_map, poly_class_mask)
    instance_map = instance_map.T
    type_map = type_map.T
    formatted_annotations = {'inst_map': instance_map, 'type_map': type_map}
    out_path = os.path.join(labels_dir, f"{image_code}_tile_{tile_index:04}.npy")
    try:
        with open(out_path, "wb") as f:
            np.save(f, formatted_annotations)
    except Exception as e:
        print(f"Error saving label {tile_index}: {e}")

def main():
    annotation_file = args.annotation_file
    batch = args.batch
    slide_path = args.slide_path
    tiles_dir = args.tiles_dir
    labels_dir = args.labels_dir
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    processed_annotations = process_one_file(annotation_file, batch)
    if processed_annotations is None:
        print("Failed to process annotation file.")
        return

    with open(annotation_file, "r") as f:
        annotation_data = geojson.load(f)
    tiles_df = extract_tiles(annotation_data)
    processed_annotations['tile_index'] = processed_annotations.apply(
        lambda row: set_tile_index(row, tiles_df), axis=1
    )
    processed_annotations = processed_annotations.dropna(subset=['tile_index'])
    processed_annotations['tile_index'] = processed_annotations['tile_index'].astype(int)
    image_code = os.path.basename(annotation_file).split("_")[0]

    # Open slide once
    try:
        slide = openslide.OpenSlide(slide_path)
    except Exception as e:
        print(f"Error opening slide: {e}")
        return

    # Parallel tile extraction
    with ThreadPoolExecutor() as executor:
        list(tqdm(
            executor.map(lambda row: save_tile(slide, row, image_code, tiles_dir), [row for _, row in tiles_df.iterrows()]),
            total=len(tiles_df),
            desc="Extracting tiles"
        ))

    # Parallel label generation
    with ThreadPoolExecutor() as executor:
        list(tqdm(
            executor.map(lambda row: save_label(row, processed_annotations, image_code, labels_dir), [row for _, row in tiles_df.iterrows()]),
            total=len(tiles_df),
            desc="Generating labels"
        ))

if __name__ == "__main__":
    main()