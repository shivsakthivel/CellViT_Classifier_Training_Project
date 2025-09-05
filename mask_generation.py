import geojson
import numpy as np
import pandas as pd
import ast
import os
import skimage

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
    for i in range(len(tiles_df)):
        current_tile = tiles_df.iloc[i]
        if (detection_x >= current_tile['start_x']) and (detection_x <= current_tile['end_x']) and (detection_y >= current_tile['start_y']) and (detection_y <= current_tile['end_y']):
            return current_tile['tile_index']
    return np.nan

def main():
    batch = "batch1"
    processed_dir = f"processed_annotations/{batch}"
    annotation_dir = f"annotations/{batch}"
    label_dir = f"labels/{batch}"
    os.makedirs(label_dir, exist_ok=True)

    processed_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('_processed_annotations.csv')])
    annotation_files = sorted([f for f in os.listdir(annotation_dir) if f.endswith('.geojson')])

    for processed_file in processed_files:
        image_code = processed_file[:6]
        # Find matching annotation file
        matching_ann = [f for f in annotation_files if f.startswith(image_code)]
        if not matching_ann:
            print(f"No annotation file found for {processed_file}")
            continue
        annotation_file = os.path.join(annotation_dir, matching_ann[0])
        processed_path = os.path.join(processed_dir, processed_file)

        print(f"Processing {processed_file} with {matching_ann[0]}")

        processed_annotations = pd.read_csv(processed_path)
        with open(annotation_file, "r") as f:
            annotation_data = geojson.load(f)
        tiles_df = extract_tiles(annotation_data)
        processed_annotations['tile_index'] = processed_annotations.apply(lambda row: set_tile_index(row, tiles_df), axis=1)
        processed_annotations = processed_annotations.dropna(subset=['tile_index'])
        processed_annotations['tile_index'] = processed_annotations['tile_index'].astype(int)

        for i in range(len(tiles_df)):
            current_tile = tiles_df.iloc[i]
            tile_index = current_tile['tile_index']
            tile_annotations = processed_annotations[processed_annotations['tile_index'] == tile_index]
            if len(tile_annotations) == 0:
                continue
            instance_map = np.zeros((256, 256), dtype=np.uint16)
            type_map = np.zeros((256, 256), dtype=np.uint8)
            for j in range(len(tile_annotations)):
                annotation = tile_annotations.iloc[j]
                polygon_points = np.array(ast.literal_eval(annotation['contour'])[0])
                adjusted_points = polygon_points - np.array([current_tile['start_x'], current_tile['start_y']])
                poly_instance_mask = create_polygon_mask_instance((256, 256), np.rint(adjusted_points), j + 1)
                poly_class_mask, class_label = create_polygon_mask_class((256, 256), np.rint(adjusted_points), annotation['Classification'])
                instance_map = np.maximum(instance_map, poly_instance_mask)
                type_map = np.maximum(type_map, poly_class_mask)
            instance_map = instance_map.T
            type_map = type_map.T
            formatted_annotations = {'inst_map': instance_map, 'type_map': type_map}
            out_path = os.path.join(label_dir, f"{image_code}_tile_{tile_index:04}.npy")
            with open(out_path, "wb") as f:
                np.save(f, formatted_annotations)

if __name__ == "__main__":
    main()