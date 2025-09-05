# Imports
import geojson
import geopandas as gpd
import json
import numpy as np
import pandas as pd
import ast
import os

batch = "batch2" # Change this to the desired batch name

# Annotation Files
annotation_files = sorted(os.listdir(f"annotations/{batch}"))
annotation_files = [f for f in annotation_files if f.endswith('.geojson')]
print(f"Found {len(annotation_files)} annotation files.")

def change_classification(current_class, label):
    if label == 'good':
        return current_class
    
    elif label == 'tum_to_str':
        return 'Connective'
    
    elif label == 'tumepi_to_imm':
        return 'Inflammatory'

# Function to process a single annotation file
def process_one_file(annotation_filename):
    image_code = annotation_filename.split("_")[0]
    example_annotation_file = f"/home/sakthi01/Downloads/classifier_training_project/annotations/{batch}/{annotation_filename}"
    with open(example_annotation_file, "r") as f:
        annotation_data = geojson.load(f)
    
    example_detection_file = f"/home/sakthi01/Downloads/classifier_training_project/cellvit_pp_outputs/{batch}/{image_code}_cell_detection.geojson"
    example_seg_file = f"/home/sakthi01/Downloads/classifier_training_project/cellvit_pp_outputs/{batch}/{image_code}_cells.geojson"

    with open(example_detection_file, "r") as f:
        detection_data = geojson.load(f)

    with open(example_seg_file, "r") as f:
        seg_data = geojson.load(f)
    
    rois = {}
    columns = ['start_x', 'end_x', 'start_y', 'end_y', 'label']
    for i in range(len(annotation_data['features'])):
        current_tile = annotation_data['features'][i]
        start_coords = np.min(np.array(current_tile['geometry']['coordinates'][0]), axis = 0)
        end_coords = np.max(np.array(current_tile['geometry']['coordinates'][0]), axis = 0)
        label = current_tile['properties']['classification']['name']
        rois[i] = [start_coords[0], end_coords[0], start_coords[1], end_coords[1], label]
    rois_df = pd.DataFrame.from_dict(rois, orient='index', columns=columns)

    annotated_nuclei = {}
    columns = ['detection_x', 'detection_y', 'contour', 'Classification']
    index = 0
    for i in range(len(detection_data)):
        current_points = detection_data[i]
        current_class = current_points['properties']['classification']['name']
        for j in range(len(current_points['geometry']['coordinates'])):
            current_point = current_points['geometry']['coordinates'][j]
            current_x = current_point[0]
            current_y = current_point[1]
            for k in range(len(rois_df)):
                current_roi = rois_df.iloc[k]
                if (current_x >= current_roi['start_x']) and (current_x <= current_roi['end_x']) and (current_y >= current_roi['start_y']) and (current_y <= current_roi['end_y']):
                    new_class = change_classification(current_class, current_roi['label'])
                    annotated_nuclei[index] = [current_x, current_y, seg_data[i]['geometry']['coordinates'][j], new_class]
                    index += 1
                    break

    annotated_nuclei_df = pd.DataFrame.from_dict(annotated_nuclei, orient='index', columns=columns)

    # Export the csv with the processed annotations
    annotated_nuclei_df.to_csv(f"processed_annotations/{batch}/{image_code}_processed_annotations.csv", index=False)

# Process all annotation files
for annotation_file in annotation_files:
    print(f"Processing {annotation_file}...")
    process_one_file(annotation_file)
print("Processing complete.")

