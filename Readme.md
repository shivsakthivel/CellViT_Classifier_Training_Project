### This file demonstrates the workflow for generating the tiles and labels from the ground truth annotations to train a CellViT++ custom classifier

### Step 1: Prepare the tiles and labels

The required files are:
1. Ground truth annotation filepath (e.g "./annotations/batch1/{uid}_good20.geojson") where the uid is the identifier associated with the slide
2. The associated image filepath for the WSI
3. The detection and segmentation contours in their appropriate locations in "./cellvit_pp_outputs"
4. An output directory for the tiles
5. An output directory for the labels (in .npy)

Note: To ensure that the numpy masks are compatible, please run the following script within the conda environment for the original CellViT++ library (it should be labelled cellvit_env) -

` (cellvit_env) user@host: ~/classifier_training_project$ python3 process_wsi_for_classifier.py --batch "batch1" --slide_path "/your/slide/filepath/here" --annotation_file "./annotations/batch1/your_annotation_file_here.geojson" --tiles_dir "./tiles/batch1" --labels_dir "./labels/batch1" `

The tiles will be named as `{uid}_tile_{index}.png` and are 256x256 RGB images and the corresponding labels will be `{uid}_tile_{index}.npy`

### Step 2: Transfer the generated image tiles and labels to the CellViT++ directory

Within the cloned CellViT++ repository, the generated tiles can be moved to the directory:

1. "./custom_dataset/train/images" for the tiles
2. "./custom_dataset/train/labels" for the labels

TO DO: Create a branch of the repository with the defined changes for the CellViT++ repository, after which the training steps defined in the repository can be followed:

`(cellvit_env) user@host: python3 ./cellvit/train_cell_classifier_head.py --config ./custom_dataset/train_configs/ViT256/fold_0.yaml`

Then, the detect_cells.py script can also be run on the selected run from the logs_local directory:

`(cellvit_env) user@host: python3 -m cellvit.detect_cells --model ./checkpoints/CellViT-256-x40-AMP.pth --outdir ./test_results --classifier_path ./logs_local/run_timestamp_cellvit++/checkpoints/model_best.pth process_wsi --wsi_path ./example_images/filepath_to_your_wsi`