import pandas as pd
import json

annotations = pd.read_csv("/home/fpshell/development/datasets/car_objects/archive/data/train_solution_bounding_boxes (1).csv")

images = []
annotations_list = []
categories = [{"id": 1, "name": "car"}]

for index, row in annotations.iterrows():
    image_id = index
    images.append({
        "id": image_id,
        "file_name": row['image'],
        "width": 676,  # Example width, replace with actual
        "height": 380  # Example height, replace with actual
    })
    annotations_list.append({
        "id": index,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [row['xmin'], row['ymin'], row['xmax'] - row['xmin'], row['ymax'] - row['ymin']],
        "area": (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin']),
        "iscrowd": 0
    })

coco_format = {
    "images": images,
    "annotations": annotations_list,
    "categories": categories
}

with open('annotations.json', 'w') as f:
    json.dump(coco_format, f)

