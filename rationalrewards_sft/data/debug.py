import json
import os

dataset_info_path = os.environ.get("DATASET_INFO_PATH", "dataset_info.json")
json.load(open(dataset_info_path, encoding="utf-8"))