import os


_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

_DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(_PROJECT_ROOT, "data"))


IMG_DIR = os.environ.get("IMG_DIR", os.path.join(_DATA_ROOT, "intersection_train_data"))

