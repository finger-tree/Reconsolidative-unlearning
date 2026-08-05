

import glob
import os
import random


import numpy as np
import torch
from torch.utils import data
from torchvision import io
import torchvision.transforms as T

import glob
import os

from torchvision import io
import torchvision.transforms as T


def _load_examples():
    """Load all image paths and associated metadata.

    Expects a directory structure of:
        self.path/<any_dir>/<person_id>/<any_dir>/color/<photos>
    """
    rt = []
    path = "data"

    # Pattern: self.path / any_dir / person_id / any_dir / color / *.jpg
    pattern = os.path.join(path, '*', '*', '*', 'color', '*.jpg')

    for img_path in sorted(glob.glob(pattern)):
        # Normalize so splitting on os.sep is reliable across platforms
        parts = os.path.normpath(img_path).split(os.sep)

        # parts[-1] = filename, parts[-2] = 'color', parts[-3] = any_dir,
        # parts[-4] = person_id, parts[-5] = any_dir (top-level)
        person_id = parts[-4]

        # Skip if person not in labels
        # if person_id not in self.labels:
        #     continue

        image = io.read_image(img_path)
        image = T.Resize((32, 32))(image)

        rt.append({
            'image': image,
            'raw_image': image.clone(),   # keep a copy of original resized image
            'id': person_id,              # keep as string for readability
            # 'gender': self.labels[person_id][2],
            # 'age_group': self.labels[person_id][0],
            # 'age': self.labels[person_id][1],
        })

    return rt

if __name__ == '__main__':
    a = _load_examples()
    print("test_examples", a)
    print("test_examples", type(a))


