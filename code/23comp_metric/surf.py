# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""The SURF dataset."""

import glob
import os
import random

from absl import logging
import numpy as np
import sklearn
from sklearn import model_selection
import torch
from torch.utils import data
from torchvision import io
import torchvision.transforms as T


# class SURFDataset(data.Dataset):
#   """The SURF dataset."""

#   def __init__(self, path, width=224, height=224, split='train'):
#     super().__init__()
#     self.path = path
#     self.examples = []

#     labels_csv = np.loadtxt(
#         os.path.join(path, 'Age_Gender.txt'),
#         dtype={
#             'names': ('id', 'age', 'gender'),
#             'formats': ('S11', 'i4', 'S1'),
#         },
#         delimiter=' ',
#     )
#     users = np.array([u.decode('utf-8') for u in np.unique(labels_csv['id'])])

#     # make sure there are no repeated entries
#     assert len(users) == len(labels_csv['id'])

#     # transform labels into a more convenient dictionary for lookup
#     ages = np.array([a for _, a, _ in labels_csv])
#     ages_groups = np.linspace(np.min(ages), np.max(ages) + 1, 11)
#     ages_digitized = (
#         np.digitize(ages, ages_groups) - 1
#     )  # substract 1 to make it zero-indexed
#     # overwrite the age with the age group
#     new_labels = []
#     for i, ag in enumerate(ages_digitized):
#       # age-group, ID, age, gender
#       new_labels.append(
#           (ag, labels_csv[i][0], labels_csv[i][1], labels_csv[i][2])
#       )

#     labels = {
#         user.decode('utf-8'): (int(age_group), int(age), gender.decode('utf-8'))
#         for age_group, user, age, gender in new_labels
#     }
#     path = os.path.join(path, 'WBackground')

#     for f in sorted(glob.glob(os.path.join(path, '*/*'))):
#       # get the last directory name
#       person_id = f.split('/')[-1]
#       for img_path in sorted(
#           glob.glob(
#               os.path.join(f, 'real.rssdk', 'color', '*.jpg')
#           )
#       ):
#         image = io.read_image(img_path)
#         # resize image to (3, width, height)
#         image = T.Resize((width, height))(image)
#         self.examples.append({
#             'image': image,
#             'raw_image': image,
#             'id': np.where(person_id == users)[0][0],
#             'gender': labels[person_id][2],
#             'age_group': labels[person_id][0],
#             'age': labels[person_id][1],
#         })

#     # Now, shuffle the examples, with the same random seed each time,
#     # to ensure the same train / validation and test splits each time.
#     random.Random(43).shuffle(self.examples)

#     num_examples = len(self.examples)
#     if split == 'train':
#       self.examples = self.examples[: int(0.8 * num_examples)]
#     elif split == 'val':
#       self.examples = self.examples[
#           int(0.8 * num_examples) : int(0.9 * num_examples)
#       ]
#     elif split == 'test':
#       self.examples = self.examples[int(0.9 * num_examples) :]
#     else:
#       raise ValueError('Unknown split {}'.format(split))

#   def __len__(self):
#     return len(self.examples)

#   def __getitem__(self, idx):
#     example = self.examples[idx]
#     image = example['image']
#     image = image.to(torch.float32)
#     example['image'] = image
#     return example

class SURFDataset(data.Dataset):
    """The SURF dataset."""

    def __init__(self, path, width=224, height=224, split='train'):
        super().__init__()
        self.path = path
        self.width = width
        self.height = height
        self.examples = []

        # Separate concerns
        self.labels = self._load_labels()
        self._load_examples()

        # Shuffle with fixed seed for reproducible splits
        random.Random(43).shuffle(self.examples)

        # Split the dataset
        self._apply_split(split)

    #TODO: change page and .txt names to include all three
    def _load_labels(self, tuples=()): 
        """Load and process labels from Age_Gender.txt"""
        labels_csv = np.loadtxt(
            os.path.join(self.path, 'Age_Gender.txt'),
            dtype={
                'names': ('id', 'age', 'gender'),
                'formats': ('S11', 'i4', 'S1'),
            },
            delimiter=' ',
        )

        # Ensure no duplicate IDs
        users = np.array([u.decode('utf-8') for u in np.unique(labels_csv['id'])])
        assert len(users) == len(labels_csv['id']), "Duplicate IDs found in Age_Gender.txt"

        # Convert ages to age groups

        ages = np.array([a for _, a, _ in labels_csv])
        ages_groups = np.linspace(np.min(ages), np.max(ages) + 1, 11)
        ages_digitized = np.digitize(ages, ages_groups) - 1  # zero-indexed

        # Build final label dictionary
        labels = {}
        for i, age_group in enumerate(ages_digitized):
            user_id = labels_csv[i][0].decode('utf-8')
            age = int(labels_csv[i][1])
            gender = labels_csv[i][2].decode('utf-8')
            
            labels[user_id] = (age_group, age, gender)

        return labels

    def _load_examples(self):
        """Load all image paths and associated metadata"""
        data_path = os.path.join(self.path, 'WBackground')

        for f in sorted(glob.glob(os.path.join(data_path, '*/*'))):
            person_id = f.split('/')[-1]

            # Skip if person not in labels
            if person_id not in self.labels:
                continue

            for img_path in sorted(glob.glob(os.path.join(f, 'real.rssdk', 'color', '*.jpg'))):
                image = io.read_image(img_path)
                image = T.Resize((self.width, self.height))(image)

                self.examples.append({
                    'image': image,
                    'raw_image': image.clone(),   # keep a copy of original resized image
                    'id': person_id,              # keep as string for readability
                    'gender': self.labels[person_id][2],
                    'age_group': self.labels[person_id][0],
                    'age': self.labels[person_id][1],
                })

    def _apply_split(self, split):
        """Apply train/val/test split"""
        num_examples = len(self.examples)
        
        if split == 'train':
            self.examples = self.examples[: int(0.8 * num_examples)]
        elif split == 'val':
            self.examples = self.examples[
                int(0.8 * num_examples): int(0.9 * num_examples)
            ]
        elif split == 'test':
            self.examples = self.examples[int(0.9 * num_examples):]
        else:
            raise ValueError(f'Unknown split: {split}')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx].copy()  # avoid modifying original
        example['image'] = example['image'].to(torch.float32)
        return example

def _get_age_group_counts(ds, name, quiet=False):
  """Get the age group counts."""
  age_group_counts = {}
  age_group_ranges = {}
  for sample in ds:
    age_group = sample['age_group']
    age = sample['age']

    if age_group in age_group_counts:
      age_group_counts[age_group][0] += 1
      age_group_ranges[age_group][0] = min(age_group_ranges[age_group][0], age)
      age_group_ranges[age_group][1] = max(age_group_ranges[age_group][1], age)
    else:
      age_group_counts[age_group] = [1, age]
      age_group_ranges[age_group] = [age, age]

  sorted_counts = sorted(age_group_counts.items(), key=lambda x: x[1][1])
  for age_group, (count, age) in sorted_counts:
    if not quiet:
      logging.info(
          '[Dataset %s] Age group %s : %d',
          name,
          age_group,
          round(100 * count / len(ds), 2),
      )
  return sorted_counts

def get_dataset(batch_size=64, quiet=False, dataset_path=''):
  """Get the SURF dataset."""
  train_ds = SURFDataset(dataset_path, width=32, height=32, split='train')
  val_ds = SURFDataset(dataset_path, width=32, height=32, split='val')
  test_ds = SURFDataset(dataset_path, width=32, height=32, split='test')

  # Get all person id from the training dataset.
  ids = np.array([int(t['id']) for t in train_ds])
  # Create a split the respect the user id
  # Reshape the arrays to have the same number of rows
  x = np.arange(len(ids)).reshape((-1, 1))

  # Split into retain and forget sets, ensuring separation of person IDs.
  sklearn.utils.check_random_state(0)
  lpgo = model_selection.LeavePGroupsOut(n_groups=15)
  retain_index, forget_index = next(lpgo.split(x, None, ids))
  retain_ds = data.Subset(train_ds, retain_index)
  forget_ds = data.Subset(train_ds, forget_index)

  # Ensure that the retain and forget sets don't have any common IDs.
  retain_ids = np.unique([int(ret['id']) for ret in retain_ds])
  forget_ids = np.unique([int(forg['id']) for forg in forget_ds])
  assert not set(retain_ids).intersection(set(forget_ids))

  if not quiet:
    logging.info('Train set size %d', len(train_ds))
    logging.info('Val set size %d', len(val_ds))
    logging.info('Test set size: %d', len(test_ds))
    logging.info('Retain set size: %d', len(retain_ds))
    logging.info('Forget set size: %d', len(forget_ds))

  train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  val_loader = data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)
  test_loader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)
  retain_loader = data.DataLoader(
      retain_ds, batch_size=batch_size, shuffle=True
  )
  forget_loader = data.DataLoader(
      forget_ds, batch_size=batch_size, shuffle=True
  )
  forget_loader_no_shuffle = data.DataLoader(
      forget_ds, batch_size=batch_size, shuffle=False
  )

  # Get the class weights:
  sorted_counts = _get_age_group_counts(train_ds, 'train', quiet=quiet)
  _get_age_group_counts(retain_ds, 'retain', quiet=quiet)
  _get_age_group_counts(forget_ds, 'forget', quiet=quiet)
  _get_age_group_counts(val_ds, 'valid', quiet=quiet)
  _get_age_group_counts(test_ds, 'test', quiet=quiet)
  class_weights = [
      1.0 / item[1][0] if item[1][0] != 0 else 1.0 for item in sorted_counts
  ]
  class_weights_tensor = torch.FloatTensor(class_weights)
  return (
      train_loader,
      val_loader,
      test_loader,
      retain_loader,
      forget_loader,
      forget_loader_no_shuffle,
      class_weights_tensor,
  )

# def get_dataset(batch_size=64, quiet=False, dataset_path=''):
#     """Load CASIA-SURF CeFA dataset."""
#     import os
#     import numpy as np
#     from sklearn.model_selection import LeavePGroupsOut

#     if not os.path.exists(dataset_path):
#         raise ValueError(f"Dataset path not found: {dataset_path}")

#     print(f"Loading CeFA dataset from: {dataset_path}")

#     image_paths = []
#     labels = []      # age group
#     subject_ids = []

#     # CeFA ethnicities
#     ethnicities = ['AF', 'CA', 'EA']

#     print("Scanning CeFA folders...")
#     for eth in ethnicities:
#         eth_dir = os.path.join(dataset_path, eth)
#         if not os.path.exists(eth_dir):
#             continue
            
#         for root, _, files in os.walk(eth_dir):
#             for file in files:
#                 if file.lower().endswith(('.jpg', '.png', '.jpeg')):
#                     full_path = os.path.join(root, file)
#                     image_paths.append(full_path)

#                     # Extract age group from Age_Gender.txt or filename
#                     # For now, use a simple placeholder - we'll improve this later
#                     try:
#                         # You can improve this by parsing Age_Gender.txt
#                         age_group = int(file.split('_')[2]) if len(file.split('_')) > 2 else 0
#                     except:
#                         age_group = 0
#                     labels.append(age_group)

#                     # Extract subject ID
#                     try:
#                         sid = int(file.split('_')[1])
#                     except:
#                         sid = hash(full_path) % 10000
#                     subject_ids.append(sid)

#     image_paths = np.array(image_paths)
#     labels = np.array(labels)
#     subject_ids = np.array(subject_ids)

#     print(f"Found {len(image_paths)} images from {len(np.unique(subject_ids))} subjects.")

#     if len(image_paths) == 0:
#         raise ValueError("No images found in CeFA folders!")

#     # Use Leave-P-Groups-Out for unlearning split
#     lpgo = LeavePGroupsOut(n_groups=1)

#     # Get first split (you can modify this later)
#     retain_index, forget_index = next(lpgo.split(image_paths, labels, subject_ids))

#     print(f"Retain set: {len(retain_index)} images | Forget set: {len(forget_index)} images")

#     # TODO: Return proper tf.data.Dataset or whatever the rest of the code expects
#     # For now, return basic info so it doesn't crash
#     return image_paths, labels, subject_ids, retain_index, forget_index

def compute_accuracy_surf(
    data_names_list,
    data_loader_list,
    net,
    model_name,
    print_per_class_=True,
    print_=True,
):
  """Compute the accuracy."""
  net.eval()
  accs = {}
  pc_accs = {}
  list_of_classes = list(range(10))
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  with torch.no_grad():
    for name, loader in zip(data_names_list, data_loader_list):
      correct = 0
      total = 0
      correct_pc = [0 for _ in list_of_classes]
      total_pc = [0 for _ in list_of_classes]
      for sample in loader:
        inputs = sample['image']
        targets = sample['age_group']
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        for c in list_of_classes:
          num_class_c = (targets == c).sum().item()
          correct_class_c = (
              ((predicted == targets) * (targets == c)).float().sum().item()
          )
          total_pc[c] += num_class_c
          correct_pc[c] += correct_class_c

      accs[name] = 100.0 * correct / total
      pc_accs[name] = [
          100.0 * c / t if t > 0 else -1.0 for c, t in zip(correct_pc, total_pc)
      ]

  print_str = '%s accuracy: ' % model_name
  for name in data_names_list:
    print_str += '%s: %.2f, ' % (name, accs[name])
  if print_:
    logging.info(print_str)
  if print_per_class_:
    for name in data_names_list:
      print_str = '%s accuracy per class: ' % name
      for _, pc_acc in enumerate(pc_accs[name]):
        print_str += ' %.2f, ' % pc_acc
      logging.info(print_str)
  return accs
