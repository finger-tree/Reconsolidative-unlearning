import os
import numpy as np
from absl import logging

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

if __name__ == '__main__':
    path = "data" # current path
    
    # Open and read lines, splitting on ANY whitespace to ignore double spaces
    with open(os.path.join(path, 'Age_Gender.txt'), 'r') as f:
        valid_lines = [line for line in f if len(line.strip().split()) == 3]
        print(valid_lines)
        print("valid_lines", type(valid_lines))

    # Feed the cleaned lines directly into loadtxt
    labels_csv = np.loadtxt(
        valid_lines,
        dtype={
            'names': ('id', 'age', 'gender'),
            'formats': ('S11', 'i4', 'S1'),
        },
        delimiter=' ',
    )

    # 3. Print the results to verify structure
    # print("--- Test Results ---")
    # print(f"Array content:\n{labels_csv}")
    # print(f"Array Dimension (ndim): {labels_csv.ndim}")
    # print(f"Array Shape: {labels_csv.shape}")
    # print(f"Data type (dtype): {labels_csv.dtype}")
    
    # # 4. Quick example of data access
    # print("\n--- Data Access Quick Test ---")
    # print(f"First row: {labels_csv[0]}")
    # print(f"All ages: {labels_csv['age']}")

    # print("what is this np thing: ", labels_csv)
    # print("one:", labels_csv[0][0], type(labels_csv[0][0]))
    
# to replace load_labels function from surf.py