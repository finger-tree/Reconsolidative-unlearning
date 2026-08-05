import os
import numpy as np



if __name__ == '__main__':
    path = "" # current path
    
    # Open and read lines, splitting on ANY whitespace to ignore double spaces
    with open(os.path.join(path, 'Age_Gender.txt'), 'r') as f:
        valid_lines = [line for line in f if len(line.strip().split()) == 3]

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
    print("--- Test Results ---")
    print(f"Array content:\n{labels_csv}")
    print(f"Array Dimension (ndim): {labels_csv.ndim}")
    print(f"Array Shape: {labels_csv.shape}")
    print(f"Data type (dtype): {labels_csv.dtype}")
    
    # 4. Quick example of data access
    print("\n--- Data Access Quick Test ---")
    print(f"First row: {labels_csv[0]}")
    print(f"All ages: {labels_csv['age']}")



