import os
import h5py
import numpy as np
import argparse

def save_to_hdf5(data_dict_list, file_path, chunk_size=None):
    def write_data_to_group(group, data):
        for key, value in data.items():
            if isinstance(value, (int, float)):
                group.attrs[key] = value
            elif isinstance(value, str):
                string_dt = h5py.string_dtype(encoding='utf-8')
                group.create_dataset(key, data=value, dtype=string_dt)
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
            elif isinstance(value, dict):
                subgroup = group.create_group(key)
                write_data_to_group(subgroup, value) 
            else:
                raise Exception(f"Invalid type for {key}: {type(value)}. Expected int, float, str, or np.ndarray.")
            # flattens nested dictionaries, by appending top level key as a prefix

    chunk = chunk_size or 1
    mode = 'w'
    try:
        with h5py.File(file_path, mode) as f:
            existing_indices = [int(k.split('_')[1]) for k in f.keys() if k.startswith("record_")]
            current_max_index = max(existing_indices) + 1 if existing_indices else 0
            total_records = len(data_dict_list)
            print(f"Saving {total_records} records to file starting at index {current_max_index}")

            for i in range(0, total_records, chunk):
                for idx in range(i, min(i + chunk, total_records)):
                    record_index = current_max_index + idx
                    record_group_name = f"record_{record_index}"

                    if record_group_name in f:
                        print(f"Skipping existing group: {record_group_name}")
                        continue

                    record_dict = data_dict_list[idx]
                    record_group = f.create_group(record_group_name)
                    write_data_to_group(record_group, record_dict)

    except Exception as err:
        raise Exception(f"Error writing to HDF5 file {file_path}: {err}")



def forward_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return `C = (A + B) ` as `float32`."""
    return (a + b).astype(np.float32)

def generate_record(grid_length: int,
                    seed: int):
    
    """Create ONE record with integer `x`.

    Algorithm
    ---------
    1. Generate random values for A and B.
    2. Calculate x such that A * B = C.
    3. Save (A, B) and C.
    """
    rng = np.random.default_rng(seed)

    shape = (grid_length, grid_length)

    C = rng.integers(1, 100)
    A = rng.integers(1, 100)
    B = C - A
    
    A_image = np.full(shape, A, dtype=np.float32)
    B_image = np.full(shape, B, dtype=np.float32)
    C_image = forward_map(A_image, B_image)

    return {
        "image": {
            "a": A_image,
            "b": B_image,
            "c": C_image,
        },
        "meta": {
            "random_seed": seed,
            "grid_length": grid_length
        },
    }

def main():
    parser = argparse.ArgumentParser(description="Generate simple HDF5 dataset")

    parser.add_argument(
        "-o", "--output-path",
        dest="output_path",
        help="Output path to save dataset file to",
        required=True
    )
    
    args = parser.parse_args()

    if not os.path.isdir(args.output_path):
        raise NotADirectoryError(f"--output-path {args.output_path} is not a existing directory")

    min_seed=1
    max_seed=1000
    total_records=max_seed-min_seed+1
    grid_length=3

    output_file = f"simple_{grid_length}x{grid_length}_{min_seed}-{max_seed}.hdf5"
    output_path = os.path.join(args.output_path, output_file)

    result_records = []
    for seed in range(min_seed, max_seed+1):
        record = generate_record(grid_length, seed)
        result_records.append(record)

    save_to_hdf5(result_records, output_path)

    print(f"Saved dataset with {total_records} samples to '{output_path}'")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
