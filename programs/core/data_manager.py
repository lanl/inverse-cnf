from programs.utils.logger_setup import get_logger
from programs.utils.common import (PT_TENSOR,
                                    namedtuple, 
                                    extract_item,
                                    as_list,
                                    convert_type,
                                    random, 
                                    np, pt, ast,
                                    os_path, 
                                    create_folder,
                                    create_file_path,
                                    save_to_json,
                                    dataframe_records_converter, 
                                    read_from_hdf5,
                                    match_shape)

pt_data=pt.utils.data
Dataset=pt_data.Dataset
Subset=pt_data.Subset
DataLoader=pt_data.DataLoader
random_split=pt_data.random_split


class MinMaxTuple(namedtuple("MinMax", ["min", "max"])):
    __slots__ = ()
    def __new__(cls, min, max):
        def to_float_or_none(x):
            return convert_type(x, float) if x is not None else None
        return super().__new__(cls, to_float_or_none(min), to_float_or_none(max))


class PreprocessDataset(Dataset):
    def __init__(self, 
                data_file, 
                model_input_keys: list[str], 
                model_target_keys: str, 
                solver_input_keys: list[str], 
                unique_id_key: str, 
                subset_split: list[float], 
                transform_method:str|None=None, 
                batch_size:int=None,
                random_seed:int=None,
                shuffle_on:bool=False, 
                minmax_range:tuple[float, float]=(0.0,1.0),
                flatten_nested_keys=False):
        
        self.in_file = data_file
        hdf5_data = read_from_hdf5(data_file, flatten=flatten_nested_keys)
        self.dataset = dataframe_records_converter(hdf5_data)
        self.dataset['original_index'] = self.dataset.index
        self.columns = self.dataset.columns

        self.batch_shuffle = shuffle_on
        self.batch_size = batch_size

        self.subset_names = ['training', 'validation', 'testing']
        self.subset_split = subset_split
        self.subset_sizes = [convert_type(ratio * len(self.dataset), int) for ratio in self.subset_split]

        if isinstance(unique_id_key, str) and unique_id_key in self.columns:
            self.unique_id_key = unique_id_key
        else:
            self.unique_id_key = 'original_index'
        self.unique_id_name = self.unique_id_key

        self.transform_method = transform_method
        self.minmax_range = minmax_range

        self.random_seed = random_seed
        self.generator = pt.Generator().manual_seed(self.random_seed) if isinstance(self.random_seed, int) else None

        self.group_data_keys = {
            "input": as_list(model_input_keys),
            "target": as_list(model_target_keys)
        }

        if solver_input_keys is not None:
            self.group_data_keys["solver"] = as_list(solver_input_keys)
            
        all_required = sum(self.group_data_keys.values(), [])
        missing = [col for col in all_required if col not in self.columns]
        if missing:
            raise ValueError(f"Missing column(s) in dataset: {missing}")
        
        self.group_names = list(self.group_data_keys.keys())
        self.transform_groups = ["input", "target"] if isinstance(self.transform_method, str) else []

        self._infer_max_length()
        self._initialize_groups()
        self._infer_dimensions()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if pt.is_tensor(idx):
            idx = idx.tolist()
        row = self.dataset.iloc[idx]
        sample = {                              
            **{group: self._process_and_combine(row, group) for group in self.group_names},
            "id": row[self.unique_id_key]
        }
        return sample

    def _get_transform(self):
        if self.transform_method is None:
            return None
        if self.transform_method == "minmax":
            return MinMaxScalerTransform(*self.minmax_range)
        elif self.transform_method == "standard":
            return StandardScalerTransform()
        else:
            raise ValueError(f"Transform type '{self.transform_method}' does not exist")

    def _fit_transform(self, col_name, col_type):
        transform = self._get_transform()
        if transform is None:
            return None
        if col_name not in self.dataset.columns:
            raise KeyError(f"Column '{col_name}' is missing from the input dataset")

        batch = []
        for v in self.dataset[col_name].values:
            if v is None:
                continue

            if col_type == "scalar":
                x = self._process_scalar(v, transform=None)  # (1,)
            elif col_type == "image":
                x = self._process_image(v, transform=None)   # (H,W) or (C,H,W)
                if x.ndim == 2:
                    x = x.unsqueeze(0)                       # (1,H,W)
            else:
                raise ValueError(f"Data type prefix '{col_name}' cannot be transformed.")
            batch.append(x)

        if not batch:
            raise ValueError(f"No valid data found in column '{col_name}' for type '{col_type}'")

        X = pt.stack(batch, dim=0)
        transform.fit(X, is_scalar=(col_type == "scalar"))
        return transform


    def _process_and_combine(self, sample, group):
        data = self.data_groups.get(group, {})
        names = data.get("names", None)
        types = data.get("types", None)

        if names is None or types is None:
            raise KeyError(f"No image or metric tensors to process for data group '{group}'")

        transforms = self.data_transforms.get(group, {})
        images, scalars = self._process_columns(sample, as_list(names), as_list(types), transforms)
        allow_broadcast = ["input", "target"]
        if images is not None and scalars is not None:
            if group in allow_broadcast:
                H, W = self.expected_hw
                scalars_map = scalars.view(-1, 1, 1).expand(-1, H, W)
                return pt.cat([images, scalars_map], dim=0)
            return (images, scalars)
        elif scalars is not None:
            if group in allow_broadcast:
                H, W = self.expected_hw
                scalars_map = scalars.view(-1, 1, 1).expand(-1, H, W)
                return scalars_map
            return scalars
        elif images is not None:
            return images

    def _process_columns(self, row, names: list, types: list, transforms: dict):
        image_data, scalar_data = [], []

        for name, typ in zip(names, types):
            col_key = name  # was f"{typ}_{name}"
            transform = transforms.get(name, None)

            if col_key not in row:
                raise KeyError(f"Missing key '{col_key}'")

            data_item = row.get(col_key)
            if typ == "scalar":
                scalar_data.append(self._process_scalar(data_item, transform))
            elif typ == "image":
                x = self._process_image(data_item, transform)
                if x.dim() == 2:
                    x = x.unsqueeze(0)   # (1,S,S)
                elif x.dim() != 3:
                    raise ValueError(f"Image must be 2D or 3D after processing, got {tuple(x.shape)}")
                image_data.append(x)
            else:
                raise ValueError(f"Invalid type '{typ}'")


        return (
            pt.cat(image_data, dim=0) if image_data else None,
            pt.cat(scalar_data, dim=0) if scalar_data else None,
        )

    def _process_image(self, image, transform=None):
        if isinstance(image, str):
            image = ast.literal_eval(image)

        x = pt.tensor(image, dtype=pt.float32)
        S, S2 = self.expected_hw
        if S != S2:
            raise ValueError(f"expected_hw must be square, got {self.expected_hw}")

        if x.ndim == 2:
            H, W = int(x.shape[-2]), int(x.shape[-1])
            if (H, W) != (S, S):
                raise ValueError(f"Image must be ({S},{S}), got ({H},{W})")
            return transform(x) if transform else x

        if x.ndim == 3:
            H, W = int(x.shape[-2]), int(x.shape[-1])
            if (H, W) != (S, S):
                raise ValueError(f"Image must be (C,{S},{S}), got {tuple(x.shape)}")
            return transform(x) if transform else x

        raise ValueError(f"Image must be 2D or 3D, got {tuple(x.shape)}")
    

    def _process_scalar(self, scalar_value, transform=None):
        scalar_tensor = pt.tensor(scalar_value, dtype=pt.float32).unsqueeze(0)
        return transform(scalar_tensor) if transform else scalar_tensor

    def _derive_type(self, value) -> str:
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value.strip())
            except Exception:
                return "scalar"

        if value is None:
            return "scalar"
        if isinstance(value, (int, float, bool, np.number, np.bool_)):
            return "scalar"

        if isinstance(value, pt.Tensor):
            if value.dim() == 0:
                return "scalar"
            if value.dim() == 1:
                raise ValueError("1D vectors are not supported. Convert to scalar or square 2D image before saving.")
            if value.dim() == 2:
                return "image"
            return "image"

        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            if arr.ndim == 0:
                return "scalar"
            if arr.ndim == 1:
                raise ValueError("1D vectors are not supported. Convert to scalar or square 2D image before saving.")
            if arr.ndim == 2:
                return "image"
            return "image"

        return "scalar"

    def _initialize_groups(self):
        data_transforms, data_groups = {}, {}

        for group, keys in self.group_data_keys.items():
            names = as_list(keys)
            types = []
            transforms = {}
            original_ranges, transformed_ranges = [], []

            for name in names:
                v0 = self.dataset[name].iloc[0]
                typ = self._derive_type(v0)
                types.append(typ)

                if group in self.transform_groups:
                    transform = self._fit_transform(name, typ)
                    transforms[name] = transform
                    original_ranges.append(tuple(transform.original_range))
                    transformed_ranges.append(tuple(transform.scaled_range))
                else:
                    transforms[name] = None
                    original_ranges.append(None)
                    transformed_ranges.append(None)

            data_transforms[group] = transforms
            data_groups[group] = {
                "names": extract_item(list(names)),
                "types": extract_item(list(types)),  # inferred types stored here
                "ranges": {
                    "original": extract_item(original_ranges),
                    "transformed": extract_item(transformed_ranges),
                },
            }

        setattr(self, "data_transforms", data_transforms)
        setattr(self, "data_groups", data_groups)

    def _infer_dimensions(self):
        data_groups = getattr(self, "data_groups")
        sample_row = self.dataset.iloc[0]
 
        for group in data_groups:
            preview = self._process_and_combine(sample_row, group)

            if isinstance(preview, pt.Tensor):
                types = as_list(data_groups[group].get("types", []))
                image_idxs  = [i for i,t in enumerate(types) if t == "image"]
                scalar_idxs = [i for i,t in enumerate(types) if t == "scalar"]

                data_groups[group]["channel_roles"] = {
                    "image": image_idxs,
                    "scalar": scalar_idxs
                }

                data_groups[group]["dimensions"] = tuple(preview.shape) # (C_total, H, W)
                data_groups[group]["dimensions"] = tuple(preview.shape) # (C_total, H, W)
            elif isinstance(preview, tuple):
                dims = tuple(t.shape for t in preview if isinstance(t, pt.Tensor))
                data_groups[group]["dimensions"] = dims

            else:
                raise ValueError(f"Cannot infer tensor shape for data group '{group}'")
                
    def _infer_max_length(self):
        S = 3
        all_cols = sum(self.group_data_keys.values(), [])

        def _parse(v):
            if isinstance(v, str):
                try:
                    return ast.literal_eval(v)
                except Exception:
                    return None
            return v

        for name in all_cols:
            for v in self.dataset[name].values:
                v = _parse(v)
                if v is None:
                    continue

                x = v if isinstance(v, pt.Tensor) else pt.tensor(np.asarray(v), dtype=pt.float32)
                if x.ndim == 0:
                    continue
                if x.ndim == 1:
                    S = max(S, int(x.numel()))
                    continue

                H, W = int(x.shape[-2]), int(x.shape[-1])
                if H != W:
                    raise ValueError(f"Non-square image in '{name}': H={H}, W={W}")
                
                S = max(S, H)

        setattr(self, "expected_hw", (S, S))


    def get_metadata(self):

        return {
            "random_seed": getattr(self, "random_seed", None),
            "batch_size": getattr(self, "batch_size", None),
            "batch_shuffle": getattr(self, "batch_shuffle", False),
            "subset_names": getattr(self, "subset_names", []),
            "subset_split": getattr(self, "subset_split", {}),
            "subset_sizes": getattr(self, "subset_sizes", {}),
            "transform_method": getattr(self, "transform_method", None),
            "data_transforms": getattr(self, "data_transforms", []),
            "unique_id_name": getattr(self, "unique_id_name", None),
            "group_names": getattr(self, "group_names", []),
            "data_groups": getattr(self, "data_groups", {}),
        }  
        
    def split_dataset(self):
        subsets = random_split(self, self.subset_sizes, generator=self.generator)
        subset_maps = {}
        for i, (name, sub) in enumerate(zip(self.subset_names, subsets)):
            subset_maps[name] = sub
        return subset_maps

    def save_dataset(self, output_path):
        create_folder(output_path)
        # Save metadata to JSON and path file
        metadata = self.get_metadata()
        metadata_path = create_file_path(output_path, "metadata.json")
        save_to_json(metadata_path, metadata)
        pt.save(metadata, metadata_path.replace("json", "pth"))

        # Save transforms
        data_transforms = getattr(self, "data_transforms", {})
        if isinstance(self.transform_method, str) and data_transforms:
            transforms_path = create_file_path(output_path, "transforms.pth")
            pt.save(data_transforms, transforms_path)

        # Save subsets
        subsets = self.split_dataset()
        for name, subset in subsets.items():
            subset_path = create_file_path(output_path, f"{name}_subset.pth")
            pt.save(subset, subset_path)


class BatchSubset(Dataset):
    def __init__(self, root_folder, subset_name):
        self.root_dir = root_folder
        self.subset_name = subset_name.lower()

        self.metadata_file = os_path.join(self.root_dir, 'metadata.pth')
        self.metadata = self.safe_load(self.metadata_file, required=True)

        self.subset_file = os_path.join(self.root_dir, f"{self.subset_name}_subset.pth")
        self.subset = self.safe_load(self.subset_file) or []

        self.transform_method = self.metadata.get("transform_method", None)
        self.transforms_file = os_path.join(self.root_dir, 'transforms.pth')
        self.data_transforms = self.safe_load(self.transforms_file)
        
        self.batch_shuffle = self.metadata.get("batch_shuffle", False) and self.subset_name == 'training'
        self.batch_size = 1 if self.subset_name == 'testing' else self.metadata.get("batch_size", 1) 

        self.subset_names = self.metadata.get("subset_names", {})
        self.subset_sizes = self.metadata.get("subset_sizes", {})

        self.group_names = self.metadata.get("group_names", {})
        self.data_groups = self.metadata.get("data_groups", {})
        self.unique_id_name = self.metadata.get("unique_id_name", {})


    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]

    def __bool__(self):
        return len(self) > 0

    @staticmethod
    def _collate_dict(batch):
        batched = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if isinstance(values[0], pt.Tensor):
                batched[key] = pt.stack(values)
            elif isinstance(values[0], tuple):
                transposed = zip(*values)
                batched[key] = tuple(pt.stack(tensors) for tensors in transposed)
            else:
                batched[key] = values
        return batched

    @staticmethod
    def _worker_init_fn(worker_id):
        base_seed = convert_type(pt.initial_seed(), int) % 2**32
        np.random.seed(base_seed + worker_id)
        random.seed(base_seed + worker_id)
        
    def safe_load(self, path, required=False):
        if os_path.exists(path):
            return pt.load(path, weights_only=False)
        if required:
            raise FileNotFoundError(f"Cannot load file from: {path}")
        return {}

    def get_dataloader(self, num_workers=1): 
        if not self:
            get_logger().warning(f"No subset data found for '{self.subset_name}'.")
            return None

        loader = DataLoader(self, 
                        batch_size=self.batch_size, 
                        shuffle=self.batch_shuffle, 
                        persistent_workers=True, 
                        num_workers=num_workers, 
                        worker_init_fn=self._worker_init_fn,
                        collate_fn=self._collate_dict)

        input_group = self.data_groups.get("input", {})
        target_group = self.data_groups.get("target", {})
        solver_group = self.data_groups.get("solver", {})

        loader.input_names = input_group.get("names", [])
        loader.target_names = target_group.get("names", [])
        loader.solver_names =  solver_group.get('names', [])

        loader.input_channel_roles = input_group.get("channel_roles", {})
        loader.target_channel_roles = target_group.get("channel_roles", {})

        loader.input_transforms = self.data_transforms.get("input", {})
        loader.target_transforms = self.data_transforms.get("target", {})

        return loader


    def inspect_batches(self, max_batches=1):
        dataloader = self.get_dataloader()
        div = f"{'-'*60}"

        for i, batch in enumerate(dataloader):
            lines = [
                f"\n{div}",
                f"{self.subset_name.title()} Batch #{i}",
                div
            ]

            for key, val in batch.items():
                if isinstance(val, pt.Tensor):
                    lines.append(f"[{key}]: Tensor, shape = {tuple(val.shape)}, dtype = {val.dtype}")
                elif isinstance(val, tuple):
                    shapes = [tuple(v.shape) for v in val]
                    dtypes = [v.dtype for v in val]
                    lines.append(f"[{key}]: Tuple[{len(val)}]")
                    for j, (s, d) in enumerate(zip(shapes, dtypes)):
                        branch = "└─" if j == len(shapes) - 1 else "├─"
                        lines.append(f"    {branch} [{j}]: shape = {s}, dtype = {d}")
                elif isinstance(val, list):
                    lines.append(f"[{key}]: List, size = {len(val)}, dtype = {val[0].dtype}")
                else:
                    lines.append(f"[{key}]: {type(val).__name__}")
            lines.append(f"{div}\n")
            get_logger().debug("\n" + "\n".join(lines))

            if i + 1 >= max_batches:
                break

class StandardScalerTransform:
    def __init__(self, device='cpu'):
        self.device = device
        self.dtype = pt.float32
        self.mean_data = None
        self.std_data = None

        self.shape = None
        self.fitted = False

        self.original_range = MinMaxTuple(min=None, max=None)
        self.scaled_range = MinMaxTuple(min=None, max=None)

    def fit(self, data, **kwargs):
        data = data.to(device=self.device, dtype=self.dtype)
        self.original_range = MinMaxTuple(min=data.min().item(), max=data.max().item())

        if data.dim() == 2:
            dimen = 0
        elif data.dim() == 3:
            dimen = (1, 2)
        elif data.dim() == 4:
            dimen = (0, 2, 3)
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        self.mean_data = data.mean(dim=dimen, keepdim=True)
        self.std_data  = data.std(dim=dimen, keepdim=True, unbiased=False)
        self.shape = list(self.mean_data.shape)
        
        safe_std, _ = self.get_safe_std(self.std_data)
        standardized = (data - self.mean_data) / safe_std
        self.scaled_range = MinMaxTuple(min=standardized.min().item(), max=standardized.max().item())

        self.fitted = True
        return self

    def get_safe_std(self, std_data):
        zero_std_mask = std_data == 0
        safe_std = std_data.clone()
        safe_std[zero_std_mask] = 1.0  
        return safe_std, zero_std_mask
    
    def __call__(self, data):
        if self.mean_data is None or self.std_data is None:
            raise ValueError("StandardScalerTransform has not been fitted yet.")
        data = data.to(device=self.device, dtype=self.dtype)

        mean_data = match_shape(self.mean_data, data)
        std_data  = match_shape(self.std_data, data)

        safe_std, zero_std_mask = self.get_safe_std(std_data)
        standardized = (data - mean_data) / safe_std
        standardized = pt.where(zero_std_mask, pt.zeros_like(standardized), standardized)

        return standardized.to(self.dtype)

    def inverse_transform(self, data, new_device=None):
        if self.mean_data is None or self.std_data is None:
            raise ValueError("StandardScalerTransform has not been fitted yet.")
    
        if new_device is None:
            new_device = data.device
    
        compute_device = pt.device(self.device)
        data = data.to(device=compute_device, dtype=self.dtype)
    
        std_data  = match_shape(self.std_data, data).to(device=data.device, dtype=data.dtype)
        mean_data = match_shape(self.mean_data, data).to(device=data.device, dtype=data.dtype)
    
        original = data * std_data + mean_data
        original = original.clamp(self.original_range.min, self.original_range.max)

        return original.to(device=new_device, dtype=self.dtype)


class MinMaxScalerTransform:
    def __init__(self, min_val=0.0, max_val=1.0, device='cpu', dtype=pt.float32):

        self.dtype = dtype
        self.device = device
        self.min_val = pt.tensor(min_val, dtype=self.dtype, device=self.device)
        self.max_val = pt.tensor(max_val, dtype=self.dtype, device=self.device)
        self.min_data = None
        self.max_data = None
        self.fitted = False
        self.original_range = MinMaxTuple(min=None, max=None)
        self.scaled_range = MinMaxTuple(min=min_val, max=max_val)

    def fit(self, data, is_scalar=True):
        data = data.to(device=self.device, dtype=self.dtype)
    
        self.original_range = MinMaxTuple(min=data.min().item(), max=data.max().item())

        if is_scalar:
            self.min_data = data.min(dim=0, keepdim=False)[0]
            self.max_data = data.max(dim=0, keepdim=False)[0]
        else:
            if data.dim() == 2: 
                dimen = 0
            elif data.dim() == 3: 
                dimen = (1, 2)
            elif data.dim() == 4:  
                dimen = (0, 2, 3)
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")
            self.min_data = data.amin(dim=dimen, keepdim=True)
            self.max_data = data.amax(dim=dimen, keepdim=True)

        self.fitted = True

        return self

    def __call__(self, data):
        if self.min_data is None or self.max_data is None:
            raise ValueError("MinMaxScalerTransform has not been fitted with min and max values.")

        data = data.to(device=self.device, dtype=self.dtype)

        min_data = match_shape(self.min_data, data)
        max_data = match_shape(self.max_data, data)

        distance_from_min = data - min_data
        minmax_range = max_data - min_data
        minmax_range[minmax_range == 0] = 1e-15  # Avoid division by zero

        normalized_data = distance_from_min / minmax_range
        scaled_data = normalized_data * (self.max_val - self.min_val) + self.min_val

        return scaled_data.to(self.dtype)

    def inverse_transform(self, data, new_device=None):
        if self.min_data is None or self.max_data is None:
            raise ValueError("MinMaxScalerTransform has not been fitted with min and max values.")

        if new_device is None:
            new_device = data.device

        data = data.to(device=self.device, dtype=self.dtype)

        normalized = (data - self.min_val) / (self.max_val - self.min_val)
        min_data = match_shape(self.min_data, data)
        max_data = match_shape(self.max_data, data)

        original = normalized * (max_data - min_data) + min_data
        original = original.clamp(self.original_range.min, self.original_range.max)

        return original.to(device=new_device, dtype=self.dtype)