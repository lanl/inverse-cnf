from programs.utils.logger_setup import get_logger

from sys import modules as sys_modules
from os import makedirs, path as os_path, getpid, cpu_count, remove as os_remove, listdir, environ, rename
from json import dump as json_dump, load as json_load, dumps as json_dumps, loads as json_loads

from collections import namedtuple, defaultdict
from functools import partial, wraps
from itertools import combinations
from importlib import import_module
from typing import List, Tuple, Dict, Type

import glob
import h5py
import re
re_match = re.match

import numpy as np
import pandas as pd
import traceback as tb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolor

import seaborn as sns
import scipy.ndimage as ndimg
import scipy.sparse as sparse

import pprint
import warnings
import torch as pt
import gc
import random
import ast

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


FLOAT_EPS = 1E-15


PT_NN_MODULE = Type[pt.nn.Module]
PT_TENSOR = pt.Tensor


def get_timestamp():
    return pd.Timestamp.now().strftime('%y%b%d_%H%M%S')


def numeric_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def as_list(value: any) -> list:
    if value is None:
        return []
    return [value] if isinstance(value, (tuple, str, int, float)) else value


def convert_type(val, typ=float):
    try:
        return typ(val) if val is not None else None
    except (ValueError, TypeError):
        return val


def extract_item(value: any) -> any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def extract_filename_number(filename, occurance=1) -> int|None:
    match = re.search(r'(\d+).', filename)
    return int(match.group(occurance)) if match else None


def import_module_path(module_path):
    try:
        module_name, class_name = module_path.rsplit('.', 1)
        module = import_module(module_name)
        return getattr(module, class_name)
    except ImportError:
        raise ImportError(f"Cannot import module from '{module_path}'. Please specify absolute path to module.")
    except AttributeError:
        raise AttributeError(f"Module '{module_name}' does not have a class or function '{class_name}'")


def validate_image_pair(func):
    def _parse_array(array_in):
        array_out = array_in
        if isinstance(array_out, pt.Tensor):
            array_out = array_out.cpu().numpy()
        array_out = np.asarray(array_out, dtype=np.float32)
        return array_out
    @wraps(func)
    def wrapper(array_a, array_b, *args, **kwargs):
        arr_a = _parse_array(array_a)
        arr_b = _parse_array(array_b)
        if arr_a.shape != arr_b.shape:
            raise ValueError(f"Expected array shape {array_a.shape} to match output array shape {array_b.shape}")
        return func(array_a, array_b, *args, **kwargs)
    return wrapper


def dataframe_records_converter(data):
    if isinstance(data, pd.DataFrame):
        dict_list = data.to_dict(orient='records')
        return dict_list
    elif isinstance(data, list) and isinstance(data[0], dict):
        df = pd.DataFrame(data)
        return df
    else:
        raise TypeError(f"Expected types 'pd.Dataframe' or 'dict[]' for conversion, but recieved type '{type(data)}'")
    

def pretty_dict(d: dict, label: str|None = None, indent: int = 4) -> str:
    if not isinstance(d, dict):
        raise TypeError(f"Cannot pretty format object of type '{type(d).__name__}' (expected dict)")
    
    pretty = f"\n{pprint.pformat(d, indent=indent)}"

    if isinstance(label, str) and len(label.strip()) > 0:
        return f"\n[{label}]:{pretty}"
    
    return f"\n{pretty}"



def search_dict(d, parent_key='', sep='_', flatten=False):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if (flatten and parent_key.strip()) else k
        if isinstance(v, dict):
            items.extend(
                search_dict(
                    v,
                    new_key if flatten else "",
                    sep=sep,
                    flatten=flatten,
                ).items()
            )
        else:
            items.append((new_key, v))
    return dict(items)


def match_shape(data_in, target):
    data = data_in.clone()
    while data.ndim > target.ndim:
        data = data.squeeze(0)
    if data.shape != target.shape:
        data = data.expand_as(target)
    return data


# creates a folder path if it doesn't exist
def create_folder(folder_name):
    if os_path.isdir(folder_name):
        return os_path.abspath(folder_name)
    folder_path=os_path.dirname(os_path.abspath(__file__))
    dir_path = os_path.join(folder_path, folder_name)

    makedirs(dir_path, exist_ok=True)
    if not os_path.exists(dir_path):
        get_logger().error(f"Cannot create folder '{folder_name}' in path: {folder_path}")
    return dir_path


def create_file_path(folder_path, file_name):
    out_path = create_folder(folder_path)
    file_path = os_path.join(out_path, file_name)
    return file_path


def remove_if_exists(file_path=None):
    if file_path and os_path.exists(file_path):
        os_remove(file_path) 

def match_file_path(file_path=None, first=True):
    expanded_path = os_path.expandvars(file_path)
    matched_files = glob.glob(expanded_path)
    if not matched_files:
        return expanded_path
    elif first:
        return matched_files[0]
    else:
        return matched_files[-1]

# writes or appends to a hdf5 file
def save_to_hdf5(data_dict_list, file_path, chunk_size=None, flatten=False):
    def write_data_to_group(group, data):
        for key, value in data.items():
            if isinstance(value, (int, float)):
                group.attrs[key] = value
                #get_logger().debug(f"Saved attribute: {key} => {value}")
            elif isinstance(value, str):
                string_dt = h5py.string_dtype(encoding='utf-8')
                group.create_dataset(key, data=value, dtype=string_dt)
                #get_logger().debug(f"Saved string dataset: {key} => {value}")
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
                #get_logger().debug(f"Saved array dataset: {key} => array with shape {value.shape}")
            elif isinstance(value, dict):
                subgroup = group.create_group(key)
                write_data_to_group(subgroup, value) 
            else:
                raise TypeError(f"Invalid type for {key}: {type(value)}. Expected int, float, str, or np.ndarray.")
            
    chunk = chunk_size or 1
    mode = 'a' if os_path.exists(file_path) else 'w'  
    try:
        with h5py.File(file_path, mode) as f:
            existing_indices = sorted(
                [int(k.split('_')[1]) for k in f.keys() if k.startswith("record_")],
                reverse=True
            )
            current_max_index = existing_indices[0] + 1 if existing_indices else 0
            total_records = len(data_dict_list)
            get_logger().debug(f"Saving {total_records} records to file starting at index {current_max_index}")

            for i in range(0, total_records, chunk):
                for idx in range(i, min(i + chunk, total_records)):
                    record_index = current_max_index + idx
                    record_group_name = f"record_{record_index}"

                    if record_group_name in f:
                        get_logger().debug(f"Skipping existing group: {record_group_name}")
                        continue
                    record_dict = search_dict(data_dict_list[idx], flatten=flatten)
                    record_group = f.create_group(record_group_name)
                    write_data_to_group(record_group, record_dict)
                    get_logger().debug(f"Created group: {record_group_name}")

    except (Exception, OSError, IOError, TypeError) as e:
        get_logger().error(f"Error writing to HDF5 file {file_path}: {e}")


# read a hdf5 file in or a random # of samples 
def read_from_hdf5(file_path, sample_size=None, chunk_size=None, flatten=False, random_seed:None|int=None):
    def load_group_data(group):
        group_dict = {}
        group_dict.update({k: v for k, v in group.attrs.items()})

        for key, item in group.items():
            if isinstance(item, h5py.Group):
                subgroup_data = load_group_data(item)
                group_dict.update(search_dict(subgroup_data, parent_key=key, flatten=flatten))

            else:
                value = item[()] if item.shape == () else item[:]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                elif isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O"}:
                    value = np.array([v.decode("utf-8") if isinstance(v, bytes) else v for v in value])

                group_dict[key] = value

        return group_dict

    rng = np.random.default_rng(random_seed)

    chunk = chunk_size or 1
    data_dict_list = []
    try:
        with h5py.File(file_path, 'r') as f:
            all_keys = list(f.keys())
            if sample_size and 0 < sample_size < len(all_keys):
                selected_keys = rng.choice(all_keys, sample_size, replace=False)
            else:
                selected_keys = all_keys

            for i in range(0, len(selected_keys), chunk):
                chunk_keys = selected_keys[i:i + chunk]
                for key in chunk_keys:
                    group = f[key]
                    data = load_group_data(group)
                    data_dict_list.append(data)
            return data_dict_list

    except (OSError, IOError, TypeError) as e:
        get_logger().error(f"Cannot read from HDF5 file: {file_path} due to: {e}")


# reads json file
def read_from_json(file_path, deserialize=False):
    try:
        with open(file_path, 'r') as json_file:
            content = json_load(json_file)
        return deserialize_dict(content) if deserialize else content
    except Exception as e:
        get_logger().error(e)


# writes or appends to a csv file
def save_to_json(file_path, content, mode='w', indent=4, serialize=True, sort_keys=False):
    try:
        if serialize:
            if isinstance(content, dict):
                content = serialize_dict(content)
            elif isinstance(content, list):
                content = [serialize_dict(item) if isinstance(item, dict) else item for item in content]
        with open(file_path, mode) as json_file:
            json_dump(content, json_file, indent=indent, ensure_ascii=False, sort_keys=sort_keys)
    except Exception as e:
        get_logger().error(f"Error saving to JSON: {e}")
        raise


def serialize_dict(input_dict):
    serializable_params = {}

    for key, value in input_dict.items():
        try:
            if isinstance(value, dict):
                serializable_params[key] = serialize_dict(value)
            elif isinstance(value, (tuple, list)):
                if hasattr(value, '_fields'):  # For namedtuples
                    serializable_params[key] = {
                        "type": "namedtuple",
                        "data": serialize_dict(value._asdict()),
                        "name": value.__class__.__name__,
                        "module": value.__class__.__module__,
                    }
                else:
                    serializable_params[key] = {
                        "type": "tuple" if isinstance(value, tuple) else "list",
                        "data": [serialize_dict(v) if isinstance(v, dict) else v for v in value],
                    }
            elif isinstance(value, (str, int, float, bool)):
                serializable_params[key] = value
            elif value is None:
                serializable_params[key] = {"type": "null"}
            elif isinstance(value, (np.floating)):
                serializable_params[key] = float(value)
            elif isinstance(value, (np.integer)):
                serializable_params[key] = int(value)
            elif isinstance(value, (np.ndarray, pt.Tensor)):
                serializable_params[key] = {
                    "type": "tensor" if isinstance(value, pt.Tensor) else "ndarray",
                    "data": value.tolist(),
                    "dtype": str(value.dtype),
                }
            elif isinstance(value, pt.device):
                serializable_params[key] = {
                    "type": "object",
                    "module": "torch",
                    "class": "device",
                    "data": str(value)
                }
            elif isinstance(value, type): 
                serializable_params[key] = {
                    "type": "class",
                    "value": repr(value),
                }
            elif callable(value):
                serializable_params[key] = {
                    "type": "callable",
                    "module": value.__module__,
                    "name": getattr(value, "__name__", type(value).__name__),
                }
            elif isinstance(value, object):
                serializable_params[key] = {
                    "type": "object",
                    "module": value.__class__.__module__,
                    "class": value.__class__.__name__,
                }
            else:
                serializable_params[key] = repr(value)
        except Exception as e:
            get_logger().error(f"Serialization error with key '{key}', value '{value}', type '{type(value)}': {e}")

    return serializable_params


def resolve_function_from_string(function_string):
    module_name, function_name = function_string.rsplit('.', 1)
    module = import_module(module_name)
    return getattr(module, function_name)

def resolve_class_from_string(class_string):
    if class_string.startswith("<class '") and class_string.endswith("'>"):
        class_string = class_string[len("<class '"):-len("'>")]
    module_name, class_name = class_string.rsplit('.', 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def deserialize_dict(serialized_dict):
    deserialized_params = {}

    for key, value in serialized_dict.items():
        try:
            if isinstance(value, dict) and "type" in value:
                if value["type"] == "list":
                    deserialized_params[key] = [deserialize_dict(v) if isinstance(v, dict) else v for v in value["data"]]
                elif value["type"] == "tuple":
                    deserialized_params[key] = tuple(deserialize_dict(v) if isinstance(v, dict) else v for v in value["data"])
                elif value["type"] == "namedtuple":
                    try:
                        module = import_module(value["module"])
                        namedtuple_cls = getattr(module, value["name"])
                    except (ImportError, AttributeError):
                        namedtuple_cls = namedtuple(value["name"], value["data"].keys())
                    deserialized_params[key] = namedtuple_cls(**deserialize_dict(value["data"]))
                elif value["type"] == "null":
                    deserialized_params[key] = None
                elif value["type"] == "class":
                    deserialized_params[key] = resolve_class_from_string(value["value"])
                elif value["type"] == "callable":
                    deserialized_params[key] = resolve_function_from_string(value["module"] + "." + value["name"])
                elif value["type"] == "object":
                    try:
                        if value["class"] == "device" and value["module"] == "torch":
                            deserialized_params[key] = pt.device(value.get("data", "cpu"))
                        else:
                            module = import_module(value["module"])
                            cls = getattr(module, value["class"])
                            deserialized_params[key] = cls()
                    except (ImportError, AttributeError) as e:
                        raise ValueError(f"Failed to deserialize object for key '{key}': {e}")
                else:
                    deserialized_params[key] = value
            elif isinstance(value, dict):
                deserialized_params[key] = deserialize_dict(value)
            elif isinstance(value, list):
                deserialized_params[key] = [deserialize_dict(v) if isinstance(v, dict) else v for v in value]
            elif isinstance(value, tuple):
                deserialized_params[key] = tuple(deserialize_dict(v) if isinstance(v, dict) else v for v in value)
            else:
                deserialized_params[key] = value

        except Exception as e:
            get_logger().error(f"Deserialization error with key '{key}', value '{value}', type '{type(value)}': {e}")

    return deserialized_params


def process_stack(data: PT_TENSOR, unstack=True, squeeze=True) -> PT_TENSOR:
    tensor = pt.tensor(data, dtype=pt.float32).detach().cpu()
    if unstack:
        return [t.numpy() for t in tensor] 
    if squeeze:
        return np.squeeze(tensor.numpy())

    return tensor.numpy()


def inspect_batch_data(batch_data):
    log_str = "==== Inspecting batch_data ====\n"  # Start with the header for the inspection
    for key, value in batch_data.items():
        if hasattr(value, "shape"):
            log_str += f"{key}: {type(value)} | shape: {value.shape}\n"
        elif isinstance(value, (list, tuple)):
            log_str += f"{key}: {type(value)} | length: {len(value)}\n"
        else:
            log_str += f"{key}: {type(value)} | value: {value}\n"
        
        if isinstance(value, pt.Tensor):
            log_str += f"  Tensor dimensions: {value.dim()} | Tensor size: {value.size()}\n"
        
        if isinstance(value, (list, tuple)) and all(isinstance(v, pt.Tensor) for v in value):
            for i, v in enumerate(value):
                log_str += f"  {key}[{i}] | shape: {v.shape} | Tensor size: {v.size()}\n"
    get_logger().info(log_str)


def validate_scalar_shape(shape, required_channels=None):
    if not isinstance(shape, (tuple, list, pt.Size)) or shape[-1] != 1:
        return False

    channels = shape[0]
    if not isinstance(channels, int) or channels <= 0:
        return False

    if (isinstance(required_channels, int) and 
        required_channels > 0 and 
        channels != required_channels):
        return False
    return True


def validate_image_shape(shape, required_channels=None):
    if not (isinstance(shape, (tuple, list, pt.Size)) and len(shape) == 3):
        return False
    if not all(isinstance(d, int) and d > 0 for d in shape):
        return False
    if (isinstance(required_channels, int) and 
        required_channels > 0 and 
        shape[0] != required_channels):
        return False

    return True

def check_mixed_shape(shape_tuple):
    if isinstance(shape_tuple, (tuple, list)) and len(shape_tuple) == 2:
        tensor_shape1, tensor_shape2 = shape_tuple
        if validate_image_shape(tensor_shape1) and validate_scalar_shape(tensor_shape2):
            return True
    return False

def split_channel_roles(tensor: pt.Tensor, roles: dict) -> Tuple[pt.Tensor, pt.Tensor | None]:
    """
    Input: tensor of shape (B,C,H,W) or (C,H,W)
    Returns: (image_tensor, scalar_vector)
    """
    image_tensor = None
    scalar_tensor = None

    if tensor.dim() == 3:  # single sample
        image_tensor = tensor[roles['image'],:,:]
        if roles['scalar']:
            broadcasted_tensor = tensor[roles['scalar'],:,:]
            scalar_tensor = broadcasted_tensor.mean(dim=(1,2))

    if tensor.dim() == 4:  # batched
        image_tensor = tensor[:,roles['image'],:,:]
        if roles['scalar']:
            broadcasted_tensor = tensor[:,roles['scalar'],:,:]
            scalar_tensor = broadcasted_tensor.mean(dim=(2,3))
    
    return image_tensor, scalar_tensor


def unbroadcast_scalar(data: np.ndarray | PT_TENSOR) -> float | np.ndarray | PT_TENSOR | None:
    """
    Collapse broadcasted scalar maps back to scalars.
    - If `data` is (H, W): returns a float.
    - If `data` is (C, H, W): returns shape (len(scalar_idxs),).
    - If `data` is (B, C, H, W): returns shape (B, len(scalar_idxs)).
    Works on both NumPy arrays and Torch tensors.
    """
    if data.ndim == 2:
        return float(data.mean())  # (H, W) to scalar value

    elif data.ndim == 3:
        return data.mean(axis=(1, 2))  # (C, H, W) to (len(scalar_idxs),)

    elif data.ndim == 4:
        return data.mean(axis=(2, 3))  # (B, C, H, W) to (B, len(scalar_idxs))

    else:
        raise ValueError(f"unsupported ndim {data.ndim}")

def unmap_vector(x: pt.Tensor, *, S: int, mode: str) -> pt.Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if tuple(x.shape) != (1, S, S):
        raise ValueError(f"Expected (1,{S},{S}), got {tuple(x.shape)}")

    if mode == "broadcast":      # L == S
        return x[0, :, 0].contiguous()          # (S,)
    if mode == "reshape":        # L == S*S
        return x[0].reshape(-1).contiguous()    # (S*S,)

    raise ValueError("mode must be 'broadcast' or 'reshape'")