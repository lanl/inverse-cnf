from programs.utils.logger_setup import get_logger
from programs.utils.common import os_path, np, pt, as_list, import_module, import_module_path, create_file_path, save_to_json
from programs.core.loss_functions import LOSS_MODULES


MODEL_MODULES = {
    'cfn': 'programs.models.ConditionalNF'
}

MODEL_KEYS = list(MODEL_MODULES.keys())

LOSS_TYPES = list(LOSS_MODULES.keys())

MODEL_LOSS_COMPATIBILITY = {
    'cfn': {'mse', 'l1', 'smooth-l1', 'ssim', 'ms-ssim', 'tv'},
}

ACTIVATION_MODULES = {
    'identity': 'torch.nn.Identity',
    'relu': 'torch.nn.ReLU',                
    'leaky-relu': 'torch.nn.LeakyReLU',     
    'elu': 'torch.nn.ELU',                  
    'selu': 'torch.nn.SELU',                
    'tanh': 'torch.nn.Tanh',                
    'sigmoid': 'torch.nn.Sigmoid',  
    'silu': 'torch.nn.SiLU',                
    'gelu': 'torch.nn.GELU'                        
}

ACTIVATION_TYPES = list(ACTIVATION_MODULES.keys())


ALL_MODULES = {
    'loss_function': LOSS_MODULES,
    'activation_function': ACTIVATION_MODULES,
    'projection_activation': ACTIVATION_MODULES
}

MODULE_TYPES = list(ALL_MODULES.keys())

def get_module_import(module_type:str, module_key:str):
    """
    Retrieves the specified configuration from model_config.
    
    Parameters:
    - module_type (str): Type of module in MODULE_TYPES.
    - model_key (object): Name for the for module to import from module list
    
    Returns:
    -  config module or function set in model_config for the specified type.
    """

    if module_type not in MODULE_TYPES:
        raise ValueError(f"Module type'{module_type}' not found in module type list:{MODULE_TYPES}")

    module_list = ALL_MODULES[module_type]

    if module_key not in list(module_list.keys()):
        raise ValueError(f"Unknown module '{module_key}' from from module type list '{list(module_list.keys())}'")
    
    module_str = module_list[module_key]

    module = import_module_path(module_str)

    return module

def set_config_module(config:dict, module_type:str, module_key:str):
    """
    Sets the specified configuration in model_config.
    
    Parameters:
    - config (object): The configuration object to update.
    - module_type (str): type pf module in MODULE_TYPES:['loss_function','activation_function','lr_scheduler']
    - module_key (str): Key name of the module to load from module lists: LOSS_TYPES, ACTIVATION_TYPES, LRSCHEDULER_TYPES
    
    Returns:
    - Updated config with the selected configuration.
    """
    module_type = module_type.lower()
    module_key =  module_key.lower()

    if module_type not in config:
        raise ValueError(f"Module type {module_type} not found in config keys: {list(config.keys())}")

    try:
        module = get_module_import(module_type, module_key)
        config[f"{module_type}_key"] = module_key
        config[module_type] = module
    except Exception as e:
        raise

def get_default_config(model_key):
    model_key = model_key.lower()
    if model_key not in MODEL_MODULES:
        raise ValueError(f"Unknown model type: {model_key}")

    model_module = import_module(MODEL_MODULES[model_key])
    model_config = getattr(model_module, 'model_config')
    return model_config

def setup_loss_instance(model_params):
    # pixel loss
    loss_key = model_params['loss_function_key']
    if loss_key not in LOSS_TYPES:
        get_logger().error(f"Loss function '{loss_key}' does not exist")
    loss_module = model_params['loss_function']
    loss_name = loss_key.upper()

    # latent loss
    latent_loss_key = model_params["latent_loss_function"]
    latent_loss_fn = get_module_import("loss_function", latent_loss_key)
    model_params['latent_loss_name'] = latent_loss_key.upper()
    model_params['latent_loss_instance'] = latent_loss_fn()

    # beta decay scheduling setup
    beta_warmup_type = model_params['beta_schedule_params']['warmup_type']
    beta_warmup_value = model_params['beta_schedule_params']['warmup_value']
    if beta_warmup_type == "fixed":
        beta_warmup_epochs = beta_warmup_value
    else:
        beta_schedule_epochs = model_params['beta_schedule_epochs']
        beta_warmup_epochs = min(int(np.ceil(beta_schedule_epochs * beta_warmup_value)), beta_schedule_epochs)
    model_params['beta_warmup_epochs'] = beta_warmup_epochs

    # data range config for SSIM loss and error
    target_group = model_params["data_groups"]["target"]
    data_dims = target_group["dimensions"]
    if isinstance(model_params["transform_method"], str):
        target_range = as_list(target_group["ranges"]["transformed"])[0]
    else:
        target_range = as_list(target_group["ranges"]["original"])[0]


    data_range = abs(target_range[1] - target_range[0])

    # custom loss function initialization
    if loss_key == 'weighted-hybrid':
        hybrid_loss_params = model_params['hybrid_loss_params']
        loss_fn = loss_module(loss_keys=hybrid_loss_params["functions"],
                                    weights=hybrid_loss_params["weights"], 
                                    device=model_params['device'],
                                    image_dims=data_dims, 
                                    data_range=data_range,
                                    multi_scale_params = model_params['multi_scale_params'])
        loss_name = loss_fn.custom_loss_name
    elif loss_key == 'ssim':
        loss_fn = loss_module(model_params['device'],
                            data_range=data_range)
    elif loss_key == 'ms-ssim':
        loss_fn = loss_module(data_dims, 
                                model_params['device'],
                                data_range=data_range,
                                kernel_size=model_params['multi_scale_params']['kernel'],
                                scale_weights=model_params['multi_scale_params']['weights'])
    else:
        loss_fn = loss_module()
    
    loss_instance = loss_fn.to(model_params['device']) if hasattr(loss_fn, "to") else loss_fn
    
    model_params['loss_instance'] = loss_instance
    model_params['loss_name'] = loss_name

    model_params['loss_function_names'] = {
        "training": f"{model_params['latent_loss_name']}_βx{model_params['loss_name']}",
        "validation": model_params['latent_loss_name'],
        "testing": model_params['loss_name']
    }

def setup_model_instance(model_params, output_folder: str):
    temp_params = model_params.copy()
    for param, value in temp_params.items():
        if param in MODULE_TYPES and isinstance(value, str):
            set_config_module(model_params, param, value) 

    setup_loss_instance(model_params)

    device = model_params['device']
    model_class = model_params['model_class']
    model = model_class(model_params).to(device)

    if isinstance(device, list):
        model = pt.nn.DataParallel(model)

    param_file = create_file_path(os_path.join(output_folder, "parameters"), "model_params.json")
    save_to_json(param_file, model_params, sort_keys=True)

    return model

