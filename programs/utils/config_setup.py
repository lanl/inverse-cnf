from programs.utils.logger_setup import get_logger
from programs.utils.common import os_path, import_module, match_file_path, create_file_path, read_from_json, save_to_json
from jsonschema import validate, ValidationError
from programs.utils.arguments import ap, executable_groups, process_args, log_args, check_args

task_modules = {}
for module_name in executable_groups.keys():
    try:
        task_modules[module_name] = import_module(f"programs.tasks.{module_name}")
    except ModuleNotFoundError as e:
        get_logger().error(f"Error loading command module '{module_name}': {e}")


def remove_none_values(data):
    return {key: value for key, value in data.items() if value is not None}


def generate_cli_schema(arg_parser, program_name):
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": f"{ program_name}_cli",
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": True
    }
    prop_object = schema["properties"]

    for action in arg_parser._actions:
        if not action.dest or action.dest == "help":
            continue
        action_name = action.option_strings[0].strip('--')


        arg_type = action.type
        if action.type is not None:
            arg_type = action.type
        elif action.nargs == 0 and action.const is not None:
            arg_type = type(action.const)
        if arg_type in [int, float]:
            json_type = "number"
        elif arg_type in [list, tuple]:
            json_type = "array"
        elif arg_type == bool:
            json_type = "boolean"
        elif arg_type == str:
            json_type = "string"
        else:
            raise ap.ArgumentTypeError(f"{action_name} is not a valid argument type")

        prop_object[action_name] = {
            "help": action.help,
            "default": action.default
        }

        enum_choices=None
        if action.choices:
            enum_choices = (action.choices + [None] if not action.required else action.choices)

        if str(action.nargs) in ['+', '*']:
            prop_object[action_name]["oneOf"] = [
                    {"type": json_type}, 
                    {"type": "array", "items": {"type": json_type}}
                ]   
            if action.choices:
                prop_object[action_name]["oneOf"][0]["enum"] = enum_choices
                prop_object[action_name]["oneOf"][1]["items"]["enum"] = enum_choices

        elif isinstance(action.nargs, int) and int(action.nargs) > 1:
            prop_object[action_name] = {
                                    "type": "array", 
                                    "items": {
                                        "type":json_type
                                    }, 
                                    "minItems":action.nargs, 
                                    "maxItems":action.nargs
                                }
            if action.choices:
                prop_object[action_name]["items"]["enum"] = enum_choices
        else:
            prop_object[action_name]["type"] = json_type
            if action.choices:
                prop_object[action_name]['enum'] = enum_choices
                                    
        if action.required:
            schema["required"].append(action_name)

    root_path = os_path.abspath(os_path.dirname("src"))
    schema_file = create_file_path(f"{root_path}/schemas", f"{program_name}_schema.json")
    save_to_json(schema_file, schema, serialize=False)
    schema_content = read_from_json(schema_file, deserialize=False)
    return schema_content

def validate_cli_schema(config_path, parser, program_name):
    try:
        config_data = read_from_json(config_path, deserialize=False)
        config_content = remove_none_values(config_data)
        schema_content = generate_cli_schema(parser, program_name)
        validate(instance=config_content, schema=schema_content)
        get_logger().info(f"Validated config for {program_name}")
        return config_content
    except ValidationError as e:
        get_logger().error(f"Config validation error: {e}")

def process_cli_config(cli_data):
    cli_args = []
    for key, value in cli_data.items():
        cli_key = f"--{key.replace('_', '-')}"
        if str(value).lower() in ['null', 'none', 'false']:
            continue
        elif isinstance(value, bool):
            cli_args.append(cli_key)
        elif isinstance(value, list):
            cli_args.append(cli_key)
            cli_args.extend(map(str, value))
        elif isinstance(value, str):
            cli_args.append(cli_key)
            str_val = match_file_path(value)
            cli_args.append(str_val)
        elif isinstance(value, (int, float)):
            cli_args.append(cli_key)
            cli_args.append(str(value))
        else:
            get_logger().warning(f"Unsupported value type for key '{key}': {value}")
    get_logger().debug(f"Processed CLI Args: {cli_args}")
    return cli_args

def validate_config_args(config_path, program_name, show_args=True):
    prog_parser = process_args(program_name, parser_only=True)
    config_data = validate_cli_schema(config_path, prog_parser, program_name)
    processed_config = process_cli_config(config_data)
    prog_args = prog_parser.parse_args(processed_config)
    valid_args = check_args(prog_args, program_name)
    if show_args:
        log_args(valid_args)
    return valid_args
    