from programs.utils.common import pt, np, re, import_module_path, as_list
from programs.core.solver_functions import SOLVER_MODULES, SOLVER_KEYS, SOLVER_DEFAULT, SOLVER_PARAMETERS

class SolverSetup:
    def __init__(self, solver_modules=SOLVER_MODULES, solver_keys=SOLVER_KEYS, solver_parameters=SOLVER_PARAMETERS):
        self.solver_modules = solver_modules
        self.solver_keys = solver_keys
        self.solver_parameters = solver_parameters

    def setup_solver_function(self, solver_key, data_groups):
        if solver_key == "bypass":
            return None

        input_names = data_groups.get("input", {}).get("names", [])
        target_names = data_groups.get("target", {}).get("names", [])
        solver_names = data_groups.get("solver", {}).get("names", [])

        target_names = as_list(target_names)
        included_params = as_list(input_names) + target_names[1:] + as_list(solver_names)
        default_kwargs = self._check_params(solver_key, included_params, target_names[0])
        return self._get_function(solver_key, default_kwargs)

    def _get_function(self, solver_name: str, default_kwargs: dict = None):
        if solver_name == "bypass":
            return None

        if solver_name not in self.solver_keys:
            raise ValueError(f"Solver '{solver_name}' not found in {self.solver_keys}")

        module_path = self.solver_modules[solver_name]
        solver_fn = import_module_path(module_path)
        if solver_fn is None:
            raise AttributeError(f"'{module_path}' does not define '{solver_name}_solver'")

        default_kwargs = default_kwargs or {}
        def wrapped_solver(params: dict[str, np.ndarray | int | float]) -> pt.Tensor:
            kwargs = {**params, **default_kwargs}
            return solver_fn(**kwargs)

        return wrapped_solver

    def _check_params(self, solver_key, included_params, target_name):
        params = self.solver_parameters.get(solver_key, None)
        required_params = params["required"]
        default_params = params["default"]

        if required_params is None:
            raise KeyError(f"Cannot find required parameters for solver_key '{solver_key}'")

        if "total_iterations" in required_params:
            match = re.match(r".*_state_(\d+)$", target_name)
            if match:
                default_params["total_iterations"] = int(match.group(1))

        captured_params = set(included_params + list(default_params.keys()))
        for param in captured_params:
            if param in required_params:
                required_params.remove(param)

        for param in required_params:
            if param not in included_params:
                raise KeyError(f"Missing required parameter for solver '{solver_key}': '{param}'")

        return default_params