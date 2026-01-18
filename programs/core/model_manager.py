from programs.utils.logger_setup import get_logger
from programs.utils.common import np, pt, os_path, create_folder, namedtuple
NN_MODULE = pt.nn.Module
F = pt.nn.functional

EpochFrequency = namedtuple('EpochFrequency', ['checkpoint', 'training', 'validation', 'testing'])
ModelPhase = namedtuple('ModelPhase', ['T', 'V', 'E'])

def create_beta_scheduler(steps_per_epoch: int,
                            total_epochs: int,
                            warmup_epochs: int = 20,
                            min_value: float = 0.0,
                            max_value: float = 1.0,
                            mode: str = "cosine"):
    """
    Returns a function beta(step) that decays β from 1.0 → 0.0 over training steps.

    Modes:
        - "linear": linear decay
        - "cosine": smooth cosine decay
        - "exp": exponential decay
        - "constant": no decay, always `max_value`

    Args:
        steps_per_epoch (int): Number of steps per epoch
        total_epochs (int): Total number of epochs
        warmup_epochs (int): Epochs to hold β at `max_value` before decaying
        min_value (float): Final β value after decay
        max_value (float): Initial β value before decay
        mode (str): Decay strategy

    Returns:
        beta_fn(epoch, batch): Callable that returns β at a given epoch step
    """

    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    decay_steps = total_steps - warmup_steps

    range_scale = max_value - min_value

    factor_map = {
        "linear": lambda s: 1.0 - s,
        "cosine": lambda s: 0.5 * (1 + np.cos(np.pi * s)),
        "exp": lambda s: np.exp(-5 * s),
        "constant": lambda _: 1.0
    }

    factor_fn = factor_map.get(mode)
    if factor_fn is None:
        raise ValueError(f"Unknown beta decay mode '{mode}'")

    def beta_fn(epoch: int, batch: int) -> float:
        step = epoch * steps_per_epoch + batch

        if step < warmup_steps:
            return max_value
        
        current_step = step - warmup_steps
        progress = min(current_step / max(1, decay_steps), 1.0)

        factor = factor_fn(progress)
        
        return min_value + range_scale * factor
    
    return beta_fn


class ModelPhaseTracker:
    def __init__(self):
        self.model_phase = ModelPhase(T='training', V='validation', E='testing')
        self.current_phase = None

    def set_phase(self, phase_name: str):
        if phase_name in self.model_phase._fields:
            self.current_phase = getattr(self.model_phase, phase_name)
        else:
            raise ValueError(f"Invalid phase name: {phase_name}")

    @property
    def is_training(self):
        return self.current_phase == self.model_phase.T

    @property
    def is_validation(self):
        return self.current_phase == self.model_phase.V

    @property
    def is_testing(self):
        return self.current_phase == self.model_phase.E


class BestModelTracker:


    def __init__(self, objective_name="validation_loss", objective_direction='minimize'):
        self.best_value = None
        self.best_epoch = None
        self.best_trial_number = None
        self.best_model_state = None
        self.best_model_path = None
        self.objective_name = objective_name
        if objective_direction not in ["min", "max", "minimize", "maximize"]:
            raise ValueError("objective_direction must be one of ['min', 'max', 'minimize', 'maximize']")
        self.objective_direction = objective_direction
        self._compare = (lambda a, b: a < b) if self.objective_direction in ["min", "minimize"] else (lambda a, b: a > b)

    def update_best_model(self, objective, epoch, model, trial_number=None):
        if self._is_better(objective):
            self.best_value = objective
            self.best_epoch = epoch
            self.best_trial_number = trial_number
            self.best_model_state = model.state_dict()

    def save_best_model(self, output_folder):
        model_filename = f"best_model_state_{self.best_epoch}"
        self.best_model_path = self.save_model_state(self.best_model_state, output_folder, model_filename)
        return self.best_model_path

    def load_best_model(self, model):
        if self.best_model_state is None:
            get_logger().error(f"Best model state is not initialized, call BestModelTracker.update_best_model(..) to save model states")
        if not isinstance(model, NN_MODULE):
            get_logger().error(f"Cannot load best model state to ({type(model)}), please ensure model type (NN_MODULE)")
        model.load_state_dict(self.best_model_state)
        return model
    
    def track_best_model(self):
        get_logger().info(f"Best {self.objective_name} is {self.best_value} at "
                    f"trial {self.best_trial_number} and epoch {self.best_epoch}")
        
        model_info={
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'best_trial_number': self.best_trial_number,
            'best_model_path': self.best_model_path,
            'objective_name': self.objective_name,
            'objective_direction': self.objective_direction
        }

        return model_info
    
    def _is_better(self, new_value: float) -> bool:
        return self.best_value is None or self._compare(new_value, self.best_value)
        
    @staticmethod
    def save_model_state(model_state, folder_name="model_results/checkpoints", file_name="model_state.pt"):
        if isinstance(model_state, NN_MODULE):
            model_state = model_state.state_dict()
        if not isinstance(model_state, dict):
            get_logger().error(f"Expected torch model or model state dict, received {type(model_state)}")
        
        folder_path = create_folder(folder_name)
        file_path = os_path.join(folder_path, f"{file_name}.pt")

        get_logger().info(f"Saving model state to: {file_path}")
        pt.save(model_state, file_path)
        return file_path

    @staticmethod
    def load_model_state(model, model_path):
        get_logger().info(f"Loading model state from: {model_path}")
        model_state = pt.load(model_path, weights_only=False)
        model.load_state_dict(model_state)
        return model