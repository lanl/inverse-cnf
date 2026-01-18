from programs.utils.logger_setup import get_logger
from programs.utils.common import os_path, os_remove, convert_type, as_list, create_folder, get_timestamp, save_to_json, extract_item, partial
from programs.utils.experiment_setup import setup_model_instance
from programs.core.experiment_tasks import train_validate_test
import optuna
from sqlite3 import connect, OperationalError
from time import sleep
from socket import gethostname


class StorageManager:
    def __init__(self, base_db_name="topology_optimization", folder_path="databases"):
        self.folder_path = create_folder(folder_path)
        self.db_filename, self.connection = self._generate_unique_db_name(base_db_name)
        self.storage_url = f"sqlite:///{self.folder_path}/{self.db_filename}"
        self.lock_file_path = os_path.join(self.folder_path, f"{self.db_filename}.lock")
        self._acquire_lock()

    def _generate_unique_db_name(self, base_name):
        hostname_prefix = (gethostname().split('.',1)[0]).lower()
        db_name = f"{base_name}_{hostname_prefix}"
        db_file = f"{db_name}.db"
        db_path = os_path.join(self.folder_path, db_file)
        
        while True:
            lock_file_path = os_path.join(self.folder_path, f"{db_name}.lock")
            # Check if the lock file database exists
            if not os_path.exists(lock_file_path):
                try:
                    conn = connect(db_path)
                    conn.execute("SELECT 1")  
                    get_logger().info(f"Using database file: '{db_path}'")
                    return db_file, conn
                except OperationalError:
                    get_logger().warning(f"Cannot to connect to database '{db_name}'")
            else:
                get_logger().warning(f"Database file '{db_path}' is locked by another process.")
        
            # Generate a new database name with a timestamp
            db_file = f"{db_name}_{get_timestamp()}.db"
            db_path = os_path.join(self.folder_path, db_file)
            continue

    def _acquire_lock(self):
        self.lock_file_path = os_path.join(self.folder_path, f"{self.db_filename}.lock")

        if os_path.exists(self.lock_file_path):
            get_logger().warning(f"Database file '{self.db_filename}' is locked by another process")
            new_db_file = f"{self.db_filename.split('.')[0]}_{get_timestamp()}.db"
            self.db_filename = new_db_file 
            self.storage_url = f"sqlite:///{os_path.join(self.folder_path, self.db_filename)}"
            get_logger().info(f"Created new database file '{self.db_filename}'")
            self._acquire_lock() 
        else:
            open(self.lock_file_path, 'w').close()
            get_logger().debug(f"Created lock file '{self.lock_file_path}' for database file '{self.db_filename}'")

    def _release_lock(self):
        if os_path.exists(self.lock_file_path):
            os_remove(self.lock_file_path)
            get_logger().debug(f"Lock file '{self.lock_file_path}' removed")

    def __del__(self):
        self.close_connection()

    def close_connection(self):
        if self.connection:
            self.connection.close()
            get_logger().info(f"Closed connection to database '{self.storage_url}'")
            self._release_lock()
            get_logger().debug(f"Released lock for database file '{self.db_filename}'")
            self.connection = None


class StudyManager:
    def __init__(self, storage_url):
        self.storage_url = storage_url
        self.study_name = None

    def create_unique_study_name(self, base_name):
        delay=1
        while True:
            study_names = self.get_study_names()
            unique_name = f"{base_name}_{get_timestamp()}"
            if unique_name not in study_names:
                self.study_name = unique_name
                break
            else:
                get_logger().debug(f"Study name '{unique_name}' already exists, regenerating in {delay} seconds")
                sleep(delay)
        return self.study_name

    def get_study_names(self):
        studies = optuna.study.get_all_study_summaries(storage=self.storage_url)
        study_names = [study.study_name for study in studies]
        return study_names

    def remove_study_from_storage(self, study_name):
        studies = self.get_study_names(self.storage_url)
        if study_name in studies:
            get_logger().info(f"Removing study '{study_name}' from '{self.storage_url}'")
            optuna.delete_study(study_name=study_name, storage=self.storage_url)

    def load_study_from_storage(self, study_name, sampler=None, pruner=None):
        studies = self.get_study_names()
        if study_name in studies:
            study = optuna.load_study(study_name=study_name, storage=self.storage_url, sampler=sampler, pruner=pruner)
            return study
        return None

    @staticmethod
    def save_study_info(study, study_config, output_folder):
        best_trial = study.best_trial
        study_folder = create_folder(os_path.join(output_folder, "study_info"))

        study_config_path = os_path.join(study_folder, "study_config.json")
        save_to_json(study_config_path, study_config)
        
        best_trial_path = os_path.join(study_folder, "best_trial_results.json")
        best_trial_info = {
                "trial": best_trial.number,
                "state": convert_type(best_trial.state, str),
                "params": best_trial.params,
                "value": best_trial.value
            }
        save_to_json(best_trial_path, best_trial_info)

    @staticmethod
    def stop_study_callback(study, trial, min_threshold=0.0, max_threshold=None, trial_patience=0, warmup_trials=20):
        curr_trial_number = trial.number
        if curr_trial_number < warmup_trials:
            return
    
        should_stop = False

        if trial_patience > 0:
            best_trial_number = study.best_trial.number
            if (curr_trial_number - best_trial_number) >= trial_patience:
                should_stop = True

        if min_threshold >= 0.0:
            best_trial_value = study.best_trial.value
            if best_trial_value <= min_threshold:
                should_stop = True

        if max_threshold is not None:
            best_trial_value = study.best_trial.value
            if best_trial_value >= max_threshold:
                should_stop = True
        
        if should_stop:
            study.stop()

# bypasses reporting more than one intermediate value to optuna db
class ObjectiveReporter:
    def __init__(self, trial, target_names=None, objective_name=""):
        self.metrics = {}
        self.trial = trial
        self.target_names = target_names
        self.objective_name = objective_name
        self.num_targets = len(self.target_names)
        self.metrics['target_names'] = self.target_names

    def get_metrics(self):
        return self.metrics
    
    def _report_value(self, step, model_phase, target_name, objective_value: float):
        if objective_value is not None:
            self.metrics[step][model_phase][self.objective_name][target_name] = convert_type(objective_value, float)
            unique_key = f'{step}-{model_phase}-{self.objective_name}-{target_name}'
            self.trial.set_user_attr(unique_key, self.metrics[step][model_phase][self.objective_name][target_name])

    def _report_list(self, step, model_phase, objective_list: list):
        if objective_list is not None:
            objective_list = [ convert_type(objective, float) for objective in objective_list]
            self.metrics[step][model_phase][self.objective_name] = dict(zip(self.target_names, objective_list))
            for objective, target in zip(objective_list, self.target_names):
                self.trial.set_user_attr(f'{step}-{model_phase}-{self.objective_name}-{target}', objective)
    
    def report(self, step, model_phase, objective_value, objective_list=None):
        if step not in self.metrics:
            self.metrics[step] = {}
        if model_phase not in self.metrics[step]:
            self.metrics[step][model_phase] = {}
        if self.objective_name not in self.metrics[step][model_phase]:
            self.metrics[step][model_phase][self.objective_name] = {}
        
        target_name=self.target_names[0] if self.num_targets == 1 else "overall"

        if target_name not in self.metrics[step][model_phase][self.objective_name]:
            self.metrics[step][model_phase][self.objective_name][target_name] = {}
        self._report_value(step, model_phase, target_name, objective_value)

        if objective_list is not None:
            self._report_list(step, model_phase, objective_list)

    def save_report(self, output_path):
        output_file =os_path.join(output_path, f"{self.objective_name}_metrics_by_epoch.json")
        save_to_json(output_file, self.metrics)


class DuplicateTrialChecker:
    def __init__(self):
        self.tried_combinations = set()

    def __call__(self, trial):
        trial_params = frozenset(trial.params.items())
        if trial_params in self.tried_combinations:
            get_logger().warning(f"Duplicate trial found: {trial.number} with parameters: {trial_params}")
            return True
        self.tried_combinations.add(trial_params)
        return False


def objective(trial, model_params, hyperparams, batch_loaders, out_folder):
    logger = get_logger()
    dupe_trial_checker = DuplicateTrialChecker()

    dupe_trial = True
    while dupe_trial:
        for param_name, param_list in hyperparams.items():
            model_params[param_name] = trial.suggest_categorical(param_name, param_list)
        dupe_trial = dupe_trial_checker(trial)

    logger.debug(f'hyperparams ={hyperparams}')
    logger.debug(f'model_params ={model_params}')

    out_folder = os_path.join(out_folder, f"trial_{trial.number}")
    model = setup_model_instance(model_params, out_folder)

    target_names = as_list(model_params['data_groups']['target']['names'])
    objective_reporter = ObjectiveReporter(trial, target_names, model_params["loss_function_names"]["validation"])
    result = train_validate_test(model, model_params, batch_loaders, out_folder, objective_reporter=objective_reporter, trial=trial)
    return result


def optimize(model_params, study_params, batch_loaders, out_folder):
    storage_manager = StorageManager()
    study_manager = StudyManager(storage_manager.storage_url)
    study_manager.create_unique_study_name(f"study_{get_timestamp()}")

    study_params['storage_url'] = storage_manager.storage_url
    study_params['study_name'] = study_manager.study_name

    num_epochs = model_params["num_epochs"]
    num_trials = study_params['num_trials']

    patient_pruner, callbacks = None, None
    if study_params['early_stop']:

        min_epochs = max(num_epochs // 4, 20)
        min_trials = max(num_trials // 4, 2)

        pruner_option = study_params.get("pruner_option", "median")
        thresh_upper = study_params.get('threshold_upper', 0.0)
        epoch_patience = study_params.get('epoch_patience', min_epochs)
        trial_patience = study_params.get('trial_patience', min_trials)
        min_delta = study_params.get('min_delta', 100.0)

        if pruner_option == "median":
            base_pruner = optuna.pruners.MedianPruner(n_startup_trials=min_trials, n_warmup_steps=min_epochs)
        elif pruner_option == "threshold":
            base_pruner = optuna.pruners.ThresholdPruner(lower=None, upper=thresh_upper, n_warmup_steps=min_epochs)

        patient_pruner = optuna.pruners.PatientPruner(
            wrapped_pruner=base_pruner,
            patience=epoch_patience,
            min_delta=min_delta
        )

        callbacks = [partial(study_manager.stop_study_callback, 
                            max_threshold=thresh_upper, 
                            trial_patience=trial_patience,
                            warmup_trials=trial_patience)]
        

    study = optuna.create_study(study_name=study_params['study_name'], 
                            direction=study_params['direction'],
                            storage=study_params['storage_url'],
                            sampler=study_params['sampler'],
                            pruner=patient_pruner,
                            load_if_exists=False)


    study.optimize(lambda trial: 
                    objective(
                        trial, 
                        model_params, 
                        study_params['hyperparams'], 
                        batch_loaders, 
                        out_folder
                    ), 
                    n_trials=num_trials,
                    callbacks=callbacks)
    
    study_manager.save_study_info(study, study_params, out_folder)
    storage_manager.close_connection() 
    return study
