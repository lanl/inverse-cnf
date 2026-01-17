# Inverse Machine Learning Model Using Conditional Normalizing Flow (CNF)
Code related to inverse CNF for generating initial conditions for scientific simulations.

## Notice of Copyright Assertion (O5042) 
This program is Open-Source under the BSD-3 License.
 
* Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
 
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
 
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
 
Neither the name of the copyright holder nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


## Table of Contents

<details>
  <summary>SHOW TABLE OF CONTENTS</summary>

- [Inverse Machine Learning Model Using Conditional Normalizing Flow (CNF)](#inverse-machine-learning-model-using-conditional-normalizing-flow-cnf)
  - [Notice of Copyright Assertion (O5042)](#notice-of-copyright-assertion-o5042)
  - [Table of Contents](#table-of-contents)
  - [Project code for `DOI:10.1145/3731599.3767343`](#project-code-for-doi10114537315993767343)
    - [Official project website](#official-project-website)
    - [Contributing and issues](#contributing-and-issues)
    - [Demonstration notes](#demonstration-notes)
    - [Limitation notes](#limitation-notes)
  - [Step 1: Project Setup](#step-1-project-setup)
    - [A. Prerequisites](#a-prerequisites)
    - [B. Environment](#b-environment)
  - [Step 2: Create/download raw datasets](#step-2-createdownload-raw-datasets)
    - [A. Simple Simulation](#a-simple-simulation)
    - [B. Electrostatic Potential](#b-electrostatic-potential)
    - [C. Heat Diffusion](#c-heat-diffusion)
    - [D. External Simulation](#d-external-simulation)
  - [Step 3: Preprocess the raw datasets](#step-3-preprocess-the-raw-datasets)
    - [Program File: programs/tasks/preprocess\_dataset.py](#program-file-programstaskspreprocess_datasetpy)
    - [Argument Options Table](#argument-options-table)
    - [A. Simple Simulation](#a-simple-simulation-1)
    - [B. Electrostatic Potential](#b-electrostatic-potential-1)
    - [C. Heat Diffusion](#c-heat-diffusion-1)
    - [D. External Simulation](#d-external-simulation-1)
    - [Notes](#notes)
  - [Step 4: Train or tune the model](#step-4-train-or-tune-the-model)
    - [Program File: programs/tasks/train\_evaluate.py](#program-file-programstaskstrain_evaluatepy)
    - [Argument Options Table](#argument-options-table-1)
    - [A. Simple Simulation](#a-simple-simulation-2)
      - [Single Training Experiment](#single-training-experiment)
      - [Hyper-parameter Tuning Experiment](#hyper-parameter-tuning-experiment)
    - [B. Electrostatic Potential](#b-electrostatic-potential-2)
      - [Single Training Experiment](#single-training-experiment-1)
      - [Hyper-parameter Tuning Experiment](#hyper-parameter-tuning-experiment-1)
    - [C. Heat Diffusion](#c-heat-diffusion-2)
      - [Single Training Experiment](#single-training-experiment-2)
      - [Hyper-parameter Tuning Experiment](#hyper-parameter-tuning-experiment-2)
    - [D. External Simulation](#d-external-simulation-2)
      - [Single Training Experiment](#single-training-experiment-3)
      - [Hyper-parameter Tuning Experiment](#hyper-parameter-tuning-experiment-3)
    - [Notes](#notes-1)
  - [(Optional) Step 5: Generate samples from saved model state](#optional-step-5-generate-samples-from-saved-model-state)
    - [Program File: programs/tasks/generate\_samples.py](#program-file-programstasksgenerate_samplespy)
    - [Argument Options Table](#argument-options-table-2)
    - [A. Simple Simulation](#a-simple-simulation-3)
    - [B. Electrostatic Potential](#b-electrostatic-potential-3)
    - [C. Heat Diffusion](#c-heat-diffusion-3)
    - [D. External Simulations](#d-external-simulations)
    - [Notes](#notes-2)
  - [Step 6: Plot model performance analysis](#step-6-plot-model-performance-analysis)
    - [Program File: programs/tasks/analyze\_model.py](#program-file-programstasksanalyze_modelpy)
    - [Argument Options Table](#argument-options-table-3)
    - [A. Simple Simulation](#a-simple-simulation-4)
    - [B. Electrostatic Potential](#b-electrostatic-potential-4)
    - [C. Heat Diffusion](#c-heat-diffusion-4)
    - [D. External Simulations](#d-external-simulations-1)
    - [Notes](#notes-3)
  - [Step 7: Plot generated sample analysis](#step-7-plot-generated-sample-analysis)
    - [Program File: programs/tasks/analyze\_samples.py](#program-file-programstasksanalyze_samplespy)
    - [Argument Options Table](#argument-options-table-4)
    - [A. Simple Simulation](#a-simple-simulation-5)
    - [B. Electrostatic Potential](#b-electrostatic-potential-5)
    - [C. Heat Diffusion](#c-heat-diffusion-5)
    - [D. External Simulations](#d-external-simulations-2)
    - [Notes](#notes-4)

</details>

## Project code for `DOI:10.1145/3731599.3767343`

### Official project website

> **[https://lezahlie.github.io/inverse-cnf.github.io/](https://lezahlie.github.io/inverse-cnf.github.io/)**
> - The project website contains all supplementary materials and related content for [DOI:10.1145/3731599.3767343](https://doi.org/10.1145/3731599.3767343) 


### Contributing and issues

> Please open an issue ticket with the appropriate tag For any bugs, enhancements, questions, etc. For contributions, please submit a PR request for review. 
> 
> *Note: This project is the preliminary work used for the experiments demonstrated in [DOI:10.1145/3731599.3767343](https://doi.org/10.1145/3731599.3767343). The code has been extended to work for datasets from external simulation codes, but has not been fully optimized in terms of resource usage and runtimes.*


### Demonstration notes

<details>
  <summary>SHOW NOTES</summary>

> The following readme provides a walkthrough that serves three goals:
> 
> 1. Provides a detailed end-to-end demonstration on running the full experimental workflow
>       - Creating/downloading provided datasets
>       - Dataset preprocessing 
>       - Run tuning and final training experiments
>       - Visual analysis of model performance
>       - Generate initial conditions during inference
>       - Visual analysis of generated results
> 2. The provided datasets, configs, and commands are the exact instructions for replicating the experiments from the paper (`DOI:10.1145/3731599.376734`)
>    - Please note that results may vary, as the code is fully deterministic only on the same hardware
> 3. Enough examples to run your own experiments with datasets from the provided or external simulation codes

</details>


### Limitation notes

<details>
  <summary>SHOW NOTES</summary>

> 1. The project only works on Unix/Linux with Python 3.10+
> 2. The model requires all inputs $X$ and targets $Y$ to be square `2D` images with $H=W \ge 3$
> 3. Preprocessing enforces a shared image size:
>    - Let $L_{\mathrm{max}}=\max(3,\,H)$ over all `2D` images in the record
>    - All `2D` images must have shape $(L_{\mathrm{max}}, L_{\mathrm{max}})$
>    - All `0D` scalars are broadcasted to $(L_{\mathrm{max}}, L_{\mathrm{max}})$ (unbroadcasting results is automated within the pipeline)
>    - `1D` vectors are not supported, but can be separately converted to `2D` images by the user
> 4. External simulation datasets:
>    - Format:
>      - Must be `HDF5`
>      - Keys may be top-level or nested
>      - All records must have the same key names and structure
>      - Duplicate keys within a record must be nested under unique parent groups
>    - Shapes:
>      - $X$: one or more `0D` scalars and/or `2D` images
>      - $Y$: one required `2D` image (the primary inference target), optionally followed by extra conditioning channels used only during training (`0D` scalars and/or `2D` images)
>     - Testing (generation) phase in [Step 4](#step-4-train-or-tune-the-model) and [Step 5](#optional-step-5-generate-samples-from-saved-model-state):
>       - Use `--solver-key bypass`
>       - The model generates $N$ candidate inputs $X_i$ conditioned on $Y_0$
>       - You must run your simulator on each $X_i$ to obtain $Y_i$ and compare against $Y_0$
>     -  Visual analysis in [Step 6](#step-6-plot-model-performance-analysis) and [Step 7](#step-7-plot-generated-sample-analysis):
>           - Metrics that require solved $Y_i$ are not plotted when using `bypass`
>           - Training/validation metrics are still plotted

</details>

## Step 1: Project Setup


### A. Prerequisites

<details>
  <summary>SHOW PREREQUISITES</summary>

1. **Required**: POSIX compatible shell in a Linux/Unix-based environment
2. **Required**: Miniconda OR Anaconda installation that supports Python version 3.10
3. **Required**: Clone the project repo with submodules
    ```bash
    git clone git@github.com:lanl/inverse-cnf.git
    cd inverse-cnf
    ```

    > Note: The above commands use `SSH` tokens for access authentication. For `HTTPS` authentication, replace the url with `https://github.com/lanl/inverse-cnf.git`

4. **Required**: Determine which environment your host architecture supports
  
    A. [nvidia_environment.yaml](./nvidia_environment.yaml): PyTorch `Stable` release built with `CUDA` support for Nvidia GPUs
      - Conda will attempt to install compatible versions for `pytorch`, `torchvision`, `pytorch-cuda`, and `cuda-toolkit`
      - Please check [NVIDIA Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) before specifying module versions in the environment file
      - If the environment file fails, try to manually install instead:
  
        ```bash
        conda config --set channel_priority strict
        conda create -n inv-cnf -c conda-forge python=3.10 -y
        conda activate inv-cnf

        conda install -c conda-forge numpy pandas scipy scikit-learn scikit-image \
            h5py matplotlib jsonschema seaborn optuna torchmetrics -y

        conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision \
            pytorch-cuda cuda-toolkit -y
        ```

    B. [mps_environment.yaml](./mps_environment.yaml): PyTorch `Stable` release including Apple Silicon  `MPS` support (macOS 12.3+)
      - For Apple Silicon `MPS` support, you must have macOS 12.3 or greater
      - Otherwise pytorch will default to a `CPU-ONLY` build
      - If the environment file fails, try to manually install instead:

        ```bash
        conda config --set channel_priority strict
        conda create -n inv-cnf -c conda-forge python=3.10 -y
        conda activate inv-cnf

        conda install -c conda-forge numpy pandas scipy scikit-learn scikit-image \
            h5py matplotlib jsonschema seaborn plotly optuna torchmetrics \
            pytorch torchvision -y

        ```

      - Test if MPS support is found/enabled:
        ```bash
        python - <<'PY'
        import torch
        print("torch", torch.__version__)
        print("MPS built:", torch.backends.mps.is_built())
        print("MPS available:", torch.backends.mps.is_available())
        PY
        # torch 2.7.1
        # MPS built: True
        # MPS available: True

        ```
    C. [cpu_environment.yaml](./cpu_environment.yaml): PyTorch `Stable` release with `CPU-ONLY` build
      - Use this environment if you do not need `CUDA` or `MPS` support
      - If the environment file fails, try to manually install instead:
        ```bash
        conda config --set channel_priority strict
        conda create -n inv-cnf -c conda-forge python=3.10 -y
        conda activate inv-cnf

        conda install -c conda-forge numpy pandas scipy scikit-learn scikit-image \
            h5py matplotlib jsonschema seaborn optuna torchmetrics -y

        conda install -c pytorch -c conda-forge pytorch torchvision cpuonly -yo
        ```

    D. [dev_environment.yaml](./dev_environment.yaml): Development environment provided for reference and debugging
      - Known compatible architectures:
          - CPU: `x86-64 (Intel)`; `x86-64 (AMD)`;
          - GPU: `Ampere 8.0 (NVIDIA)`; `Ada Lovelace 8.9 (NVIDIA)`;
      - If the environment file fails, try to manually install instead:
  
        ```bash
        conda config --set channel_priority strict
        conda create -n inv-cnf -c conda-forge python=3.10 -y
        conda activate inv-cnf

        conda install -c conda-forge numpy pandas scipy scikit-learn scikit-image \
            h5py matplotlib jsonschema seaborn optuna torchmetrics -y

        conda install -c pytorch -c nvidia -c conda-forge pytorch=2.4.1 \
            torchvision=0.19.1 pytorch-cuda=11.8 cuda-toolkit=11.8 -y
        ```

</details>

### B. Environment

<details>
  <summary>SHOW ENVIRONMENT</summary>

0. Make sure conda channel priority is strict

    ```bash
    conda config --set channel_priority strict
    ```

    > Optional commands for faster conda solver:
    > ```bash
    > conda install -n base conda-libmamba-solver
    > conda config --set solver libmamba
    > ```

1. Create new conda env with the environment file from the previous step
   
    ```bash
    conda env create -f <environment.yaml>
    ```

2. Activate the project conda environment and setup environment variables
    ```bash
    conda activate inv-cnf
    ```


3. Update project paths in [setup_project_paths.sh](./setup_project_paths.sh)

    - Update the paths at the top of the file
      - `PROJECT_PATH` is the absolute path to the project root directory
      - `OUTPUT_PATH` is the absolute path to your desired output directory

        ```bash
        nano +8 setup_project_paths.sh
        sed -n '8, 10 p' setup_project_paths.sh
        ```
        > Snippet of paths in script
        > ```bash
        > INVERSE_CNF_PROJECT_DIR="/absolute/path/to/inverse-cnf"
        > 
        > INVERSE_CNF_OUTPUT_DIR="/absolute/path/to/output_directory"
        > ```


4. Source the updated script [setup_project_paths.sh](./setup_project_paths.sh)

    ```bash
    . setup_project_paths.sh
    ```
    > **Important notes**:
    > - Be sure to include the leading `.` to *source* the script for changes to persist in the current shell
    > - Do not run this script outside of the `inv-cnf` environment to prevent changes to the parent shell
    > - The script must be re-sourced each time the `inv-cnf` environment is activated 

    - Example of expected output on success

        ```
        ----------------------------------------------------------------------------------------------------
        Exporting project paths as environment variables

        Exported: INVERSE_CNF_PROJECT_DIR=/absolute/path/to/inverse-cnf
        Exported: INVERSE_CNF_OUTPUT_DIR=/absolute/path/to/output_directory
        ----------------------------------------------------------------------------------------------------
        Checking if project paths are present in PYTHONPATH

        Appended: /absolute/path/to/inverse-cnf
        Appended: /absolute/path/to/inverse-cnf/programs
        Appended: /absolute/path/to/inverse-cnf/simulators

        PYTHONPATH: :/absolute/path/to/inverse-cnf:/absolute/path/to/inverse-cnf/programs:/absolute/path/to/inverse-cnf/simulators:
        ----------------------------------------------------------------------------------------------------
        Checking if project output folders exist

        Created: /absolute/path/to/output_directory/datasets
        Created: /absolute/path/to/output_directory/subsets
        Created: /absolute/path/to/output_directory/databases
        Created: /absolute/path/to/output_directory/experiments
        ```

        
    - For SBATCH scripts, source the script just before the program execution


        ```bash
        #!/bin/sh

        #SBATCH --job-name="example"
        #SBATCH --output="/path/to/logs/example_%j.log"
        #SBATCH --error="/path/to/logs/example_%j.err"
        #SBATCH --partition="<partition_name>"
        #SBATCH <...other commands>

        # Step 1: activate (or source) conda env 'inv-cnf' 
        <...conda commands>

        # Step 2: source the updated setup script 
        . /absolute/path/to/inverse-cnf/setup_project_paths.sh

        # Step 3: run python program command(s) 
        python <program_name>.py [...args]
        ```

</details> 

## Step 2: Create/download raw datasets

### A. Simple Simulation

> Program: [simulators/create_simple_dataset.py](./simulators/create_simple_dataset.py)

```bash
python simulators/create_simple_dataset.py --output-path="${INVERSE_CNF_OUTPUT_DIR}/datasets"
```

### B. Electrostatic Potential


> Submodule: [simulators/esp_simulation](./simulators/esp_simulation)

```bash
python simulators/esp_simulation/create_dataset.py \
--output-path="${INVERSE_CNF_OUTPUT_DIR}/datasets" \
--output-folder="electrostatic_dataset_1k" \
--min-seed=1 \
--max-seed=1000 \
--seed-step=100 \
--ntasks=2 \
--image-size=32 \
--conductive-cell-prob=0.5 \
--conductive-material-range=1,10 \
--max-iterations=2000 \
--save-states="first-20,interval-100"
```

> **Alternative link to download dataset to for recreating [DOI:10.1145/3731599.3767343](https://doi.org/10.1145/3731599.3767343) paper experiments:** 
> - **Dataset download:** [https://oceans11.lanl.gov/electrostaticEquations/](https://oceans11.lanl.gov/electrostaticEquations/)


### C. Heat Diffusion


> Submodule: [simulators/heat_diffusion_simulation](./simulators/heat_diffusion_simulation)

```bash
python simulators/heat_diffusion_simulation/create_dataset.py \
--output-path="${INVERSE_CNF_OUTPUT_DIR}/datasets" \
--output-folder="heat_diffusion_dataset_1k" \
--min-seed 1 \
--max-seed 1000 \
--seed-step 100 \
--ntasks 1 \
--grid-length 32 \
--max-iterations 5000 \
--boundary-condition "neumann" \
--convergence-tolerance 1e-4 \
--save-states="first-20,interval-100"
```

> **Alternative link to download dataset to for recreating [DOI:10.1145/3731599.3767343](https://doi.org/10.1145/3731599.3767343) paper experiments:** 
> - **Dataset download:** [https://oceans11.lanl.gov/heatEquations/](https://oceans11.lanl.gov/heatEquations/)


### D. External Simulation

> For testing purposes the examples configs and commands will use the dataset created from `A. Simple Simulation`. Please see [Limitation notes](#limitation-notes) for more information on dataset format constraints.*

</details>

## Step 3: Preprocess the raw datasets

### Program File: [programs/tasks/preprocess_dataset.py](programs/tasks/train_evaluate.py)


### Argument Options Table

<details>
<summary> SHOW ARGUMENT OPTIONS </summary> 

| Option                                | Description                                                        | Choices/Types                                                                              |
|---------------------------------------|--------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| `-h, --help`                          | Show help message and exit                                          |                                                                                          |
| `--debug, -d`                         | Enables debug and verbose printing                                  | Flag (presence means `'On'`)                                                                          |
| `--ntasks`                            | Number of tasks (cpu cores) to run in parallel                     | Any integer value                                                                          |
| `--dataset-file`                      | Input `path/to/<input>` dataset file                                  | String (path to dataset file)                                                              |
| `--output-folder`                     | Output `path/to/folder` to save batches to                            | String (path to output folder)                                                             |
| `--batch-size`                        | Batch size for loading dataset into pytorch model                   | Any integer value                                                                          |
| `--batch-shuffle`                     | Enables shuffling training batches during data loading             | Flag (presence means `'On'`)                                                                          |
| `--subset-split`                      | Ratios to split the original dataset into [`Training`, `Validation`, `Testing`]    | List of floats (e.g., `[0.8, 0.1, 0.1]`)                                                  |
| `--random-seed`                       | Random seed for dataset Train-Validate-Test split                  | Any integer value                                                                          |
| `--transform-method`                  | Type of data transformation method                                  | `['minmax', 'standard']`                                                                  |
| `--minmax-range`                      | Target min and max values for [--transform-method] 'minmax'        | Tuple of two floats                                                                         |
| `--model-input-keys`                  | List of dataset keys used as model input features (X)              | List of strings (e.g., `['image_input_a', 'image_input_b']`)                                                 |
| `--model-target-key`                  | Dataset key for the model target (Y)                               | String (e.g., `'image_state_100'`)                                                              |
| `--solver-input-keys`                 | List of dataset keys for physics solver inputs                     | List of strings (e.g., `['meta_total_iterations']`)                                                 |
| `--unique-id-key`                     | Name of the dataset column to use as a unique record identifier    | String (e.g., `'meta_random_seed'`)                                                               |
| `--flatten-nested-keys`                     | Name of the dataset column to use as a unique record identifier    | String (e.g., `'meta_random_seed'`)                                                               |

</details>



### A. Simple Simulation

<details>
  <summary>SHOW COMMANDS</summary>

- Run via JSON config

    ```bash
    python run_task.py -t preprocess_dataset -c configs/simple/preprocess_dataset.json
    ```

- Run via direct command 

    ```bash
    python run_task.py \
    --task "preprocess_dataset" \
    --dataset-file "${INVERSE_CNF_OUTPUT_DIR}/datasets/simple_3x3_1-1000.hdf5" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/subsets/simple_split_702010_batch_25" \
    --model-input-keys "image_a" "image_b" \
    --model-target-keys "image_c" \
    --unique-id-key "meta_random_seed" \
    --transform-method "standard" \
    --subset-split 0.7 0.2 0.1 \
    --batch-size 25 \
    --random-seed 42 \
    --ntasks 1 \
    --flatten-nested-keys \
    --debug
    ```

</details>

### B. Electrostatic Potential

<details>
  <summary>SHOW COMMANDS</summary>

- Run via JSON config

    ```bash
    python run_task.py -t preprocess_dataset -c configs/electrostatic/preprocess_dataset.json
    ```

- Run via direct command 

    ```bash
    python run_task.py \
    --task "preprocess_dataset" \
    --ntasks 1 \
    --random-seed 42 \
    --dataset-file "${INVERSE_CNF_OUTPUT_DIR}/datasets/electrostatic_dataset_1k/electrostatic_poisson_32x32_1-1000.hdf5" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/subsets/electrostatic_split_702010_batch_25" \
    --model-input-keys "image_charge_distribution" "image_permittivity_map" \
    --model-target-keys "image_potential_state_10" \
    --solver-input-keys "image_potential_state_initial" \
    --unique-id-key "meta_random_seed" \
    --transform-method "standard" \
    --subset-split 0.7 0.2 0.1 \
    --batch-size 25 \
    --debug
    ```

</details>

### C. Heat Diffusion

<details>
  <summary>SHOW COMMANDS</summary>

- Run via JSON config

    ```bash
    python run_task.py -t preprocess_dataset -c configs/heat_diffusion/preprocess_dataset.json
    ```

- Run via direct command 


    ```bash
    python run_task.py \
    --task "preprocess_dataset" \
    --ntasks 2 \
    --random-seed 42 \
    --dataset-file "${INVERSE_CNF_OUTPUT_DIR}/datasets/heat_diffusion_dataset_1k/heat_diffusion_32x32_1-1000.hdf5" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/subsets/heat_diffusion_split_801505_batch_25" \
    --model-input-keys "image_temp_state_initial" "image_diffusion_map" \
    --model-target-keys "image_temp_state_final" \
    --solver-input-keys "meta_total_iterations" \
    --unique-id-key "meta_random_seed" \
    --transform-method "standard" \
    --subset-split 0.8 0.15 0.05 \
    --batch-size 25 \
    --debug
    ```

</details>

### D. External Simulation

<details>
  <summary>SHOW COMMANDS</summary>

> *The commands below are for testing purposes and will require modification of the arguments for your experiment. Please see [Limitation notes](#limitation-notes) for more information.*

- Run via JSON config

    ```bash
    python run_task.py -t preprocess_dataset -c configs/external/preprocess_dataset.json
    ```

- Run via direct command 


    ```bash
    python run_task.py \
    --task "preprocess_dataset" \
    --dataset-file "${INVERSE_CNF_OUTPUT_DIR}/datasets/external.hdf5" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/subsets/external_split_702010_batch_25" \
    --model-input-keys "<x1_name>" "<xi_name>" "<xn_name>" \
    --model-target-keys "<y_name>" \
    --unique-id-key "<id_name>" \
    --transform-method "standard" \
    --subset-split 0.7 0.2 0.1 \
    --batch-size 25 \
    --random-seed 42 \
    --ntasks 1 \
    --debug
    ```

</details>


### Notes

> - `--flatten-nested-keys` flattens nested records by prefixing child keys with their parent path (with delimiter `_`)
>   - If enabled, flattened names must be passed into `--model-input-keys`, `--model-target-keys`, and `--unique-id-key`
>   - If disabled, preprocessing searches nested dicts but only keeps the child key name (so collisions are possible)
>   - Any duplicate child keys within a single record must be nested under unique parent groups, and they require `--flatten-nested-keys`


## Step 4: Train or tune the model

### Program File: [programs/tasks/train_evaluate.py](programs/tasks/train_evaluate.py)

### Argument Options Table

<details>
<summary> SHOW ARGUMENT OPTIONS </summary> 

| Option Name                                | Description                                                        | Choices/Types                                                                              |
|---------------------------------------|--------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| `-h, --help`                          | Show help message and exit                                          |                                                                                          |
| `--debug, -d`                         | Enables debug and verbose printing                                  | Flag (presence means `'On'`)                                                                          |
| `--plot-debug-images`                 | Enable plotting random images for debugging                        | Flag (presence means `'On'`)                                                                          |
| `--gpu-device-list`                   | Specify which GPU(s) to use via device ids                                      | e.g., `"0 1"`                                                                             |
| `--gpu-memory-fraction`               | Fraction of GPU memory to allocate per process                     | Any float value                                                                            |
| `--cpu-device-only`                   | Use only CPU (overrides other device options)                       | Flag (presence means `'On'`)                                                                          |
| `--ntasks`                            | Number of tasks (i.e., cpu cores), recommended `$\geq 3$`                   | Any integer value                                                                          |
| `--random-seed`                       | Random RNG seed for reproducibility                                | Any integer value                                                                           |
| `--input-folder`                      | Input directory for results batches                                | String (path to results)                                                                   |
| `--output-folder`                     | Output directory to save results                                   | String (path to output directory)                                                         |
| `--num-epochs`                        | Total number of training epochs                                    | Any positive integer value                                                                 |
| `--model-key`                         | Model key name (Option for adding future models)                                         | `['cfn']`                                                                           |
| `--solver-key`                        | Solver key for physics loss                                        | `['simple', 'electrostatic', 'heat_diffusion', 'bypass']`                                                                          |
| `--enable-tuning`                     | Enable model hyper-parameter tuning with Optuna                     | Flag (presence means `'On'`)                                                                          |
| `--loss-function` (Optuna)                   | Loss function                                                      | `['mse', 'l1', 'smooth-l1', 'ssim', 'ms-ssim', 'tv', 'kl-div', 'nll']` |
| `--hybrid-loss-functions`             | Hybrid loss functions                                              | `['mse', 'l1', 'smooth-l1', 'ssim', 'ms-ssim', 'tv', 'kl-div', 'nll']` |
| `--hybrid-loss-weights`               | Weights for hybrid loss functions                                   | Any float value                                                                            |
| `--multi-scale-kernel`                | Explicit kernel for MS-SSIM Loss                                   | Any odd integer value in range [3, 15]                                                     |
| `--multi-scale-weights`               | Weights per scale for MS-SSIM Loss                                 | Must match number of scales in MS-SSIM kernel                                              |
| `--activation-function` (Optuna)              | Activation function                                                | `['identity', 'relu', 'leaky-relu', 'elu', 'selu', 'tanh', 'sigmoid', 'silu', 'gelu']`              |
| `--projection-method` (Experimental)               | Method for transforming latent Z into condition Y                  | `['linear', 'mlp', 'resnet']`                                                 |
| `--projection-activation` (Experimental)           | Activation function used when `--projection-method` is not set to `linear` | `['identity', 'relu', 'leaky-relu', 'elu', 'selu', 'tanh', 'sigmoid', 'silu', 'gelu']`              |
| `--affine-log-transform` (Experimental)             | Affine coupling log transformation method                           | `['squared_tanh', 'scaled_tanh', 'sigmoid', 'softplus', 'clamp']`                        |
| `--affine-log-bounds` (Experimental)                | Affine coupling log bounds                                         | Any tuple of two float values                                                               |
| `--beta-schedule-mode`               | Mode for β decay over time                                         | `['linear', 'cosine', 'exp', 'constant']`                                                 |
| `--beta-value-range`                  | Allowed range (min, max) for β value during scheduling             | Tuple of two floats                                                                        |
| `--beta-warmup-epochs`                | Number of warmup epochs where β is fixed at 1.0                    | Any positive integer value                                                                 |
| `--beta-schedule-epochs` (Optuna)             | Number of epochs for β decay                                       | Any positive integer value                                                                 |
| `--learn-rate` (Optuna)                      | Learning rate for model optimizer                                  | Any float value                                                                            |
| `--block-networks` (Optuna)                  | Number of network blocks                                           | Any integer value                                                                          |
| `--hidden-layers` (Optuna)                   | Number of network layers                                           | Any integer value                                                                          |
| `--num-neurons` (Optuna)                      | Number of neurons per layer                                        | Any integer value                                                                          |
| `--conv-kernel`                       | Kernel size for convolutional layers                               | Any odd integer value                                                                      |
| `--conv-stride`                       | Stride for convolutional layers                                    | Any positive integer value                                                                 |
| `--checkpoint-frequency`              | Frequency for saving model states                                  | Any positive integer value                                                                 |
| `--validation-frequency`              | Epoch frequency for validation phase                               | Any positive integer value                                                                 |
| `--testing-frequency`                 | Epoch frequency for testing phase                                  | Any positive integer value                                                                 |
| `--testing-samples`                   | Number of samples to generate during testing phase                 | Any positive integer value                                                                 |
| `--generation-samples`                | Number of samples to generate during inference                    | Any positive integer value                                                                  |
| `--generation-noise`                  | Scaling factor for noise added to generated samples                | Any float value                                                                            |
| `--generation-limit`                  | Maximum number of samples to generate at once (prevents OOM errors) | Any positive integer value                                                                |
| `--enable-earlystop`                  | Enable early stopping with Optuna                                  | Flag (presence means `'On'`)                                                               |
| `--pruner-option`                     | Pruner strategy for early stopping                                 | `['median', 'threshold']`                                                                  |
| `--threshold-upper`                   | Upper bound threshold for early stopping                           | Any float value                                                                            |
| `--min-delta`                         | Minimum improvement for early stopping                             | Any positive float value                                                                   |
| `--epoch-patience`                    | Patience for early stopping (epochs)                               | Any positive integer value                                                                 |
| `--trial-patience`                    | Patience for early stopping (trials)                               | Any positive integer value                                                                 |


> - The above options can be used as cmd-line arguments or in a JSON files with leading `--` removed.
> - (Optuna) labeled arguments are supported for hyper-parameter tuning optimization
>   - These arguments can be passed in as single values for training OR as lists for tuning
> - (Experimental) labeled arguments are new features that need further testing
>   - This arguments can be omitted or specified at the risk of training stability

</details>


### A. Simple Simulation


#### Single Training Experiment

<details>
  <summary>SHOW COMMANDS</summary>

- Run via JSON config

    ```bash
    python run_task.py -t "train_evaluate" -c "configs/simple/train_model.json"
    ```

    > **Small training with debugging enabled**:
    > ```bash
    > python run_task.py -t "train_evaluate" -c "configs/simple/debug_model.json"
    > ```

- Run via direct command 

    ```bash
    python run_task.py \
    --task "train_evaluate" \
    --gpu-device-list 0 \
    --gpu-memory-fraction 0.5 \
    --ntasks 2 \
    --random-seed 42 \
    --input-folder "${INVERSE_CNF_OUTPUT_DIR}/subsets/simple_split_702010_batch_25" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/simple_train_model" \
    --model-key "cfn" \
    --solver-key "simple" \
    --num-epochs 2000 \
    --loss-function "smooth-l1" \
    --activation-function "tanh" \
    --beta-schedule-mode "cosine" \
    --beta-value-range 0.0 1.0 \
    --beta-warmup-epochs 0.1 \
    --beta-schedule-epochs 100 \
    --learn-rate 1e-4 \
    --block-networks 6 \
    --hidden-layers 3 \
    --num-neurons 128 \
    --conv-kernel 1 \
    --conv-stride 1 \
    --checkpoint-frequency 0 \
    --validation-frequency 10 \
    --testing-frequency 0 \
    --testing-samples 5 \
    --generation-samples 100 \
    --generation-limit 100 \
    --generation-noise 1.0
    ```

</details>

#### Hyper-parameter Tuning Experiment

<details>
  <summary>SHOW COMMANDS</summary>

> Warning: This tuning experiment has $(2^5) = 32$ Optuna trials

- Run via JSON config

    ```bash
    python run_task.py -t "train_evaluate" -c "configs/simple/tune_model.json"
    ```

- Run via direct command 

    ```bash
    python run_task.py \
    --task "train_evaluate" \
    --gpu-device-list 0 \
    --gpu-memory-fraction 0.5 \
    --ntasks 2 \
    --random-seed 42 \
    --input-folder "${INVERSE_CNF_OUTPUT_DIR}/subsets/simple_split_702010_batch_25" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/simple_tune_model" \
    --model-key "cfn" \
    --solver-key "simple" \
    --enable-tuning \
    --num-epochs 1000 \
    --loss-function "smooth-l1" \
    --activation-function "tanh" \
    --beta-schedule-mode "cosine" \
    --beta-value-range 0.0 1.0 \
    --beta-warmup-epochs 0.1 \
    --beta-schedule-epochs 100 200 \
    --learn-rate 1e-4 1e-5 \
    --block-networks 3 6 \
    --hidden-layers 2 3 \
    --num-neurons 64 128 \
    --conv-kernel 1 \
    --conv-stride 1 \
    --checkpoint-frequency 0 \
    --validation-frequency 10 \
    --testing-frequency 0 \
    --testing-samples 5 \
    --generation-samples 100 \
    --generation-limit 100 \
    --generation-noise 1.0 \
    --pruner-option "median" \
    --epoch-patience 50 \
    --trial-patience 8 \
    --threshold-upper 0.0 \
    --min-delta 20.0
    ```

</details>

### B. Electrostatic Potential


#### Single Training Experiment

<details>
  <summary>SHOW COMMANDS</summary>

- Run via JSON config

    ```bash
    python run_task.py -t "train_evaluate" -c "configs/electrostatic/train_model.json"
    ```

    > **Small training with debugging enabled**:
    > ```bash
    > python run_task.py -t train_evaluate -c configs/electrostatic/debug_model.json
    > ```


- Run via direct command 


    ```bash
    python run_task.py \
    --task "train_evaluate" \
    --gpu-device-list 0 \
    --gpu-memory-fraction 1.0 \
    --ntasks 2 \
    --random-seed 42 \
    --input-folder "${INVERSE_CNF_OUTPUT_DIR}/subsets/electrostatic_split_702010_batch_25" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/electrostatic_train_model" \
    --model-key "cfn" \
    --solver-key "electrostatic" \
    --num-epochs 3000 \
    --loss-function "smooth-l1" \
    --activation-function "tanh" \
    --beta-schedule-mode "cosine" \
    --beta-value-range 0.0 1.0 \
    --beta-warmup-epochs 0.1 \
    --beta-schedule-epochs 200 \
    --learn-rate 1e-4 \
    --block-networks 9 \
    --hidden-layers 4 \
    --num-neurons 128 \
    --conv-kernel 3 \
    --conv-stride 1 \
    --checkpoint-frequency 0 \
    --validation-frequency 10 \
    --testing-frequency 0 \
    --testing-samples 5 \
    --generation-samples 100 \
    --generation-limit 25 \
    --generation-noise 1.0
    ```

</details>

#### Hyper-parameter Tuning Experiment

<details>
  <summary>SHOW COMMANDS</summary>

> Warning: This tuning experiment has $(2^3 + 3^2) = 72$ Optuna trials

- Run via JSON config

    ```bash
    python run_task.py -t "train_evaluate" -c "configs/electrostatic/tune_model.json"
    ```

- Run via direct command 


    ```bash
    python run_task.py \
    --task "train_evaluate" \
    --gpu-device-list 0 \
    --gpu-memory-fraction 1.0 \
    --ntasks 2 \
    --random-seed 42 \
    --input-folder "${INVERSE_CNF_OUTPUT_DIR}/subsets/electrostatic_split_702010_batch_25" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/electrostatic_tune_model" \
    --model-key "cfn" \
    --solver-key "electrostatic" \
    --enable-tuning \
    --num-epochs 1500 \
    --loss-function "smooth-l1" \
    --activation-function "tanh" \
    --beta-schedule-mode "cosine" \
    --beta-value-range 0.0 1.0 \
    --beta-warmup-epochs 0.1 \
    --beta-schedule-epochs 200 400 \
    --learn-rate 1e-3 1e-4 1e-5 \
    --block-networks 6 9 \
    --hidden-layers 3 4 \
    --num-neurons 128 256 512 \
    --conv-kernel 3 \
    --conv-stride 1 \
    --checkpoint-frequency 0 \
    --validation-frequency 10 \
    --testing-frequency 0 \
    --testing-samples 5 \
    --generation-samples 100 \
    --generation-limit 25 \
    --generation-noise 1.0 \
    --enable-earlystop \
    --pruner-option "median" \
    --epoch-patience 50 \
    --trial-patience 24 \
    --threshold-upper 0.0 \
    --min-delta 20.0
    ```

</details>

### C. Heat Diffusion

#### Single Training Experiment

<details>
  <summary>SHOW COMMANDS</summary>

- Run via JSON config

    ```bash
    python run_task.py -t "train_evaluate" -c "configs/heat_diffusion/train_model.json"
    ```

    > **Small training with debugging enabled**:
    > ```bash
    > python run_task.py -t train_evaluate -c configs/heat_diffusion/debug_model.json
    > ```

- Run via direct command 

    ```bash
    python run_task.py \
    --task "train_evaluate" \
    --gpu-device-list 0 \
    --gpu-memory-fraction 1.0 \
    --ntasks 2 \
    --random-seed 42 \
    --input-folder "${INVERSE_CNF_OUTPUT_DIR}/subsets/electrostatic_split_702010_batch_25" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/electrostatic_train_model" \
    --model-key "cfn" \
    --solver-key "electrostatic" \
    --num-epochs 3000 \
    --loss-function "smooth-l1" \
    --activation-function "tanh" \
    --beta-schedule-mode "cosine" \
    --beta-value-range 0.0 1.0 \
    --beta-warmup-epochs 0.1 \
    --beta-schedule-epochs 200 \
    --learn-rate 1e-4 \
    --block-networks 9 \
    --hidden-layers 4 \
    --num-neurons 128 \
    --conv-kernel 3 \
    --conv-stride 1 \
    --checkpoint-frequency 0 \
    --validation-frequency 10 \
    --testing-frequency 0 \
    --testing-samples 5 \
    --generation-samples 100 \
    --generation-limit 25 \
    --generation-noise 1.0
    ```

</details>

#### Hyper-parameter Tuning Experiment

<details>
  <summary>SHOW COMMANDS</summary>

> Warning: This tuning experiment has $(2^3 + 3^2) = 72$ Optuna trials

- Run via JSON config

    ```bash
    python run_task.py -t "train_evaluate" -c "configs/heat_diffusion/tune_model.json"
    ```

- Run via direct command 

    ```bash
    python run_task.py \
    --task "train_evaluate" \
    --gpu-device-list 0 \
    --gpu-memory-fraction 1.0 \
    --ntasks 2 \
    --random-seed 42 \
    --input-folder "${INVERSE_CNF_OUTPUT_DIR}/subsets/heat_diffusion_split_801505_batch_25" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/heat_diffusion_tune_model" \
    --model-key "cfn" \
    --solver-key "heat_diffusion" \
    --num-epochs 2000 \
    --enable-tuning \
    --loss-function "smooth-l1" \
    --activation-function "tanh" \
    --beta-schedule-mode "cosine" \
    --beta-value-range 0.0 1.0 \
    --beta-warmup-epochs 0.1 \
    --beta-schedule-epochs 250 500 \
    --learn-rate 1e-3 1e-4 1e-5 \
    --block-networks 6 9 \
    --hidden-layers 3 4 \
    --num-neurons 128 256 512 \
    --conv-kernel 3 \
    --conv-stride 1 \
    --checkpoint-frequency 0 \
    --validation-frequency 10 \
    --testing-frequency 0 \
    --testing-samples 5 \
    --generation-samples 100 \
    --generation-limit 25 \
    --generation-noise 1.0 \
    --enable-earlystop \
    --pruner-option "median" \
    --epoch-patience 50 \
    --trial-patience 24 \
    --threshold-upper 0.0 \
    --min-delta 20.0
    ```

</details>

### D. External Simulation

> *The commands below are for testing purposes and will require modification of the arguments for your experiment. Please see [Limitation notes](#limitation-notes) for more information.*

#### Single Training Experiment

<details>
  <summary>SHOW COMMANDS</summary>

- Run via JSON config

    ```bash
    python run_task.py -t "train_evaluate" -c "configs/external/train_model.json"
    ```

    > **Small training with debugging enabled**:
    > ```bash
    > python run_task.py -t "train_evaluate" -c "configs/external/debug_model.json"
    > ```

- Run via direct command 

    ```bash
    python run_task.py \
    --task "train_evaluate" \
    --gpu-device-list 0 \
    --gpu-memory-fraction 0.5 \
    --ntasks 2 \
    --random-seed 42 \
    --input-folder "${INVERSE_CNF_OUTPUT_DIR}/subsets/external_split_702010_batch_25" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/external_train_model" \
    --model-key "cfn" \
    --solver-key "bypass" \
    --num-epochs 2000 \
    --loss-function "smooth-l1" \
    --activation-function "tanh" \
    --beta-schedule-mode "cosine" \
    --beta-value-range 0.0 1.0 \
    --beta-warmup-epochs 0.1 \
    --beta-schedule-epochs 100 \
    --learn-rate 1e-4 \
    --block-networks 6 \
    --hidden-layers 3 \
    --num-neurons 128 \
    --conv-kernel 1 \
    --conv-stride 1 \
    --checkpoint-frequency 0 \
    --validation-frequency 10 \
    --testing-frequency 0 \
    --testing-samples 5 \
    --generation-samples 100 \
    --generation-limit 100 \
    --generation-noise 1.0
    ```

</details>

#### Hyper-parameter Tuning Experiment

<details>
  <summary>SHOW COMMANDS</summary>

> Warning: This tuning experiment has $(2^5) = 32$ Optuna trials

- Run via JSON config

    ```bash
    python run_task.py -t "train_evaluate" -c "configs/external/tune_model.json"
    ```

- Run via direct command 


    ```bash
    python run_task.py \
    --task "train_evaluate" \
    --gpu-device-list 0 \
    --gpu-memory-fraction 0.5 \
    --ntasks 2 \
    --random-seed 42 \
    --input-folder "${INVERSE_CNF_OUTPUT_DIR}/subsets/external_split_702010_batch_25" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/external_tune_model" \
    --model-key "cfn" \
    --solver-key "bypass" \
    --enable-tuning \
    --num-epochs 1000 \
    --loss-function "smooth-l1" \
    --activation-function "tanh" \
    --beta-schedule-mode "cosine" \
    --beta-value-range 0.0 1.0 \
    --beta-warmup-epochs 0.1 \
    --beta-schedule-epochs 100 200 \
    --learn-rate 1e-4 1e-5 \
    --block-networks 3 6 \
    --hidden-layers 2 3 \
    --num-neurons 64 128 \
    --conv-kernel 1 \
    --conv-stride 1 \
    --checkpoint-frequency 0 \
    --validation-frequency 10 \
    --testing-frequency 0 \
    --testing-samples 5 \
    --generation-samples 100 \
    --generation-limit 100 \
    --generation-noise 1.0 \
    --pruner-option "median" \
    --epoch-patience 50 \
    --trial-patience 8 \
    --threshold-upper 0.0 \
    --min-delta 20.0
    ```

</details>


### Notes

> - It is worth while checking the JSON configs to update common parameters
>   - For example `"ntasks": 2` sets the # cores to `2`, but at least `3` is recommended
>   - The `--generation-limit` may be increased or decreased based on memory usage
>   - If not using `CUDA` acceleration, enable `--cpu-device-only` flag to override gpu related options
>   - Replace exported var `${INVERSE_CNF_OUTPUT_DIR}` in paths to read/write to other storage locations
> - Tuning configs and commands are examples to demonstrate defining hyper-parameters arguments lists
>   - Please be aware that some examples possible combinations and could result in too MANY trials
>   - In practice, it simpler and faster to split combinations across two or more experiments
> - The `--solver-key` argument must be set to `bypass` for training external simulation datasets
> - A final testing phase is always run with the best model state (based on the epoch with the lowest validation loss) regardless of `--testing-frequency`



## (Optional) Step 5: Generate samples from saved model state


### Program File: [programs/tasks/generate_samples.py](programs/tasks/generate_samples.py)

### Argument Options Table

<details>
<summary> SHOW ARGUMENT OPTIONS </summary> 

| Option                                | Description                                                        | Choices/Types                                                                              |
|---------------------------------------|--------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| `-h, --help`                          | Show help message and exit                                          |                                                                                          |
| `--debug, -d`                         | Enables debug and verbose printing                                  | Flag (presence means `'On'`)                                                                          |
| `--gpu-device-list`                   | Specify which GPU(s) to use                                         | e.g., `"0 1"`                                                                             |
| `--gpu-memory-fraction`               | Fraction of GPU memory to allocate per process                     | Any float value                                                                            |
| `--cpu-device-only`                   | Use only CPU (overrides other device options)                       | Flag (presence means `'On'`)                                                                          |
| `--ntasks`                            | Number of tasks (cpu cores) to run in parallel                     | Any integer value                                                                          |
| `--random-seed`                       | Random RNG seed for reproducing sample generation                  | Any integer value                                                                          |
| `--generation-samples`                | Number of samples to generate during inference                      | Any positive integer value                                                                 |
| `--generation-limit`                  | Maximum number of samples to generate at once (prevents OOM errors) | Any positive integer value                                                                |
| `--generation-noise`                  | Scaling factor for noise added to generated samples                | Any float value                                                                            |
| `--input-folder`                      | Input `path/to/directory` where results batches are saved            | String (path to input folder)                                                              |
| `--output-folder`                     | Output `path/to/directory` to save results                           | String (path to output folder)                                                             |
| `--model-state-path`                  | Input file path to the saved model state file                       | String (path to model state file)                                                          |
| `--model-params-path`                 | Input file path to where model params are saved to                 | String (path to model params file)                                                        |

</details>


### A. Simple Simulation

<details>
  <summary>SHOW COMMANDS</summary>

- Run via JSON config

    ```bash
    python run_task.py -t "generate_samples" -c "configs/simple/generate_samples.json"
    ```

- Run via direct command 
    
    ```bash
    python run_task.py \
    --task "generate_samples" \
    --ntasks 2 \
    --random-seed 42 \
    --gpu-device-list 0 \
    --gpu-memory-fraction 1.0 \
    --input-folder "${INVERSE_CNF_OUTPUT_DIR}/subsets/simple_split_702010_batch_25" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/simple_train_model/best_model_state" \
    --model-state-path "${INVERSE_CNF_OUTPUT_DIR}/experiments/simple_train_model/checkpoints/best_model_state_*.pt" \
    --model-params-path "${INVERSE_CNF_OUTPUT_DIR}/experiments/simple_train_model/parameters/model_params.json" \
    --generation-samples 1000 \
    --generation-limit 100 \
    --generation-noise 1.0
    ```

</details>

### B. Electrostatic Potential

<details>
  <summary>SHOW COMMANDS</summary>

- Run via JSON config

    ```bash
    python run_task.py -t "generate_samples" -c "configs/electrostatic/generate_samples.json"
    ```

- Run via direct command 


    ```bash
    python run_task.py \
    --task "generate_samples" \
    --gpu-device-list 0 \
    --gpu-memory-fraction 1.0 \
    --ntasks 2 \
    --random-seed 42 \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/subsets/electrostatic_split_702010_batch_25" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/electrostatic_train_model/best_model_state" \
    --model-state-path "${INVERSE_CNF_OUTPUT_DIR}/experiments/electrostatic_train_model/checkpoints/best_model_state_*.pt" \
    --model-params-path "${INVERSE_CNF_OUTPUT_DIR}/experiments/electrostatic_train_model/parameters/model_params.json"
    --generation-samples 1000 \
    --generation-limit 25 \
    --generation-noise 1.0
    ```

</details>

### C. Heat Diffusion

<details>
  <summary>SHOW COMMANDS</summary>

- Run via JSON config
  
    ```bash
    python run_task.py -t "generate_samples" -c "configs/heat_diffusion/generate_samples.json"
    ```

- Run via direct command 


    ```bash
    python run_task.py \
    --task "generate_samples" \
    --gpu-device-list 0 \
    --gpu-memory-fraction 1.0 \
    --ntasks 2 \
    --random-seed 42 \
    --input-folder "${INVERSE_CNF_OUTPUT_DIR}/subsets/heat_diffusion_split_801505_batch_25" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/heat_diffusion_train_model/best_model_state" \
    --model-state-path "${INVERSE_CNF_OUTPUT_DIR}/experiments/heat_diffusion_train_model/checkpoints/best_model_state_*.pt" \
    --model-params-path "${INVERSE_CNF_OUTPUT_DIR}/experiments/heat_diffusion_train_model/parameters/model_params.json"
    --generation-samples 1000 \
    --generation-limit 25 \
    --generation-noise 1.0
    ```

</details>

### D. External Simulations

<details>
  <summary>SHOW COMMANDS</summary>

> *The commands below are for testing purposes and will require modification of the arguments for your experiment. Please see [Limitation notes](#limitation-notes) for more information.*

- Run via JSON config

    ```bash
    python run_task.py -t "generate_samples" -c "configs/external/generate_samples.json"
    ```

- Run via direct command 


    ```bash
    python run_task.py \
    --task "generate_samples" \
    --ntasks 2 \
    --random-seed 42 \
    --gpu-device-list 0 \
    --gpu-memory-fraction 1.0 \
    --input-folder "${INVERSE_CNF_OUTPUT_DIR}/subsets/external_split_702010_batch_25" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/external_train_model/best_model_state" \
    --model-state-path "${INVERSE_CNF_OUTPUT_DIR}/experiments/external_train_model/checkpoints/best_model_state_*.pt" \
    --model-params-path "${INVERSE_CNF_OUTPUT_DIR}/experiments/external_train_model/parameters/model_params.json" \
    --generation-samples 1000 \
    --generation-limit 100 \
    --generation-noise 1.0
    ```

</details>


### Notes

> - Filenames `best_model_state_*.pt` wildcards `*` will resolve to the first matching occurrence
> - This step is optional because training and tuning experiments always run a final testing phase (inference mode) with the best model state (based on the epoch with lowest validation loss)


## Step 6: Plot model performance analysis


### Program File: [programs/tasks/analyze_model.py](programs/tasks/analyze_model.py)

### Argument Options Table
<details>
<summary> SHOW ARGUMENT OPTIONS </summary> 

| Option                                | Description                                                        | Choices/Types                                                                              |
|---------------------------------------|--------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| `-h, --help`                          | Show help message and exit                                          |                                                                                          |
| `--debug, -d`                         | Enables debug and verbose printing                                  | Flag (presence means `'On'`)                                                                          |
| `--output-folder`                     | Output folder to save analysis to                                   | String (path to output folder)                                                             |
| `--input-folder`                      | Input path to folder containing result data files                  | String (path to input folder)                                                              |
| `--max-epoch`                         | Maximum epoch for model performance plots                          | Any positive integer value                                                                 |


</details>


### A. Simple Simulation

<details>
  <summary>SHOW COMMANDS</summary>

- Run via JSON config

    ```bash
    python run_task.py -t "analyze_model" -c "configs/simple/analyze_model.json"
    ```

- Run via direct command 


    ```bash
    python run_task.py \
    --task "analyze_model" \
    --input-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/simple_train_model" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/simple_train_model" \
    --debug
    ```

</details>

### B. Electrostatic Potential

<details>
  <summary>SHOW COMMANDS</summary>

- Run via JSON config

    ```bash
    python run_task.py -t "analyze_model" -c "configs/electrostatic/analyze_model.json"
    ```

- Run via direct command 
    

    ```bash
    python run_task.py \
    --task "analyze_model" \
    --input-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/electrostatic_train_model" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/electrostatic_train_model" \
    --max-epoch 2000 \
    --debug
    ```

</details>

### C. Heat Diffusion

<details>
  <summary>SHOW COMMANDS</summary>

- Run via JSON config

    ```bash
    python run_task.py -t "analyze_model" -c "configs/heat_diffusion/analyze_model.json"
    ```

- Run via direct command 


    ```bash
    python run_task.py \
    --task "analyze_model" \
    --input-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/heat_diffusion_train_model" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/heat_diffusion_train_model" \
    --max-epoch 3000 \
    --debug
    ```

</details>

### D. External Simulations

<details>
  <summary>SHOW COMMANDS</summary>

> *The commands below are for testing purposes and will require modification of the arguments for your experiment. Please see [Limitation notes](#limitation-notes) for more information.*

- Run via JSON config

    ```bash
    python run_task.py -t "analyze_model" -c "configs/external/analyze_model.json"
    ```

- Run via direct command 


    ```bash
    python run_task.py \
    --task "analyze_model" \
    --input-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/external_train_model" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/external_train_model" \
    --debug
    ```

</details>


### Notes

> For tuning experiments, specify the trial subfolder under the experiment folder in `--input-folder` and `--output-folder`
>   - e.g., `--input-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/electrostatic_tune_model/trial_1"`
>   - e.g., `--output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/electrostatic_tune_model/trial_1"`

</details>

## Step 7: Plot generated sample analysis

### Program File: [programs/tasks/analyze_samples.py](programs/tasks/analyze_samples.py)

### Argument Options Table

<details>
<summary> SHOW ARGUMENT OPTIONS </summary> 

| Option                                | Description                                                        | Choices/Types                                                                              |
|---------------------------------------|--------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| `-h, --help`                          | Show help message and exit                                          |                                                                                          |
| `--debug, -d`                         | Enables debug and verbose printing                                  | Flag (presence means `'On'`)                                                                          |
| `--output-folder`                     | Output folder to save analysis to                                   | String (path to output folder)                                                             |
| `--input-data-path`                   | Input file path to the input data file                              | String (path to input data file)                                                           |
| `--random-seed`                       | Random seed for getting random records from the data file          | Any integer value                                                                          |
| `--num-records`                       | Number of records to read from the data file                       | Any positive integer value                                                                 |


</details>

### A. Simple Simulation

<details>
  <summary>SHOW COMMANDS</summary>

- Run via JSON config

    ```bash
    python run_task.py -t "analyze_samples" -c "configs/simple/analyze_samples.json"
    ```

- Run via direct command 

    ```bash
    python run_task.py \
    --task "analyze_samples" \
    --input-data-path "${INVERSE_CNF_OUTPUT_DIR}/experiments/simple_train_model/best_epoch_test/testing_results/result_records/testing_records_*.hdf5" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/simple_train_model/best_epoch_test" \
    --random-seed 42 \
    --num-records 10
    ```

</details>

### B. Electrostatic Potential

<details>
  <summary>SHOW COMMANDS</summary>

- Run via JSON config

    ```bash
    python run_task.py -t "analyze_samples" -c "configs/electrostatic/analyze_samples.json"
    ``` 

- Run via direct command 

    ```bash
    python run_task.py \
    --task "analyze_samples" \
    --input-data-path "${INVERSE_CNF_OUTPUT_DIR}/experiments/electrostatic_train_model/best_epoch_test/testing_results/result_records/testing_records_*.hdf5" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/electrostatic_train_model/best_epoch_test" \
    --random-seed 42 \
    --num-records 10
    ```

</details>

### C. Heat Diffusion

<details>
  <summary>SHOW COMMANDS</summary>

- Run via JSON config
  
    ```bash
    python run_task.py -t "analyze_samples" -c "configs/heat_diffusion/analyze_samples.json"
    ```

- Run via direct command 

    ```bash
    python run_task.py \
    --task "analyze_samples" \
    --input-data-path "${INVERSE_CNF_OUTPUT_DIR}/experiments/heat_diffusion_train_mode/best_epoch_test/testing_results/result_records/testing_records_*.hdf5" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/heat_diffusion_train_model/best_epoch_test" \
    --random-seed 42 \
    --num-records 10
    ```

</details>


### D. External Simulations

<details>
  <summary>SHOW COMMANDS</summary>

> *The commands below are for testing purposes and will require modification of the arguments for your experiment. Please see [Limitation notes](#limitation-notes) for more information.*

- Run via JSON config

    ```bash
    python run_task.py -t "analyze_samples" -c "configs/external/analyze_samples.json"
    ```

- Run via direct command 
    
    ```bash
    python run_task.py \
    --task "analyze_samples" \
    --input-data-path "${INVERSE_CNF_OUTPUT_DIR}/experiments/external_train_model/best_epoch_test/testing_results/result_records/testing_records_*.hdf5" \
    --output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/external_train_model/best_epoch_test" \
    --random-seed 42 \
    --num-records 10
    ```


</details>


### Notes

> - Filenames `testing_records_*.hdf5` wildcards `*` will resolve to the first matching occurrence
> - For tuning experiments, specify the trial subfolder under the experiment folder in `--input-data-path`and `--output-folder`
>   - e.g., `--input-data-path "${INVERSE_CNF_OUTPUT_DIR}/experiments/heat_diffusion_tune_model/trial_2/best_epoch_test/testing_results/result_records/testing_records_*.hdf5"`
>   - e.g., `--output-folder "${INVERSE_CNF_OUTPUT_DIR}/experiments/heat_diffusion_tune_model/trial_2/best_epoch_test"`
