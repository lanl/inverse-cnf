from programs.utils.logger_setup import get_logger
from programs.utils.common import pprint, PT_TENSOR, os_path, as_list, convert_type, pt, np, process_stack, create_folder, create_file_path, save_to_json, save_to_hdf5, split_channel_roles, unbroadcast_scalar
from programs.core.model_manager import BestModelTracker, ModelPhaseTracker, EpochFrequency, create_beta_scheduler
from programs.utils.solver_setup import SolverSetup
from programs.core.sample_evaluation import LatentStatistics, ImageContinuousMetrics
from programs.viz.sample_plots import plot_generated_images, plot_generated_sample_scores, plot_reconstructed_images, plot_latent_images


def save_debug_plots(result_records, current_phase, epoch_num, output_path, num_samples=3, data_ranges=None, latent_records=None, random_seed = None):
    num_records = len(result_records)
    rng = np.random.default_rng(seed=random_seed)
    random_indices = rng.choice(num_records, size=max(min(num_records, num_samples), 1), replace=False)
    random_records = [result_records[i] for i in random_indices]

    plot_folder = create_folder(os_path.join(output_path, "debug_images", f"epoch_{epoch_num}"))
    if current_phase == "testing":
        plot_generated_images(random_records, epoch_num, plot_folder)
        plot_generated_sample_scores(random_records, epoch_num, plot_folder, data_ranges)
    else:
        plot_reconstructed_images(random_records, epoch_num, current_phase, plot_folder)
        if latent_records:
            random_latents = [latent_records[i] for i in random_indices]
            plot_latent_images(random_latents, epoch_num, current_phase, plot_folder)



def format_model_loss(name: str, target: str|list[str], summary:dict, partial:dict={}):
    loss_dict = {
        "function": {
            'name': name,
            'target': target
        },
        "summary": summary
    }

    if partial and "z_loss" in partial and "y_loss" in partial:
        z_name, y_block = name.split("_", 1)
        beta_term, y_name = y_block.split("x", 1)

        partial["z_loss"]['name'] = z_name

        if "raw" in partial["y_loss"]:
            partial["y_loss"]["raw"]['name'] = y_name

        if "weighted" in partial["y_loss"]:
            partial["y_loss"]['weighted']['beta_term'] = beta_term
            partial["y_loss"]['weighted']['name'] = y_block

        loss_dict['partial'] = partial

    return {"loss": loss_dict}


def compute_model_scores(scores, is_overall=False):
    score_dict = {}

    if not is_overall:
        # individual target averages (avg of samples or single summary)
        if isinstance(scores, dict) and 'metrics' in scores:
            score_dict['metrics'] = scores['metrics']
        else:
            # list of {'metrics': ...}
            score_dict['metrics'] = {
                key:  convert_type(np.mean([s['metrics'][key] for s in scores]), float)
                for key in scores[0]['metrics'].keys()
            }
    else:
        # is_overall = aggregate across targets
        metric_keys = scores[0].get('metrics', {}).keys()
        score_dict['metrics'] = {}
        for key in metric_keys:
            values = [s.get('metrics', {}).get(key, 0.0) for s in scores]
            score_dict['metrics'][key] = convert_type(np.mean(values), float)

    return score_dict

def save_generated(model_results, model_meta, current_phase, model_losses=None, bypass_solver=False, epoch_num=None, output_folder="model_results", debug_plots=False):
    output_path = create_folder(os_path.join(output_folder, f"{current_phase}_results"))

    get_logger().debug(f"phase_tracker.current_phase: {current_phase}")
    
    input_group = model_meta["data_groups"]['input']
    target_group = model_meta["data_groups"]['target']

    x_name_list = input_group['names']
    y_name = as_list(target_group['names'])[0]
    
    id_name = model_meta["unique_id_name"]

    data_extrema = as_list(input_group["ranges"]["original"]) + as_list(target_group["ranges"]["original"])
    data_names = as_list(x_name_list) + as_list(y_name)
    data_ranges = {dname: convert_type(max(dvals) - min(dvals), float) for dname, dvals in zip(data_names, data_extrema)}

    all_results, all_scores = [], []

    unique_ids = model_results.get('unique_id', None)
    x0_data = model_results.get('x0_data', None)
    x1_data = model_results.get('x1_data', None)
    y0_data = model_results.get('y0_data', None)

    x_scalar_indices = input_group.get("channel_roles", {}).get("scalar", [])


    get_logger().debug(f"x0_data.shape = {x0_data.shape}")
    get_logger().debug(f"x1_data.shape = {x1_data.shape}")
    get_logger().debug(f"y0_data.shape = {y0_data.shape}")

    num_records, num_generated = x1_data.shape[:2]

    ###############################################################################################################

    x0_stack = process_stack(x0_data, squeeze=False)
    y0_stack = process_stack(y0_data, squeeze=False)
    x1_stack = process_stack(x1_data, squeeze=False)
    
    if not bypass_solver:
        sample_metrics = {f"s{i}": [] for i in range(1, num_generated+1)}
        y1_data = model_results.get('y1_data', None)
        get_logger().debug(f"y1_data.shape = {y1_data.shape}")
        y1_stack = process_stack(y1_data, squeeze=False)


    for i in range(num_records):
        unique_id = convert_type(unique_ids[i], int)
        x0 = x0_stack[i]
        x1 = x1_stack[i]
        y0 = y0_stack[i]
    
        get_logger().debug(f"x0.shape = {x0.shape}")
        get_logger().debug(f"x1.shape = {x1.shape}")
        get_logger().debug(f"y0.shape = {y0.shape}")

        result= {f"id_{id_name}": unique_id}

        for c, x_name in enumerate(x_name_list):
            if c in x_scalar_indices:
                x0_curr = unbroadcast_scalar(x0[c, :, :])
                x1_curr = unbroadcast_scalar(x1[:, c, :, :])
            else:
                x0_curr = x0[c, :, :].squeeze()
                x1_curr = x1[:, c, :, :].squeeze()

            result[f"x0_{x_name}"] = x0_curr
            result[f"x1_{x_name}"] = x1_curr

        result[f"y0_{y_name}"] = y0.squeeze()

        if not bypass_solver:
            y1 = y1_stack[i]
            get_logger().debug(f"y1.shape = {y1.shape}")
            result[f"y1_{y_name}"] = y1[:, 0, :, :]
            if num_generated > 1:
                for j, s_key in enumerate(sample_metrics.keys()):
                    metrics = ImageContinuousMetrics.all_metrics(y0, y1[j], data_ranges[y_name])
                    sample_metrics[s_key].append({"metrics": metrics})
            else:
                metrics = ImageContinuousMetrics.all_metrics(y0, y1.squeeze(), data_ranges[y_name])
                sample_metrics["s1"].append({"metrics": metrics})

        all_results.append(result)

    records_file = f'{current_phase}_records_{epoch_num}.hdf5' if epoch_num else f'{current_phase}_records_.hdf5'
    records_folder = os_path.join(output_path, "result_records")
    records_file_path = create_file_path(records_folder, records_file)
    save_to_hdf5(all_results, records_file_path)

    ###############################################################################################################
    if bypass_solver:
        return

    if debug_plots:
        save_debug_plots(all_results, current_phase, epoch_num, output_path, 
                        num_samples = 5, data_ranges = data_ranges, 
                        random_seed = model_meta["random_seed"])

    ###############################################################################################################

    loss_name = model_meta["loss_function_names"][current_phase]
    loss_target = "y"
    sample_losses_list = model_losses.get('sample', [])
    if not sample_losses_list:
        raise ValueError("Missing loss summary for each sample generated during model inference")

    for s, (s_key, s_list) in enumerate(sample_metrics.items()):  
        sample_losses = format_model_loss(loss_name, loss_target, summary=sample_losses_list[s])
        sample_scores = compute_model_scores(s_list, is_overall=False)

        scores_dict = {
            'epoch': epoch_num,
            'target': "y",
            'model_phase': current_phase,
            'sample_id': s_key,
            'x': x_name_list,
            'y': y_name,
            **sample_losses,
            **sample_scores
        }

        all_scores.append(scores_dict)

    if len(all_scores) > 1:
        overall_losses = format_model_loss(loss_name, loss_target, summary=model_losses["summary"])
        overall_scores = compute_model_scores(all_scores, is_overall=True)

        overall_scores = {
            'epoch': epoch_num,
            'target': "overall",
            'model_phase': current_phase,
            'sample_id': list(sample_metrics.keys()),
            'x': x_name_list,
            'y': y_name,
            **overall_losses,
            **overall_scores
        }
        all_scores.insert(0, overall_scores)

    scores_file = f'model_scores_{epoch_num}.json' if epoch_num else 'model_scores.json'
    save_to_json(create_file_path(f"{output_path}/model_scores", scores_file), all_scores, serialize=False)


def process_batch_data(x_full_norm, y_full_norm,  data_roles, data_names, data_transforms, s_params = {}, bypass_solver=False):

    x_roles, y_roles = data_roles['x'], data_roles['y']
    x_names, y_names = data_names['x'], data_names['y']
    x_transforms, y_transforms  = data_transforms['x'], data_transforms['y']

    x0_image_norm, x0_scalar_norm = split_channel_roles(x_full_norm, x_roles)

    HxW = x0_image_norm.shape[-2:]

    # --- Inverse-transform x0 ---
    x0_raw_chans = []
    for idx, nm in enumerate(x_names):
        if idx in x_roles["image"]:
            i = x_roles["image"].index(idx)
            x_raw = x_transforms[nm].inverse_transform(x0_image_norm[0, i], new_device="cpu") if x_transforms else x0_image_norm[0, i]
            x0_raw_chans.append(x_raw)
        else:
            j = x_roles["scalar"].index(idx)
            x_raw = x_transforms[nm].inverse_transform(x0_scalar_norm[0, j], new_device="cpu").squeeze().item() if x_transforms else x0_scalar_norm[0, j].squeeze().item()
            x0_raw_chans.append(pt.full(HxW, x_raw))

    x0_raw_stack = pt.stack(x0_raw_chans)

    # --- Inverse-transform y0 ---
    y0_image_norm, y0_scalar_norm = split_channel_roles(y_full_norm, y_roles)

    y0_raw_chans = []
    for idx, nm in enumerate(y_names):
        if idx in y_roles["image"]:
            i = y_roles["image"].index(idx)
            y_raw = y_transforms[nm].inverse_transform(y0_image_norm[0, i], new_device="cpu") if y_transforms else y0_image_norm[0, i]
            y0_raw_chans.append(y_raw)

    y0_raw_stack = pt.stack(y0_raw_chans)

    # --- Solver params from Y0 scalars  ---
    solver_base = {}

    if bypass_solver:
        return x0_raw_stack, y0_raw_stack, y0_image_norm, solver_base

    s_names = data_names['s']

    # a) any y-scalar that’s also a solver param
    for idx, nm in enumerate(y_names):
        if idx in y_roles["scalar"]:
            j = y_roles["scalar"].index(idx)
            y_raw = y_transforms[nm].inverse_transform(y0_scalar_norm[0, j], new_device="cpu").squeeze().item() if y_transforms else y0_scalar_norm[0, j].squeeze().item()
            solver_base[nm] = y_raw

    # --- Solver params from solver aux  ---
    if len(s_names) > 0:
        if isinstance(s_names, (list, tuple)):
            s_flat = []
            if isinstance(s_params, tuple):
                s_flat = [s_params[0].squeeze(0), s_params[1].squeeze(0)]
            elif isinstance(s_params, PT_TENSOR):
                s_flat = [s_params.squeeze(0)]
            else:
                s_flat = list(s_params)

            for i, s_name in enumerate(s_names):
                param = s_flat[i].squeeze().detach().cpu()
                solver_base[s_name] = param.item() if param.dim() == 0 else param.numpy()
        else:
            param = s_params.squeeze().detach().cpu()
            solver_base[s_names] = param.item() if param.dim() == 0 else param.numpy()

    return x0_raw_stack, y0_raw_stack, y0_image_norm, solver_base


def update_solver_params(x_sample_norm, x_roles, x_names, x_transforms, base_params={}, bypass_solver=False):

    x1_image_norm, x1_scalar_norm = split_channel_roles(x_sample_norm, x_roles)

    HxW = x1_image_norm.shape[-2:]

    x_sample_raw = []

    solver_params = base_params.copy() if base_params and not bypass_solver else {}
    
    # generated x images
    for i, idx in enumerate(x_roles['image']):
        name = x_names[idx]
        x_raw = x_transforms[name].inverse_transform(x1_image_norm[i], new_device='cpu') if x_transforms else x1_image_norm[i]
        x_sample_raw.append(x_raw)
        if bypass_solver:
            continue
        solver_params[name] = x_raw.detach().cpu().numpy()


    # generated x scalars
    for j, idx in enumerate(x_roles['scalar']):
        name = x_names[idx]
        x_raw = x_transforms[name].inverse_transform(x1_scalar_norm[j], new_device='cpu').squeeze().item() if x_transforms else x1_scalar_norm[j].squeeze().item()
        # broadcast the scalar to image for stacking
        x_sample_raw.append(pt.full(HxW, x_raw))
        if bypass_solver:
            continue
        solver_params[name] = x_raw
        
    x_sample_stack = pt.stack(x_sample_raw)

    return x_sample_stack, solver_params



def run_testing(model, dataloader, criterion,  device, solver_fn=None, samples = 5, noise = 1.0, limit = 100):
    model.eval()

    loss_values = []
    cumulative_loss_list = None
    result_lists = {
        'unique_id': [],
        'x0_data': [],
        'y0_data': [],
        'x1_data': [],
    }

    if model.training:
        get_logger().error(f"model.training={model.training} when testing")

    data_names = {
        'x': as_list(dataloader.input_names),
        'y': as_list(dataloader.target_names),
    }

    data_roles = {
        'x': dataloader.input_channel_roles,
        'y': dataloader.target_channel_roles
    }

    data_transforms = {
        'x': dataloader.input_transforms if dataloader.input_transforms else {},
        'y': dataloader.target_transforms if dataloader.target_transforms else {}
    }

    y_image_transform = data_transforms['y'].get(data_names['y'][0], None)

    bypass_solver = solver_fn is None
    if not bypass_solver:
        result_lists['y1_data'] = []
        if hasattr(dataloader, "solver_names"):
            data_names['s'] = as_list(dataloader.solver_names)

    for _, batch_data in enumerate(dataloader):
        unique_id = batch_data["id"]
        x_full_norm = batch_data["input"].to(device)     # (B, Cx, H, W)
        y_full_norm = batch_data["target"].to(device)    # (B, Cy, H, W)
        s_params = {} if bypass_solver else batch_data.get("solver", {}) 

        with pt.no_grad():
            x0_stack, y0_stack, y0_image_norm, base_params = process_batch_data(x_full_norm, y_full_norm, 
                                                                                data_roles, data_names, data_transforms, 
                                                                                s_params, bypass_solver=bypass_solver)
            result_lists['x0_data'].append(x0_stack)
            result_lists['y0_data'].append(y0_stack)

            # --- Generate samples of x ---
            x_samples_norm = model.generate_x(y_full_norm, samples=samples, noise=noise, limit=limit)
            if x_samples_norm.dim() == 5 and x_samples_norm.shape[0] == 1:
                x_samples_norm = x_samples_norm.squeeze(0)

            x1_samples_raw = []
            y1_samples_raw = []
            sample_losses = []

            for s in range(samples):
                x1_sample = x_samples_norm[s]
                x1_stack, solver_params = update_solver_params(x1_sample, data_roles['x'], data_names['x'], data_transforms['x'], base_params)
                x1_samples_raw.append(x1_stack)

                if bypass_solver:
                    continue
            
                y1_solved_raw = solver_fn(solver_params)
                y1_samples_raw.append(y1_solved_raw.unsqueeze(0))
                y1_solved_norm = y_image_transform(y1_solved_raw).to(device) if y_image_transform else y1_solved_raw.to(device)
                loss = criterion(y1_solved_norm, y0_image_norm)
                sample_losses.append(loss.detach().cpu().item())
                
            if sample_losses:
                loss_values.append(sum(sample_losses))

        # Add batch results to result_lists
        result_lists['unique_id'].append(unique_id)
        result_lists['x1_data'].append(pt.stack(x1_samples_raw))

        if not bypass_solver:
            result_lists['y1_data'].append(pt.stack(y1_samples_raw))
            if sample_losses:
                # Collect per-sample results
                if cumulative_loss_list is None:
                    cumulative_loss_list = [0.0] * len(sample_losses)
                for i, loss_val in enumerate(sample_losses):
                    cumulative_loss_list[i] += loss_val

    result_dict = {
        key: pt.stack(val, dim=0).cpu().numpy() 
            if key != 'unique_id' 
            else np.concatenate(val, axis=0)
        for key, val in result_lists.items()
    }
    
    if bypass_solver:
        return result_dict, {}
    
    sample_losses = np.array(cumulative_loss_list)
    num_batches = len(dataloader)

    loss_dict = {
        "summary": {
            'total_loss': np.sum(loss_values),
            'avg_loss': np.mean(loss_values),
            'num_batches': num_batches,
            'batch_size': 1
        },
        "sample": [
            {
                "index": i,
                "total_loss": sample_losses[i],
                "avg_loss": sample_losses[i] / num_batches
            }
            for i in range(len(sample_losses))
        ]
    }

    return result_dict, loss_dict


def save_results(model_results, model_losses, model_meta, current_phase, epoch_num=None, output_folder="model_results", save_dataset = True, debug_plots=False):
    get_logger().debug(f"current_phase: {current_phase}")

    output_path = create_folder(os_path.join(output_folder, f"{current_phase}_results"))

    z_data = model_results.get('z_data', None)

    latent_stats = LatentStatistics.all_stats(z_data)

    scores = compute_model_scores([{"metrics": latent_stats}])

    input_group = model_meta["data_groups"]['input']
    target_group = model_meta["data_groups"]['target']

    x_scalar_indices = input_group.get("channel_roles", {}).get("scalar", {})
    y_scalar_indices = target_group.get("channel_roles", {}).get("scalar", {})

    id_name = model_meta["unique_id_name"]
    x_name_list = input_group['names']
    y_name = as_list(target_group['names'])[0]

    losses = format_model_loss(model_meta["loss_function_names"][current_phase], 
                                ["z","y"] if current_phase=="training" else "z", 
                                summary=model_losses["summary"],
                                partial=model_losses.get("partial", {}))

    scores_dict = {'epoch': epoch_num, 
                    'target': "overall",
                    'model_phase':current_phase, 
                    'x': x_name_list,
                    'y': y_name, 
                    'z': "latent", 
                    **losses,
                    **scores}
    
    get_logger().debug(f"scores: {scores_dict.keys()}")

    scores_file = f'model_scores_{epoch_num}.json' if epoch_num else 'model_scores.json'
    scores_path = create_file_path(os_path.join(output_path, "model_scores"), scores_file)
    save_to_json(scores_path, [scores_dict], serialize=False)
    
    ###############################################################################################################
    if not (debug_plots or save_dataset):
        return None
    
    unique_ids = model_results.get('unique_id', None)
    num_records = len(unique_ids)
    
    x0_data = model_results.get('x0_data', None)
    x1_data = model_results.get('x1_data', None)
    y0_data = model_results.get('y0_data', None)
    y1_data = model_results.get('y1_data', None)

    x_transforms = model_meta.get('data_transforms', {}).get('input', {})
    y_transforms =  model_meta.get('data_transforms', {}).get('target', {})

    all_results = []
    z_results = []
    
    x0_stack = process_stack(x0_data, squeeze=False)
    x1_stack = process_stack(x1_data, squeeze=False)

    y0_stack = process_stack(y0_data, squeeze=False)      
    y1_stack = process_stack(y1_data, squeeze=False)

    z_stack = process_stack(z_data, squeeze=False)

    for i in range(num_records):
        x0 = x0_stack[i]
        x1 = x1_stack[i]

        y0 = y0_stack[i]
        y1 = y1_stack[i]

        unique_id = convert_type(unique_ids[i], int)

        result = {f"id_{id_name}": convert_type(unique_id, int)}

        for j, x_name in enumerate(x_name_list):
            x0_chan, x1_chan = x0[j], x1[j]

            if current_phase == "validation" and x_transforms.get(x_name):
                x0_chan = x_transforms[x_name].inverse_transform(pt.as_tensor(x0_chan), new_device="cpu").cpu().numpy()
                x1_chan = x_transforms[x_name].inverse_transform(pt.as_tensor(x1_chan), new_device="cpu").cpu().numpy()

            if j in x_scalar_indices:
                x0_chan = unbroadcast_scalar(x0_chan)
                x1_chan = unbroadcast_scalar(x1_chan)
            else:
                x0_chan = x0_chan.squeeze()
                x1_chan = x1_chan.squeeze()

            result[f"x0_{x_name}"] = x0_chan
            result[f"x1_{x_name}"] = x1_chan
        
        y0_chan, y1_chan = y0, y1

        if current_phase == "validation" and y_transforms.get(y_name):
            y0_chan = y_transforms[y_name].inverse_transform(pt.as_tensor(y0_chan), new_device="cpu").cpu().numpy()
            y1_chan = y_transforms[y_name].inverse_transform(pt.as_tensor(y1_chan), new_device="cpu").cpu().numpy()

        if 0 in y_scalar_indices:
            y0_chan = unbroadcast_scalar(y0_chan)
            y1_chan = unbroadcast_scalar(y1_chan)
        else:
            y0_chan = y0_chan.squeeze()
            y1_chan = y1_chan.squeeze()

        result[f"y0_{y_name}"] = y0_chan
        result[f"y1_{y_name}"] = y1_chan

        all_results.append(result)
        z_results.append({"id": f"{unique_id}", "z": z_stack[i]})

    ################################################################################################################

    if save_dataset:
        records_file = f'{current_phase}_records_{epoch_num}.hdf5' if epoch_num else f'{current_phase}_records_.hdf5'
        records_folder = os_path.join(output_path, "result_records")
        records_file_path = create_file_path(records_folder, records_file)
        save_to_hdf5(all_results, records_file_path)

    ################################################################################################################

    if debug_plots:
        z_stack = process_stack(z_data, squeeze=False)
        data_extrema = as_list(input_group["ranges"]["original"]) + as_list(target_group["ranges"]["original"])
        data_names = as_list(x_name_list) + as_list(y_name)
        data_ranges = {dname: convert_type(max(dvals) - min(dvals), float) for dname, dvals in zip(data_names, data_extrema)}
        save_debug_plots(all_results, current_phase, epoch_num, output_path, 
                        num_samples=3, data_ranges=data_ranges, 
                        latent_records=z_results, random_seed = model_meta["random_seed"])


def run_validation(model, dataloader, latent_criterion, device):
    model.eval()  # Set the model to evaluation mode

    if model.training:
        get_logger().error(f"model.training={model.training} when validating")


    loss_values = []

    result_lists = {
        'unique_id': [],
        'x0_data': [],
        'y0_data': [],
        'x1_data': [],
        'y1_data': [],
        'z_data': []
    }

    target_roles = dataloader.target_channel_roles

    for batch_num, batch_data in enumerate(dataloader):
        unique_id = batch_data["id"]
        x_full = batch_data["input"].to(device, non_blocking=True)
        y_full = batch_data["target"].to(device, non_blocking=True)
        y0_image, _ = split_channel_roles(y_full, target_roles)
        get_logger().debug(f"batch_data: {batch_data}")

        with pt.no_grad():
            z_latent, log_det = model.encode_z(x_full, y_full)
            assert pt.isfinite(log_det).all(), f"log_det contains NaN or ±Inf: log_det = {log_det}"
            
            z_loss = latent_criterion(z_latent, log_det)
            loss_values.append(z_loss.detach().cpu().item())

            get_logger().debug(f"batch[{batch_num}]: z_loss = {z_loss}, log_det = {log_det}")
            x_recon = model.decode_x(z_latent, y_full)
            y_recon = model.project_y(z_latent)


        y1_image, _ = split_channel_roles(y_recon, target_roles)

        result_lists['unique_id'].append(unique_id)
        result_lists['x0_data'].append(x_full.detach())
        result_lists['x1_data'].append(x_recon.detach())

        result_lists['y0_data'].append(y0_image.detach())
        result_lists['y1_data'].append(y1_image.detach())

        result_lists['z_data'].append(z_latent.detach())

    result_dict = {
        key: pt.cat(val, dim=0).cpu().numpy() if key != 'unique_id' else np.concatenate(val, axis=0)
        for key, val in result_lists.items()
    }

    loss_dict = {
        "summary": {
            'total_loss': np.sum(loss_values),
            'avg_loss': np.mean(loss_values),
            'num_batches': len(loss_values),
            'batch_size': len(batch_data)
        }
    }

    return result_dict, loss_dict


def run_training(model, dataloader, optimizer, latent_criterion, recon_criterion, beta_scheduler, epoch_num, device):
    model.train()

    if not model.training:
        get_logger().error(f"model.training={model.training} when training")

    total_loss_values = []

    z_loss_values = []
    y_loss_values = []
    y_loss_weighted = []
    beta_weights = []

    result_lists = {
        'unique_id': [],
        'x0_data': [],
        'y0_data': [],
        'x1_data': [],
        'y1_data': [],
        'z_data': []
    }


    target_roles = dataloader.target_channel_roles

    for batch_num, batch_data in enumerate(dataloader):
        x_full = batch_data["input"].to(device, non_blocking=True)
        y_full = batch_data["target"].to(device, non_blocking=True)
        unique_id = batch_data["id"]

        y0_image, y0_scalar = split_channel_roles(y_full, target_roles)

        get_logger().debug(f"x_input = {x_full.shape}, y_cond = {y_full.shape}")
        beta = beta_scheduler(epoch_num, batch_num)   
        beta_weights.append(beta)

        with pt.set_grad_enabled(model.training):
            z_latent, log_det = model.encode_z(x_full, y_full)
            assert pt.isfinite(log_det).all(), f"log_det contains NaN or ±Inf: log_det = {log_det}"

            z_loss = latent_criterion(z_latent, log_det)
            assert pt.isfinite(z_loss), f"z_loss is NaN or ±Inf: z_loss = {z_loss}"
            z_loss_values.append(z_loss.detach().cpu().item())
    
            x_recon = model.decode_x(z_latent, y_full)
            y_recon = model.project_y(z_latent)

            y1_image, y1_scalar = split_channel_roles(y_recon, target_roles)

            y_loss = recon_criterion(y1_image, y0_image)

            if y1_scalar is not None:
                y_loss += recon_criterion(y1_scalar, y0_scalar)

            assert pt.isfinite(y_loss), f"y_loss is NaN or ±Inf: y_loss = {y_loss}"
            y_loss_values.append(y_loss.detach().cpu().item())

            if beta > 0:
                weighted_loss = beta * y_loss
                total_loss = z_loss + weighted_loss
                y_loss_weighted.append(weighted_loss.detach().cpu().item())
            else:
                total_loss = z_loss
                y_loss_weighted.append(0.0)

            total_loss_values.append(total_loss.detach().cpu().item())

            get_logger().debug(f"Training[epoch={epoch_num}, batch={batch_num}]: total_loss = {total_loss}, z_loss = {z_loss}, y_loss = {y_loss}, beta = {beta}")

            total_loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        result_lists['unique_id'].append(unique_id)
        result_lists['x0_data'].append(x_full.detach())
        result_lists['x1_data'].append(x_recon.detach())

        result_lists['y0_data'].append(y0_image.detach())
        result_lists['y1_data'].append(y1_image.detach())
        
        result_lists['z_data'].append(z_latent.detach())

    result_dict = {
        key: pt.cat(val).cpu().numpy() 
            if key != 'unique_id' 
            else np.concatenate(val, axis=0)
        for key, val in result_lists.items()
    }


    loss_dict = {
        "summary": {
            "total_loss": convert_type(np.sum(total_loss_values), float),
            "avg_loss": convert_type(np.mean(total_loss_values), float),
            "num_batches": len(total_loss_values),
            "batch_size": len(batch_data)
        },
        "partial": {
            "z_loss": {
                "total_loss": convert_type(np.sum(z_loss_values), float),
                "avg_loss": convert_type(np.mean(z_loss_values), float)
            },
            "y_loss": {
                "raw":{
                    "total_loss": convert_type(np.sum(y_loss_values), float),
                    "avg_loss": convert_type(np.mean(y_loss_values), float)
                },
                "weighted":{
                    "total_loss": convert_type(np.sum(y_loss_weighted), float),
                    "avg_loss": convert_type(np.mean(y_loss_weighted), float),
                    "beta_weight": convert_type(np.mean(beta_weights), float)
                }
            }
        }
    }

    return result_dict, loss_dict 


def train_validate_test(model, model_params, batch_loaders, out_folder, objective_reporter=None, trial=None):

    train_loader = batch_loaders.get("training", None)
    val_loader = batch_loaders.get("validation", None)
    test_loader = batch_loaders.get("testing", None)

    device = model_params['device']
    num_epochs = model_params['num_epochs']
    loss_prefix = model_params.get('loss_prefix', 'avg')

    generation_samples = model_params.get('generation_samples', 100)
    generation_noise = model_params.get('generation_noise', 1.0)
    generation_limit = model_params.get("generation_limit", 1)

    testing_samples = model_params.get('testing_samples', 5)
    testing_limit = min(testing_samples, generation_limit)
    solver_key = model_params["solver_key"]
    
    recon_criterion = model_params['loss_instance']
    latent_criterion = model_params['latent_loss_instance']
    optimizer = pt.optim.Adam(model.parameters(), lr=model_params['learn_rate'], betas = (0.9, 0.999), weight_decay= 1e-4)
    
    phase_tracker = ModelPhaseTracker()
    model_tracker = BestModelTracker(objective_name=f"{model_params['latent_loss_name']}")

    freq = EpochFrequency(
        checkpoint=model_params['epoch_frequency']['checkpoint'], 
        training=model_params['epoch_frequency']['training'],
        validation=model_params['epoch_frequency']['validation'], 
        testing=model_params['epoch_frequency']['testing']
    )

    beta_scheduler = create_beta_scheduler(
        len(train_loader), 
        total_epochs = model_params['beta_schedule_epochs'], 
        warmup_epochs = model_params['beta_warmup_epochs'],
        min_value =  model_params['beta_schedule_params']["min_value"],
        max_value =  model_params['beta_schedule_params']["max_value"],
        mode = model_params['beta_schedule_params']["mode"]
    )

    solver_fn = SolverSetup().setup_solver_function(solver_key, model_params["data_groups"])
    bypass_solver = solver_key == "bypass" or solver_fn is None

    state_folder = os_path.join(out_folder, "checkpoints")

    checkpoint_model_state = (lambda epoch: epoch % freq.checkpoint == 0) if freq.checkpoint else (lambda _: False)
    run_validation_epoch = (lambda epoch: epoch == 1 or epoch % freq.validation == 0)
    run_testing_epoch = (lambda epoch: epoch % freq.testing == 0) if test_loader is not None and freq.testing > 0 else (lambda _: False)
    plot_debug_on = model_params.get("plot_debug_images", False)


    for epoch in range(1, num_epochs + 1):

        is_validation_epoch = run_validation_epoch(epoch)
        is_testing_epoch = run_testing_epoch(epoch)
        is_checkpoint_epoch = checkpoint_model_state(epoch)

        phase_tracker.set_phase('T')

        train_result, train_loss = run_training(model, train_loader, optimizer, latent_criterion, recon_criterion, beta_scheduler, epoch, device)

        get_logger().info(f"{phase_tracker.current_phase.title()} Epoch[{epoch}]: "
                                f"avg_loss = {train_loss['summary']['avg_loss']}, "
                                f"total_loss = {train_loss['summary']['total_loss']}")
        
        if objective_reporter: 
            objective_reporter.report(epoch, phase_tracker.current_phase, train_loss['summary'][f"{loss_prefix}_loss"])

        save_results(train_result, train_loss, model_params, phase_tracker.current_phase, 
                        epoch_num = epoch, output_folder = out_folder, 
                        save_dataset = is_validation_epoch, debug_plots = is_validation_epoch and plot_debug_on)

        if is_validation_epoch:
            phase_tracker.set_phase('V')

            val_result, val_loss = run_validation(model, val_loader, latent_criterion, device)
            loss_objective = val_loss['summary'][f"{loss_prefix}_loss"]

            get_logger().info(f"{phase_tracker.current_phase.title()} Epoch[{epoch}]: "
                                    f"avg_loss = {val_loss['summary']['avg_loss']}, "
                                    f"total_loss = {val_loss['summary']['total_loss']}")
            
            save_results(val_result, val_loss, model_params, phase_tracker.current_phase, 
                            epoch_num = epoch, output_folder = out_folder, 
                            save_dataset = is_validation_epoch, debug_plots = plot_debug_on)

            if objective_reporter: 
                objective_reporter.report(epoch, phase_tracker.current_phase, loss_objective)

            model_tracker.update_best_model(loss_objective, epoch, model, trial_number=trial.number if trial else None)

            if trial: trial.report(loss_objective, epoch)

        if is_checkpoint_epoch: 
            model_tracker.save_model_state(model, folder_name=state_folder, file_name=f"model_state_{epoch}")

        if is_testing_epoch:
            phase_tracker.set_phase('E')
            test_result, test_loss = run_testing(model, test_loader, recon_criterion, device, solver_fn=solver_fn,
                                                samples = testing_samples, noise = generation_noise, limit = testing_limit)
    
            if not (bypass_solver or test_loss):
                get_logger().info(f"{phase_tracker.current_phase.title()} Epoch[{epoch}]: "
                                        f"avg_loss = {test_loss['summary']['avg_loss']}, "
                                        f"total_loss = {test_loss['summary']['total_loss']}")
                if objective_reporter: 
                    objective_reporter.report(epoch, phase_tracker.current_phase, test_loss['summary'][f"{loss_prefix}_loss"])

            save_generated(test_result, model_params, phase_tracker.current_phase, 
                            model_losses=test_loss, bypass_solver=bypass_solver, epoch_num = epoch, 
                            output_folder = out_folder, debug_plots = plot_debug_on)

        if trial and trial.should_prune():
            get_logger().info(f"Epoch [{epoch}]: Stopping training early")
            break

    BestModelTracker.save_model_state(model, folder_name=state_folder, file_name=f"final_model_state_{epoch}")
    
    if model_tracker:
        phase_tracker.set_phase('E')
        model_tracker.save_best_model(state_folder)
        best_model = model_tracker.load_best_model(model)
        best_result, best_loss = run_testing(best_model, 
                                        test_loader, 
                                        recon_criterion,
                                        model_params['device'], 
                                        solver_fn=solver_fn,
                                        samples = generation_samples,
                                        noise = generation_noise,
                                        limit = generation_limit)
        
        save_generated(best_result,
                        model_params, 
                        phase_tracker.current_phase, 
                        model_losses=best_loss,
                        bypass_solver=bypass_solver,
                        epoch_num=model_tracker.best_epoch, 
                        output_folder=f"{out_folder}/best_epoch_test",
                        debug_plots = True)

    return loss_objective


def generate_samples(model, model_params, test_loader, epoch, output_folder):

    device = model_params["device"]
    criterion = model_params["latent_loss_instance"]

    generation_samples = model_params.get('generation_samples', 2)
    generation_noise = model_params.get('generation_noise', 1.0)
    generation_limit = model_params.get("generation_limit", 1)

    data_groups = model_params["data_groups"]
    solver_key = model_params["solver_key"]
    plot_debug_on = model_params["plot_debug_images"]

    phase_tracker = ModelPhaseTracker()
    phase_tracker.set_phase('E')

    solver_fn = SolverSetup().setup_solver_function(solver_key, data_groups)
    bypass_solver = solver_key == "bypass" or solver_fn is None

    test_results, test_loss = run_testing(model, 
                                        test_loader, 
                                        criterion, 
                                        device, 
                                        solver_fn = solver_fn,
                                        samples = generation_samples,
                                        noise = generation_noise,
                                        limit = generation_limit)

    save_generated(test_results, 
                    model_params, 
                    phase_tracker.current_phase, 
                    model_losses=test_loss, 
                    bypass_solver=bypass_solver,
                    epoch_num = epoch,
                    output_folder = output_folder,
                    debug_plots = plot_debug_on)
