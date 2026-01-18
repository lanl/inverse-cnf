from programs.utils.logger_setup import get_logger
from programs.utils.common import np, pt, import_module_path
from torchmetrics.image import StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure

LOSS_MODULES = {
    'mse': 'torch.nn.MSELoss',
    'l1': 'torch.nn.L1Loss',
    'smooth-l1': 'torch.nn.SmoothL1Loss',
    'ssim': 'programs.core.loss_functions.SSIMLoss',
    'ms-ssim': 'programs.core.loss_functions.MSSSIMLoss',
    'tv': 'programs.core.loss_functions.TVLoss',
    'kl-div': 'programs.core.loss_functions.KLDivergenceLoss',
    'nll': 'programs.core.loss_functions.NegativeLogLikelihoodLoss',
    'weighted-hybrid': 'programs.core.loss_functions.WeightedHybridLoss'
}


class WeightedHybridLoss:
    def __init__(self, loss_keys:list[str],
                    weights:list[float], 
                    image_dims:int=None, 
                    device:str=None, 
                    data_range:float=None,
                    multi_scale_params:dict=None):

        self.device = device
        self.image_dims = image_dims
        self.multi_scale_params = multi_scale_params
        self.loss_keys = loss_keys
        self.data_range = data_range
        self.weights = weights

        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError(f"sum of {weights} must equal 1.0")
        if len(weights) != len(loss_keys):
            raise ValueError("Each loss function should have a corresponding weight")

        self.loss_functions = []
        for key in self.loss_keys:
            loss_module = import_module_path(LOSS_MODULES[key])
            self.loss_functions.append(self._initialize(loss_module, key))

        self.custom_loss_name = "_".join([f"{w}X{key}" for w, key in zip(self.weights, self.loss_keys)])

    def _initialize(self, loss_module, loss_name):
        if loss_name == 'ssim':
            return loss_module(self.device, data_range=self.data_range).to(self.device)
        if loss_name == 'ms-ssim':
            return loss_module(self.image_dims, self.device, 
                                data_range=self.data_range, 
                                kernel_size=self.multi_scale_params['kernel'], 
                                scale_weights=self.multi_scale_params['weights']).to(self.device)
        return loss_module()

    def to(self, device):
        if device != self.device:
            self.device = device
            self.loss_functions = [lf.to(device) 
                                    if hasattr(lf, 'to') else 
                                    lf for lf in self.loss_functions]
            
        return self

    def __call__(self, y_pred, y_true):
        if hasattr(y_true, "device") and y_true.device != self.device:
            y_true = y_true.to(self.device)

        if hasattr(y_pred, "device") and y_pred.device != self.device:
            y_pred = y_pred.to(self.device)

        hybrid_loss = 0.0
        for weight, loss_key, loss_fn in zip(self.weights, self.loss_keys, self.loss_functions):
            if isinstance(y_pred, tuple) and loss_key != "kl":
                loss = loss_fn(y_pred[0], y_true)
            else:
                loss = loss_fn(y_pred, y_true)
            hybrid_loss += weight * loss

        return hybrid_loss


class SSIMLoss:
    def __init__(self, device, data_range=1.0):
        self.data_range = data_range   
        self.device = device
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=self.data_range).to(self.device)

    def to(self, device):
        if device != self.device:  # Avoid redundant transfers
            self.device = device
            self.ssim_metric = self.ssim_metric.to(device)
        return self

    def __call__(self, y_pred, y_true):
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        if y_true.device != self.device:
            y_true = y_true.to(self.device)
        if y_pred.device != self.device:
            y_pred = y_pred.to(self.device)
        ssim_value = self.ssim_metric(y_pred, y_true)
        return 1-ssim_value


class MSSSIMLoss:
    def __init__(self, image_dims, device, data_range=1.0, kernel_size: None | int = None, scale_weights: None | tuple[float] = None):
        self.data_range = data_range   
        self.device = device

        self.image_size = min(image_dims[-2:])

        self.kernel_size = kernel_size
        self.scale_weights = tuple(scale_weights)
        
        self._validate_params()

        self.ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=self.data_range,
            kernel_size=self.kernel_size,
            betas=self.scale_weights
        ).to(self.device) 

        get_logger().info(f"MS-SSIM params: kernel = {self.kernel_size}, betas = {self.scale_weights}")

    def _validate_params(self):
        if self.kernel_size > self.image_size:
            raise ValueError(f"Kernel size {self.kernel_size} cannot be larger than image size {self.image_size}")

        max_valid_scales = int(np.log2( self.image_size)) - 2  
        expected_num_scales = ( self.kernel_size + 1) // 2

        if expected_num_scales > max_valid_scales:
            raise ValueError(f"Image size { self.image_size} is too small to support {expected_num_scales} scales (max valid scales = {max_valid_scales}) for kernel size { self.kernel_size}")

    def to(self, device):
        if device != self.device:
            self.device = device
            self.ms_ssim_metric = self.ms_ssim_metric.to(device)
        return self

    def __call__(self, y_pred, y_true):
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        if y_true.device != self.device:
            y_true = y_true.to(self.device)
        if y_pred.device != self.device:
            y_pred = y_pred.to(self.device)
        ms_ssim_value = self.ms_ssim_metric(y_pred, y_true)
        return 1 - ms_ssim_value


class TVLoss(pt.nn.Module):
    def __init__(self, reduction='mean'):
        super(TVLoss, self).__init__()
        self._supported = ['mean', 'sum']

        if reduction not in self._supported:
            raise ValueError(f"Invalid reduction method '{reduction}'. Supported methods: {self._supported}")

        self.reduction = reduction  # 'mean' or 'sum'

    def forward(self, y_pred, y_true):
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        diff_x = pt.abs(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:])
        diff_y = pt.abs(y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :])

        if self.reduction == 'mean':
            return (diff_x.mean() + diff_y.mean())
        elif self.reduction == 'sum':
            return (diff_x.sum() + diff_y.sum())


class KLDivergenceLoss(pt.nn.Module):
    def __init__(self, reduction='mean'):
        super(KLDivergenceLoss, self).__init__()
        self.reduction = reduction.lower()
        self._supported = ['mean', 'sum', 'none']
        if self.reduction not in {'mean', 'sum', 'none'}:
            raise ValueError(f"Invalid reduction method '{reduction}'. Supported methods: {self._supported}")

    def forward(self, y_pred, y_true):
        if not isinstance(y_pred, tuple):
            raise ValueError(f"Expected <y_pred> to be tuple: reconstructed, (logvar, mu)")

        _, (logvar, mu) = y_pred
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl = kl.sum(dim=1)

        if self.reduction == 'mean':
            return kl.mean()
        elif self.reduction == 'sum':
            return kl.sum()
        else:
            return kl


class NegativeLogLikelihoodLoss(pt.nn.Module):
    def __init__(self, reduction='mean'):
        super(NegativeLogLikelihoodLoss, self).__init__()

        self.reduction = reduction
        if self.reduction not in {'mean', 'sum', 'none'}:
            raise ValueError(f"Invalid reduction method '{reduction}'. Supported methods: {self._supported}")

    def forward(self, z, log_det):
        # Negative log p(z) for standard normal
        neg_log_pz = 0.5 * (z ** 2).sum(dim=tuple(range(1, z.dim())))
        # Combine with change-of-variable term
        nll = neg_log_pz - log_det

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll
