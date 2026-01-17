from programs.utils.common import pt, validate_image_shape, PT_NN_MODULE, PT_TENSOR
NN = pt.nn
F = NN.functional
LA = pt.linalg


class ResidualBlock(NN.Module):
    def __init__(self, num_neurons: int, kernel: int, stride: int, padding: int, activation_func: PT_NN_MODULE):
        super().__init__()
        self.block = NN.Sequential(
            NN.Conv2d(
                in_channels=num_neurons, 
                out_channels=num_neurons, 
                kernel_size=kernel, 
                stride=stride, 
                padding=padding
            ),
            activation_func(),
            NN.Conv2d(
                in_channels=num_neurons, 
                out_channels=num_neurons, 
                kernel_size=kernel, 
                stride=stride, 
                padding=padding
            )
        )

    def forward(self, x):
        return x + self.block(x)


class Invertible1x1ConvLU(NN.Module):
    """
    Invertible 1x1 convolution with LU decomposition.

    1. LU Decomposition:
        - The weight matrix `W` is initialized by LU decomposition, where `W = P * L * U`.
        - `P` is a permutation matrix, `L` is lower triangular, and `U` is upper triangular.

    2. Diagonal Scaling:
        - The diagonal of `U` is used to compute a log-scale term (`log_s`).

    3. Efficient Inversion and Log-Determinant:
        - Inverse: W^(-1) = U^(-1) * L^(-1) * P^(-1).
        - Log-determinant: log|det(W)| = sum_i log|s_i|, where `s_i` are the diagonal entries of `U`.

    4. Forward Pass:
        - The convolution is applied with `W` (or `W^(-1}` if `reverse=True`).
        - The log-determinant is computed using the stored diagonal values of `U`, reducing computational complexity.

    5. Mathematical Formulas:
        - In forward mode: z = W * x.
        - In reverse mode: z = W^(-1) * x.
        - The log-determinant of the Jacobian is computed as: log|det(W)| = H * W * sum_i log|s_i|.
    """

    def __init__(self, input_channels: int, LU_decomposed: bool = True, generator: pt.Generator = None, eps: float = 1e-6):
        super().__init__()

        self.generator = generator

        C = input_channels

        # random orthonormal init
        w_init = LA.qr(pt.randn(C, C, generator = self.generator))[0]

        self.eps = eps

        if LU_decomposed:
            # LU factorization: P @ w_init = L @ U
            P, L, U = LA.lu(w_init)

            # split U into diag (s) and strict upper
            s      = pt.diag(U)
            sign_s = pt.sign(s)                     # ±1
            log_s  = pt.log(pt.abs(s) + eps)        # learn log‑scale
            U      = pt.triu(U, 1)                  # zero out diag

            # store P as a constant buffer, L/U/log_s/sign_s as params
            self.register_buffer("P", P)
            self.register_buffer("sign_s", sign_s)
            self.log_s = NN.Parameter(log_s)
            self.L = NN.Parameter(L)
            self.U = NN.Parameter(U)
        else:
            # fallback: full‑matrix
            self.weight = NN.Parameter(w_init + eps * pt.eye(C, device=self.L.device))

        self.LU = LU_decomposed

    def _assemble_W(self):
        if not self.LU:
            return self.weight

        # Reconstruct W = P @ (L with unit diag) @ (U with diag=sign_s*exp(log_s))
        L = pt.tril(self.L, -1) + pt.eye(self.L.size(0), device=self.L.device)
        U = pt.triu(self.U, 1) + pt.diag(self.sign_s * pt.exp(self.log_s) + self.eps)
        return self.P @ L @ U

    def forward(self, x: pt.Tensor, *args, reverse: bool =False):
        B, C, H, W = x.shape
        W_mat = self._assemble_W()

        if not reverse:
            W4d = W_mat.view(C, C, 1, 1)
            z = F.conv2d(x, W4d)
        else:
            x_flat = x.view(B, C, -1)
            x_perm = self.P.t() @ x_flat

            L_lower = pt.tril(self.L, -1) + pt.eye(C, device=x.device)
            y = LA.solve_triangular(L_lower, x_perm, upper=False, unitriangular=True)

            U_full = pt.triu(self.U, 1) + pt.diag(self.sign_s * pt.exp(self.log_s) + self.eps)
            z_flat = LA.solve_triangular(U_full, y, upper=True)

            z = z_flat.view(B, C, H, W)

        # log‑det via sum of log_s
        if self.LU:
            log_det_W = pt.sum(self.log_s)
        else:
            log_det_W = pt.logdet(self.weight)

        log_det = log_det_W * H * W

        if reverse:
            log_det = -log_det
        
        return z, log_det.expand(B)


class ConditionalAffineCoupling(NN.Module):
    def __init__(self, 
                input_channels: int, 
                condition_channels: int, 
                num_layers: int, 
                num_neurons: int, 
                activation_func: PT_NN_MODULE, 
                kernel: int, 
                stride: int, 
                padding: int,
                log_transform: str|None = None,
                log_bounds: tuple[float, float] = (0.0, 2.0),
                generator: pt.Generator = None
            ):
        super().__init__()

        self.generator = generator

        self.valid_methods = {None, "squared_tanh", "scaled_tanh", "sigmoid", "softplus", "clamp"}
        if log_transform not in self.valid_methods:
            raise ValueError("bound_method must be one of ['tanh', 'sigmoid', 'clamp', None]")

        if isinstance(log_transform, str):
            if not (isinstance(log_bounds, (tuple, list)) and len(log_bounds) == 2
                    and all(isinstance(v, (int, float)) for v in log_bounds)):
                raise TypeError("log_bounds must be a tuple/list of two numeric values (min_log, max_log), e.g. (-3.0, 3.0)")

            self.min_log, self.max_log = log_bounds
            if self.min_log > self.max_log:
                raise ValueError("log_scale must in format (min_Log, max_log), where min_Log <= max_log")
            
        self.bounded_fn = self._init_transform(log_transform)

        self.split1 = input_channels // 2
        self.split2 = input_channels - self.split1

        layers: list[NN.Module] = []

        layers += [
            NN.Conv2d(in_channels=self.split1 + condition_channels, 
                    out_channels=num_neurons, 
                    kernel_size=kernel, 
                    stride=stride, 
                    padding=padding),
            activation_func()
        ]
    
        for _ in range(num_layers):
            layers += [
                NN.Conv2d(in_channels=num_neurons, 
                        out_channels=num_neurons, 
                        kernel_size=kernel, 
                        stride=stride, 
                        padding=padding),
                activation_func()
            ]

        last = NN.Conv2d(in_channels=num_neurons, 
                            out_channels=2 * self.split2, 
                            kernel_size=kernel, 
                            stride=stride, 
                            padding=padding)

        #  identity initialization 
        if last.weight.ndim == 2:
            NN.init.eye_(last.weight) 
        else:
            NN.init.xavier_normal_(last.weight, gain=1.0, generator = self.generator)  

        NN.init.zeros_(last.bias)

        layers.append(last)

        self.net = NN.Sequential(*layers)

    def _init_transform(self, method):
        transforms = {
            "squared_tanh": lambda x: pt.tanh(x)**2,
            "scaled_tanh" : lambda x: self.min_log + (self.max_log - self.min_log) * pt.tanh(x).abs(),
            "sigmoid" : lambda x: self.min_log + (self.max_log - self.min_log) * pt.sigmoid(x),
            "softplus": lambda x: pt.clamp(
                self.min_log + (self.max_log - self.min_log) * F.softplus(x),
                self.min_log, self.max_log
            ),
            "clamp" : lambda x: pt.clamp(x, self.min_log, self.max_log),
        }
        return transforms.get(method, lambda x: x)

    def forward(self, x: PT_TENSOR, cond: PT_TENSOR, reverse: bool = False):

        cond = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=False)

        x1, x2 = x.split([self.split1, self.split2], dim=1)

        h = self.net(pt.cat([x1, cond], dim=1))
        raw_log_s, t = h.chunk(2, dim=1)
        log_s = self.bounded_fn(raw_log_s)
        
        if reverse:
            s = pt.exp(-log_s)
            y2 = (x2 - t) * s
            log_det = -log_s.flatten(1).sum(-1)
        else:
            s = pt.exp(log_s)
            y2 = s * x2 + t
            log_det = log_s.flatten(1).sum(-1)

        x_out = pt.cat([x1, y2], dim=1)

        return x_out, log_det


class ConditionalNF(NN.Module):
    def __init__(self, model_params: dict[str, any]):
        super(ConditionalNF, self).__init__()
        groups = model_params.get('data_groups', {})

        input_data = groups.get('input', {})
        target_data = groups.get('target', {})

        self.input_dims = input_data.get('dimensions', None)
        assert self.input_dims != None, "Model requires input dimensions for initialization"
        assert validate_image_shape(self.input_dims), "Model inputs must be 'image' types with at least 1 channel"

        self.target_dims = target_data.get('dimensions', None)
        assert self.target_dims != None, "Model requires target dimensions for initialization"
        assert validate_image_shape(self.target_dims), "Model targets must be 'image' types with at least 1 channel"
        assert len(target_data['channel_roles']['image']) == 1, "Model target channels must contain only one 'image' and any additional channels must be broadcasted scalars"

        self.device = model_params.get('device', 'cpu')
        self.random_seed = model_params.get('random_seed', None)
        self.init_generator = pt.Generator(device="cpu").manual_seed(self.random_seed)
        self.sample_generator = pt.Generator(device=self.device).manual_seed(self.random_seed)

        self.input_channels = self.input_dims[0]
        self.target_channels = self.target_dims[0]

        self.num_blocks = model_params.get('block_networks', 3)
        self.num_layers = model_params.get('hidden_layers', 3)
        self.num_neurons = model_params.get('num_neurons', 64)
        self.activation_func = model_params.get('activation_function', NN.Tanh)

        self.kernel = model_params['conv_params']['kernel']
        input_hw, target_hw = self.input_dims[1:], self.target_dims[1:]
        min_hw = min(*input_hw, *target_hw)
        if self.kernel > min_hw:
            self.kernel = min_hw

        self.stride = model_params['conv_params']['stride']
        self.padding = (self.kernel - self.stride) // 2

        log_params = model_params.get('affine_log_params', {})
        self.log_transform = log_params.get("transform", None)
        self.log_bounds = log_params.get('log_bounds', (0, 2.0))

        self.log_det_lb, self.log_det_ub = -1e10, 1e10


        blocks = []
        for _ in range(self.num_blocks):
            blocks.append(
                ConditionalAffineCoupling(
                    input_channels=self.input_channels,
                    condition_channels=self.target_channels,
                    num_layers=self.num_layers,
                    num_neurons=self.num_neurons,
                    activation_func=self.activation_func,
                    kernel=self.kernel,
                    stride=self.stride,
                    padding=self.padding,
                    log_transform=self.log_transform,
                    log_bounds=self.log_bounds,
                    generator = self.init_generator
                )
            )
            blocks.append(
                Invertible1x1ConvLU(
                    self.input_channels, 
                    generator = self.init_generator
                )
            )

        self.blocks = NN.ModuleList(blocks)

        self.reverse_blocks = list(reversed(self.blocks))

        self.projection_method = model_params.get('projection_method', 'linear')
        self.projection_activation = model_params.get('projection_activation', NN.ReLU)
        
        self._init_projection()

        self.to(self.device)

    def _init_projection(self):

        if self.projection_method == "linear":

            layers = [NN.Conv2d(
                in_channels=self.input_channels, 
                out_channels=self.target_channels,
                kernel_size=1, 
                stride=1, 
                padding=0
            )]

        elif self.projection_method == "mlp":
            layers = []
            in_channels = self.input_channels
            out_channels = self.num_neurons

            # input layer + N hidden layers
            for _ in range(self.num_layers+1):
                layers.append(NN.Conv2d(
                        in_channels=in_channels, 
                        out_channels=out_channels,
                        kernel_size=self.kernel, 
                        stride=self.stride, 
                        padding=self.padding
                ))
                layers.append(self.projection_activation())
                in_channels = out_channels
                out_channels = max(in_channels//2, self.target_channels)

            layers.append(NN.Conv2d(in_channels=in_channels, 
                            out_channels=self.target_channels,
                            kernel_size=self.kernel, 
                            stride=self.stride, 
                            padding=self.padding))

        elif self.projection_method == "resnet":

            layers = [
                NN.Conv2d(
                    in_channels=self.input_channels, 
                    out_channels=self.num_neurons, 
                    kernel_size=self.kernel, 
                    stride=self.stride, 
                    padding=self.padding
                ),
                ResidualBlock(
                    self.num_neurons, 
                    self.kernel, 
                    self.stride, 
                    self.padding,
                    self.projection_activation
                ),
                ResidualBlock(
                    self.num_neurons, 
                    self.kernel, 
                    self.stride, 
                    self.padding,
                    self.projection_activation
                ),
                NN.Conv2d(
                    in_channels=self.num_neurons, 
                    out_channels=self.target_channels, 
                    kernel_size=self.kernel, 
                    stride=self.stride, 
                    padding=self.padding
                )
            ]

        else:
            raise ValueError(f"Unknown projection_method '{self.projection_method}'")
        
        self.projection_head = NN.Sequential(*layers)


    def _forward(self, x: PT_TENSOR, y: PT_TENSOR):
        """Forward process to learn the mapping from input x to condition y.
        The total Log determinant is computed for each block network.

        Args:
            x (tensor): input x
            y (tensor): condition y

        Returns:
            tensor: latent z, log determinant
        """
        log_det = 0.0
        for blk in self.blocks:
            x, ld = blk(x, y, reverse=False)
            log_det += ld
        log_det = pt.clamp(log_det, min=self.log_det_lb, max=self.log_det_ub)
        return x, log_det
    
    def _reverse(self, z: PT_TENSOR, y: PT_TENSOR):
        """Reverse process to invert the mapping condition y to input x.
        The total Log determinant omitted as reverse is designed for inference.

        Args:
            z (tensor): latent z
            y (tensor): condition y

        Returns:
            tensor: latent z
        """
        for blk in self.reverse_blocks:
            z, _ = blk(z, y, reverse=True)
        return z
    
    def encode_z(self, x: PT_TENSOR, y: PT_TENSOR):
        """Encodes input x and condition y to latent z: (x, y) -> z
        
        Args:
            x (tensor): input x
            y (tensor): condition y

        Returns:
            tensor: latent z
        """
        return self._forward(x, y)

    def decode_x(self, z: PT_TENSOR, y: PT_TENSOR):
        """Decodes input x from latent z and condition y: (z, y) -> x
        Reconstructs original x during training and generates new x during inference.

        Args:
            z (tensor): latent z
            y (tensor): condition y

        Returns:
            tensor: input x
        """
        return self._reverse(z, y)

    def project_y(self, z: PT_TENSOR):
        """Projects latent z to condition y via 1x1 convolution: z ~> y

        Args:
            z (tensor): latent z

        Returns:
            tensor: "predicted" condition y
        """

        return self.projection_head(z)

    def generate_x(self, y: PT_TENSOR, samples: int = 2, noise: int|float=1.0, limit: int = 1):
        """Generates n samples of x for a given y

        Args:
            y (tensor): y condition to generate samples of x. Shape (B, C=1, H, W)
            samples (int, optional): total samples to generate. Defaults to 2.
            noise (float, optional): latent noise scaling factor [0.0, 1.0). Defaults to 1.0. 
                - noise = 0.0 → No noise: generates identical outputs for each sample (deterministic)
                - noise = 1.0 → Standard noise: generates typical stochastic samples (default behavior)
                - noise > 1.0 → Adds extra noise: increases diversity and randomness of samples
                - noise < 1.0 → Reduces noise: outputs are similar and tightly clustered around the mean
            sample_limit (int, optional): number of samples per sub-batch. Must be in [1, samples].
                    Defaults to 1 (max memory savings). If equal to `samples`, runs full-batch sampling.

        Returns:
            tensor: x samples 
        """

        assert y.dim() == 4, f"Expected 'y' to be a 4D tensor (B, C, H, W), but got shape: {y.shape}"
        assert isinstance(samples, int) and samples >= 1, f"Expected 'samples' to be a int greater than or equal to 1"
        assert isinstance(noise, (int, float)) and noise >= 0, f"Expected 'noise' top be a float greater than or equal to 0"
        assert isinstance(limit, int) and limit > 0, f"limit must be in [1, {samples}] but got {limit}"

        B, _, H, W = y.shape

        sample_batches = []
        for start in range(0, samples, limit):
            n = min(limit, samples - start)
            z = pt.randn(
                B * n,
                self.input_channels,
                H,
                W,
                device=y.device,
                generator=self.sample_generator
            ) * noise

            y_rep = y.unsqueeze(1).repeat(1, n, 1, 1, 1)
            y_rep = y_rep.view(B * n, -1, H, W)

            x_flat = self.decode_x(z, y_rep)
            chunk = x_flat.view(B, n, self.input_channels, H, W)

            sample_batches.append(chunk)
            del z, y_rep, x_flat
            pt.cuda.empty_cache()

        return pt.cat(sample_batches, dim=1)


model_config = {
    'model_key': 'cfn',
    'model_class': ConditionalNF,
    'loss_prefix': 'avg',
    'loss_function': 'mse',
    'latent_loss_function': 'nll',
    'activation_function': 'tanh',
    'conv_params': {'kernel': 3, 'stride': 1},
    'hyperparams':['learn_rate', 
                    'num_neurons', 
                    'block_networks',
                    'hidden_layers', 
                    'loss_function', 
                    'activation_function',
                    'beta_schedule_epochs',
                    'projection_method',
                    'projection_activation']
}



if __name__ == "__main__":
    model_params = {
        'data_groups':{
            'input':{
                'dimensions': (2, 64, 64),
            },
            'target':{
                'dimensions': (3, 64, 64),
            },
        },
        'conv_params': {'kernel': 3, 'stride': 1},
        'block_networks': 3,
        'hidden_layers': 3,
        'num_neurons': 32,
        'activation_function': NN.Tanh
    }

    model = ConditionalNF(model_params)
    B = 10

    input_dims = model_params['data_groups']['input']['dimensions']
    x  = pt.randn(B, *input_dims)

    target_dims = model_params['data_groups']['target']['dimensions']
    y  = pt.randn(B, *target_dims)

    print("Input shape:", x.shape)
    z, log_det = model.encode_z(x, y)
    print("Encode Shape (Forward):", z.shape)
    print("Log Det = ", log_det)
    x_recon = model.decode_x(z, y)
    print("Decode Shape (Reconstruct):", x_recon.shape)
    x_samples= model.generate_x(y, samples=5)
    print("Generate Shape (Generate):", x_samples.shape)
