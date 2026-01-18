from programs.utils.common import pt, np, sparse

SOLVER_MODULES = {
    "bypass": None,
    "electrostatic": "programs.core.solver_functions.electrostatic_solver",
    "heat_diffusion": "programs.core.solver_functions.heat_diffusion_solver",
    "simple": "programs.core.solver_functions.simple_solver",
}

SOLVER_KEYS = list(SOLVER_MODULES.keys())
SOLVER_DEFAULT = SOLVER_KEYS[0]

SOLVER_PARAMETERS = {
    "bypass": {
        "required": ["a", "b"],
        "default": {}
    },
    "electrostatic": {
        "required": ["potential_state_initial", "permittivity_map", "charge_distribution", "total_iterations"],
        "default": {}
    },
    "heat_diffusion": {
        "required": ["diffusion_map", "temp_state_initial", "total_iterations"],
        "default": {}
    },
    "simple": {
        "required": ["a", "b"],
        "default": {}
    },
}


def electrostatic_solver(potential_state_initial:np.ndarray=None, 
                        permittivity_map:np.ndarray=None, 
                        charge_distribution:np.ndarray=None, 
                        total_iterations:int|float=None) -> pt.Tensor:

    if not isinstance(potential_state_initial, np.ndarray) or potential_state_initial.ndim != 2:
        raise ValueError(f"potential_state_initial must be a 2D numpy array, but got {type(potential_state_initial)} with shape {getattr(potential_state_initial, 'shape', 'N/A')}")
    if not isinstance(permittivity_map, np.ndarray) or permittivity_map.ndim != 2:
        raise ValueError(f"permittivity_map must be a 2D numpy array, but got {type(permittivity_map)} with shape {getattr(permittivity_map, 'shape', 'N/A')}")
    if not isinstance(charge_distribution, np.ndarray) or charge_distribution.ndim != 2:
        raise ValueError(f"charge_distribution must be a 2D numpy array, but got {type(charge_distribution)} with shape {getattr(charge_distribution, 'shape', 'N/A')}")
    if not isinstance(total_iterations, (int, float)) or total_iterations <= 0:
        raise ValueError(f"total_iterations must be a positive nonzero float or int, but got {type(total_iterations)} with value {total_iterations}")

    num_iterations = int(total_iterations)
    inverse_permittivity_map = np.where(permittivity_map != 0, 1.0 / permittivity_map, 0.0)
    potential_map = potential_state_initial.copy()
    H, W = potential_map.shape[-2:]

    for _ in range(num_iterations+1):
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                neighbor_avg = (potential_map[i-1, j] + potential_map[i+1, j] +
                                potential_map[i, j-1] + potential_map[i, j+1]) / 4.0
                update_value = (charge_distribution[i, j] * inverse_permittivity_map[i, j] + neighbor_avg)
                potential_map[i, j] = update_value

    return pt.tensor(potential_map, dtype=pt.float32)


def heat_diffusion_solver(
    diffusion_map: np.ndarray,
    temp_state_initial: np.ndarray,
    total_iterations: int
) -> pt.Tensor:
    """
    Crank-Nicolson solver for heat diffusion with fixed settings.

    Args:
        diffusion_map (np.ndarray): Thermal diffusivity values, shape (H, W)
        temp_state_initial (np.ndarray): Initial temperature distribution, shape (H, W)
        total_iterations (int): Maximum number of iterations
        boundary_condition (str): boundary condition name
    Returns:
        pt.Tensor: Final temperature map as a torch tensor
    """
    if not isinstance(temp_state_initial, np.ndarray) or temp_state_initial.ndim != 2:
        raise ValueError(f"temp_state_initial must be a 2D numpy array, got {type(temp_state_initial)} with shape {getattr(temp_state_initial, 'shape', 'N/A')}")
    if not isinstance(diffusion_map, np.ndarray) or diffusion_map.ndim != 2:
        raise ValueError(f"diffusion_map must be a 2D numpy array, got {type(diffusion_map)} with shape {getattr(diffusion_map, 'shape', 'N/A')}")
    if temp_state_initial.shape != diffusion_map.shape:
        raise ValueError(f"Shape mismatch: temp_state_initial has shape {temp_state_initial.shape}, but diffusion_map has shape {diffusion_map.shape}")
    if not isinstance(total_iterations, (int, float)) or total_iterations <= 0:
        raise ValueError(f"total_iterations must be a positive nonzero float or int, but got {type(total_iterations)} with value {total_iterations}")

    def _build_implicit_laplacian(
        H: int,
        W: int,
        dx: float
    ) -> sparse.csr_matrix:
        """
        2D Laplacian via Kron-sum for 'neumann' homogeneous cases
        """
        def lap1d(n: int) -> sparse.csr_matrix:
            off  = np.ones(n-1)
            main = -2.0 * np.ones(n)
            main[[0,-1]] = -1.0
            T = sparse.diags([off, main, off], [-1,0,1], (n,n), format="csr")
            return T.tocsr()

        T_w = lap1d(W)
        T_h = lap1d(H)
        I_w = sparse.eye(W, format="csr")
        I_h = sparse.eye(H, format="csr")

        return (sparse.kron(I_h, T_w) + sparse.kron(T_h, I_w)) / dx**2

    H, W = temp_state_initial.shape
    N = H * W

    # ensure neumann boundary is applied
    temp_state_initial[0, :]   = temp_state_initial[1, :]
    temp_state_initial[-1, :]  = temp_state_initial[-2, :]
    temp_state_initial[:, 0]   = temp_state_initial[:, 1]
    temp_state_initial[:, -1]  = temp_state_initial[:, -2]

    steps = int(total_iterations)

    diffusion_flat = diffusion_map.ravel()
    dx = 1e-4
    dt = 0.4 * dx * dx / (4 * diffusion_flat.mean())

    L = _build_implicit_laplacian(H, W, dx)
    A = sparse.eye(N) - 0.5 * dt * sparse.diags(diffusion_flat) @ L
    B = sparse.eye(N) + 0.5 * dt * sparse.diags(diffusion_flat) @ L
    lu = sparse.linalg.splu(A.tocsc())

    T_prev = temp_state_initial.ravel().copy()
    T_curr = np.empty_like(T_prev)

    for _ in range(1, steps + 1):
        rhs = B @ T_prev
        T_curr = lu.solve(rhs)
        T_prev[:] = T_curr

    T_final = T_curr.reshape(H, W).copy()

    return pt.tensor(T_final, dtype=pt.float32)



def simple_solver(a: np.ndarray,
                    b: np.ndarray) -> pt.Tensor:
    """
    Args:
        a (np.ndarray): image 1
        b (np.ndarray): image 2


    Returns:
        d (np.ndarray)
    """
    a_img = a.squeeze().copy()
    if not isinstance(a, np.ndarray) or a.ndim != 2:
        raise ValueError(f"a must be a 2D numpy array, but got {type(a)} with shape {getattr(a, 'shape', 'N/A')}")
    b_img = b.squeeze().copy()
    if not isinstance(b, np.ndarray) or a.ndim != 2:
        raise ValueError(f"b must be a 2D numpy array, but got {type(b)} with shape {getattr(b, 'shape', 'N/A')}")

    c = a_img + b_img

    return pt.tensor(c, dtype=pt.float32)