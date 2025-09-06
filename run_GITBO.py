
import torch
import argparse
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime
from tabpfn.base import initialize_tabpfn_model
import math
from torch import Tensor
from dataclasses import dataclass
from torch.quasirandom import SobolEngine

import warnings
warnings.filterwarnings("ignore")

def reset_device_memory(device: torch.device):
    """Reset memory stats depending on device type."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        # MPS doesn't support empty_cache/reset_peak_memory_stats
        # Just a no-op
        pass

def get_max_memory_allocated(device: torch.device) -> float:
    """Return peak memory used in bytes, if supported."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated()
    elif device.type == "mps":
        # torch.mps.current_allocated_memory() exists, but not peak
        return torch.mps.current_allocated_memory()
    else:
        return 0.0  # CPU fallback

@dataclass
class Unified_TS_State:
    """
    Unified state class that handles both constrained (SCBO) and unconstrained (TuRBO) cases.
    When constraints are not present (C is None), it behaves like TuRBO.
    When constraints are present, it behaves like SCBO.
    Source: https://botorch.org/docs/tutorials/turbo_1/
    """

    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3, botorch uses 10
    best_value: float = -float("inf")
    best_constraint_values: Tensor = None  # Will be initialized if constraints exist
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_ts_state(state: Unified_TS_State, Y_next: Tensor, C_next: Tensor = None):
    """
    Unified update method that handles both constrained and unconstrained cases.
    Args:
        state: UnifiedTurboState instance
        Y_next: Tensor of objective values
        C_next: Optional tensor of constraint values. If None, unconstrained case is assumed.
    Source: https://botorch.org/docs/tutorials/turbo_1/
    """
    if C_next is None:
        # Unconstrained case (TuRBO)
        if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = max(Y_next).item()
        else:
            state.success_counter = 0
            state.failure_counter += 1
    else:
        # Constrained case (SCBO)
        # Initialize best_constraint_values if not already done
        if state.best_constraint_values is None:
            state.best_constraint_values = torch.ones_like(C_next[0]) * float("inf")

        # Pick the best point from the batch
        best_ind = get_sorted_indices(Y=Y_next, C=C_next)
        best_ind = best_ind[0]
        y_next, c_next = Y_next[best_ind], C_next[best_ind]

        if (c_next <= 0).all():
            # At least one new candidate is feasible
            improvement_threshold = state.best_value + 1e-3 * math.fabs(
                state.best_value
            )
            if (
                y_next > improvement_threshold
                or (state.best_constraint_values > 0).any()
            ):
                state.success_counter += 1
                state.failure_counter = 0
                state.best_value = y_next.item()
                state.best_constraint_values = c_next
            else:
                state.success_counter = 0
                state.failure_counter += 1
        else:
            # No new candidate is feasible
            total_violation_next = c_next.clamp(min=0).sum(dim=-1)
            total_violation_center = state.best_constraint_values.clamp(min=0).sum(
                dim=-1
            )
            if total_violation_next < total_violation_center:
                state.success_counter += 1
                state.failure_counter = 0
                state.best_value = y_next.item()
                state.best_constraint_values = c_next
            else:
                state.success_counter = 0
                state.failure_counter += 1

    # Update trust region length
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:
        state.restart_triggered = True
        state.length = min(8.0 * state.length, state.length_max)

    return state


def generate_batch_PFN(
    state,
    WEIGHTS,  # REQUIRE WEIGHT DEFINITION
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    n_candidates=None,  # Number of candidates for Thompson sampling
    C=None,  # Constraint values (optional)
    tkwargs={"device": torch.device("cpu"), "dtype": torch.float32},
):
    """
    Generalized generate_batch function that handles both constrained and unconstrained cases.
    Only generates trust region bounds and candidate points.

    Args:
        state: TurboState or ScboState object
        WEIGHTS: Weights derived from PFN
        X: Evaluated points on the domain [0, 1]^d
        Y: Function values
        n_candidates: Number of candidates for Thompson sampling
        C: Constraint values (optional, if None then unconstrained problem)

    Returns:
        tr_lb: Trust region lower bounds
        tr_ub: Trust region upper bounds
        X_cand: Candidate points generated
    """
    X = X.to(**tkwargs)
    Y = Y.to(**tkwargs)
    WEIGHTS = WEIGHTS.to(**tkwargs)
    if C is not None:
        C = C.to(**tkwargs)

    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    with torch.no_grad():
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Select trust region center

        # Get the best as center
        score_idx = get_sorted_indices(
            Y, C
        )  # The largest feasible value should be at index 0
        x_center = X[score_idx[0], :].clone()

        # Ensure state.length is a tensor on the correct device
        length = torch.tensor(state.length, **tkwargs)

        # Calculate trust region bounds using TuRBO's scaling
        # No weights scaling for PFN
        # weights = WEIGHTS / weights.mean()
        # weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        weights = WEIGHTS.clone()
        tr_lb = torch.clamp(x_center - weights * length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * length / 2.0, 0.0, 1.0)

        # Generate candidates
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(**tkwargs)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[
            ind, torch.randint(0, dim - 1, size=(len(ind),), device=tkwargs["device"])
        ] = 1

        # Create candidate points
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

    return tr_lb, tr_ub, X_cand


def get_sorted_indices(Y, C):
    """Return sorted indices based on three scenarios.

    Args:
        Y: Tensor of objective function values
        C: Tensor of constraint values (optional)

    Returns:
        sorted_indices: Tensor of indices sorted from best to worst according to:
            - If no constraints: sorted by Y values (descending)
            - If constraints but no feasible points: sorted by total constraint violation (ascending)
            - If constraints with feasible points: sorted by Y values with infeasible points set to -inf (descending)
    """
    # print(f"C: {C == None}")
    if C is None:
        # Case 1: No constraints - sort by objective value (descending)
        # print(f'Case1 Y: {Y}')

        return torch.argsort(Y.squeeze(-1), descending=True)

    # Check feasibility for all points
    is_feas = (C <= 0).all(dim=1)
    # print(f'is_feas: {is_feas}')

    if is_feas.any():
        # Case 3: Has feasible points - set infeasible points to -inf and sort
        score = Y.clone()
        score[~is_feas] = float("-inf")

        # print(f'Case3 score: {torch.argsort(score.squeeze(-1), descending=True)}')
        return torch.argsort(score.squeeze(-1), descending=True)
    else:
        # Case 2: No feasible points - sort by total constraint violation (ascending)
        violation = C.clamp(min=0).sum(dim=-1)

        # print(f'Case2 violation')
        return torch.argsort(violation)


class VanillaDirectTabPFNRegressor:
    """
    Minimal "direct" TabPFN regressor wrapper for inference-only usage.
    Uses the real `load_model` function to load a pretrained TabPFN model.
    """

    def __init__(
        self,
        model_path: str = "auto",
        fit_mode: str = "fit_preprocessors",
        device: str = "auto",
        inference_precision: str = "auto",
        random_state=None,
    ):
        self.model_path = model_path
        self.fit_mode = fit_mode
        self.device = device
        self.inference_precision = inference_precision
        self.random_state = random_state

        # 1) Static seed for reproducibility
        static_seed, _ = self._infer_random_state(self.random_state)

        # 2) Load the pretrained TabPFN model, config, and bar distribution
        self.model_, self.config_, self.bardist_ = self._initialize_tabpfn_model(
            model_path=self.model_path,
            which="regression",  # TabPFN's recognized keyword is "regression"
            fit_mode=self.fit_mode,
            static_seed=static_seed,
        )

        # 3) Choose device + set precision
        self.device_ = self._infer_device_and_type(self.device)

        self.model_.to(self.device_)
        self.bardist_.borders = self.bardist_.borders.to(self.device_)

        self.use_autocast_, self.forced_inference_dtype_, byte_size = (
            self._determine_precision(
                self.inference_precision,
                self.device_,
            )
        )

        # Move the loaded model to device, set to eval mode
        self.model_.to(self.device_)
        self.model_.eval()

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        single_eval_pos: int,
    ) -> dict:
        """
        Forward pass for inference with shape [time, batch, features] for X
        and [time, batch] for Y.
        """
        # Move to device
        X = X.to(self.device_)
        Y = Y.to(self.device_)

        # Suppose Y has shape [time, batch, 1]
        # We want to compute mean & std only for the "train" portion
        # i.e. time steps [0 .. single_eval_pos-1]
        y_train_part = Y[:single_eval_pos]  # shape: [single_eval_pos, batch, 1]

        y_mean = y_train_part.mean(dim=0, keepdim=True)
        y_std = y_train_part.std(dim=0, keepdim=True) + 1e-9  # add epsilon
        self.y_mean = y_mean
        self.y_std = y_std

        # Now standardize the train portion
        Y_eval = Y.clone().to(self.device_)
        Y_eval[:single_eval_pos, :, :] = (Y[:single_eval_pos, :, :] - y_mean) / y_std

        # Operate under the mode where we can get gradients
        output_dict = self.model_(
            None,  # style is None
            X,  # shape: (time, batch, features)
            Y_eval,  # shape: (time, batch)
            single_eval_pos=single_eval_pos,
            only_return_standard_out=False,
        )
        return output_dict

    def compute_loss(self, logits: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        If your loaded model is a regression distribution model using bar distributions,
        you can compute the negative log-likelihood (NLL) via self.bardist_.
        """
        Y = Y.to(self.device_)
        logits = logits.to(self.device_)
        self.bardist_ = self.bardist_.to(self.device_)
        loss = self.bardist_(logits, Y)
        return loss.mean()

    def predict_mean(self, logits: torch.Tensor) -> torch.Tensor:
        """
        For the distribution-based model, return the predicted mean from the bar distribution.
        """
        mean_pred = self.bardist_.mean(logits)
        mean_pred = mean_pred * self.y_std.squeeze(-1) + self.y_mean.squeeze(-1)
        return mean_pred

    def predict_variance(self, logits: torch.Tensor) -> torch.Tensor:
        """
        For the distribution-based model, return the predicted mean from the bar distribution.
        """
        var_pred = self.bardist_.variance(logits)
        return var_pred

    def predict_ei(self, logits: torch.Tensor, best_f: torch.Tensor) -> torch.Tensor:
        ei_pred = self.bardist_.ei(logits, best_f)
        return ei_pred

    ########################################################################
    # These internal helpers may reference your project's logic:
    ########################################################################

    def _infer_random_state(self, random_state):
        """
        Example helper that returns (static_seed, rng).
        """
        if random_state is None:
            random_state = 0
        rng = np.random.default_rng(random_state)
        return random_state, rng

    def _infer_device_and_type(self, device):
        """
        Picks device. E.g. "auto" => cuda if available, else cpu.
        """
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _determine_precision(self, inference_precision, device):
        """
        Returns (use_autocast, forced_dtype, byte_size).
        Adjust if you have your own logic for half-precision vs float32, etc.
        """
        if inference_precision == "auto":
            # default: use mixed-precision if CUDA, else float32
            if device.type == "cuda":
                return (True, None, 2)
            else:
                return (False, None, 4)
        elif inference_precision == "float32":
            return (False, torch.float32, 4)
        elif inference_precision == "float16":
            return (False, torch.float16, 2)
        return (False, None, 4)

    def _initialize_tabpfn_model(self, model_path, which, fit_mode, static_seed):
        # This is a simplified version of the initialize_tabpfn_model function

        return initialize_tabpfn_model(
            model_path=model_path,
            which=which,
            fit_mode=fit_mode,
            static_seed=static_seed,
        )


def GITBO(
    Function,
    SEED: int,
    Trail_N: int = 0,
    N_iterations: int = 100,
    Acquisition: str = "ThompsonSampling",
    INITIAL_DIR: str = None,
    SAVE_DIR: str = None,
    N_PENDING: int = 5000,
    N_CANDIDATES: int = 1,
    DEVICE: str = "cpu",
    GPU_DEVICE: str = "cuda:0",
    GI_SUBSPACE=False,
    rank_r=1,
    scale=1.0,
):
    """
    Bayesian Optimization using TabPFN v2 as surrogate model.

    Args:
        Function: Objective function to optimize
        SEED: Random seed for reproducibility
        N_iterations: Number of BO iterations
        Acquisition: Type of acquisition function
        N_PENDING: Number of candidate points to evaluate acquisition on
        N_CANDIDATES: Number of points to evaluate per iteration
        PREPROCESS: Whether to use preprocessing pipeline

    Returns:
        tuple: (evaluated_points, optimization_history)
    """
    print(f"Compute Setting: {DEVICE}")
    print(f"GI_SUBSPACE: {GI_SUBSPACE}, Acquisition: {Acquisition}")
    tkwargs = {"device": torch.device(DEVICE), "dtype": torch.float32}

    if GI_SUBSPACE:
        rank_r = rank_r
        scale = scale
        INIT_SCALE = scale
    else:
        rank_r = None
        scale = None
        INIT_SCALE = None
    # with torch.no_grad():
    # Set random seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    DIM = Function.dim

    # Initialize training data
    trained_X = (
        torch.load(f"{INITIAL_DIR}/_trial_{Trail_N}.pt")
        if INITIAL_DIR
        else torch.rand(50, DIM, **tkwargs)
    )
    N_INIT = trained_X.shape[0]

    # Initial evaluation
    GX, trained_Y = Function.evaluate(trained_X)

    # Move tensors to the specified device
    trained_X = trained_X.to(**tkwargs)
    trained_Y = trained_Y.to(**tkwargs)
    if GX is not None:
        GX = GX.to(**tkwargs)

    # Initialize PREV_Xpen and PREV_Eval
    PREV_Xpen = trained_X
    PREV_Eval = trained_Y

    # Pre-allocate memory for all iterations - create tensors directly on the correct device
    INIT_LOC = trained_X.shape[0]
    trained_X = torch.cat(
        [trained_X, torch.zeros(N_iterations * N_CANDIDATES, DIM, **tkwargs)]
    )
    trained_Y = torch.cat(
        [trained_Y, torch.zeros(N_iterations * N_CANDIDATES, 1, **tkwargs)]
    )
    if GX is not None:
        GX = torch.cat(
            [GX, torch.zeros(N_iterations * N_CANDIDATES, GX.shape[1], **tkwargs)]
        )
    ITER_IND_LOC = INIT_LOC

    # Initialize tracking arrays
    TIME_ARR = torch.zeros(N_iterations, **tkwargs)
    MAX_ARR = torch.zeros(N_iterations, **tkwargs)
    TOTAL_TIME = 0

    # Setup for trust region methods
    sobol_engine = SobolEngine(Function.dim, scramble=True)
    if "TR" in Acquisition:
        TR_LB_List = torch.zeros(N_iterations, DIM, **tkwargs)
        TR_UB_List = torch.zeros(N_iterations, DIM, **tkwargs)
        batch_size = N_CANDIDATES
        state = Unified_TS_State(Function.dim, batch_size=batch_size)
        weights = torch.ones(1, Function.dim, **tkwargs)
    else:
        TR_LB_List = TR_UB_List = state = weights = batch_size = None

    grad_est = None

    # Main optimization loop
    for iter_ in range(N_iterations):

        # Start the timer
        start_time = time.monotonic()

        # Get the Xpen and compute the acquisition values
        tr_lb, tr_ub, X_pen = get_Xpen(
            Function,
            Acquisition,
            sobol_engine,
            N_PENDING,
            N_CANDIDATES,
            DIM,
            state,
            weights,
            trained_X[:ITER_IND_LOC, :],
            trained_Y[:ITER_IND_LOC, :],
            GX[:ITER_IND_LOC] if GX is not None else None,
            rank_r,
            scale,
            trained_X[:ITER_IND_LOC, :],
            trained_Y[:ITER_IND_LOC, :],
            GI_SUBSPACE,
            grad_est,
            tkwargs,
        )
        ACQ, CONS, grad_est = compute_acquisition_values(
            Acquisition,
            DIM,
            sobol_engine,
            N_PENDING,
            N_CANDIDATES,
            None,
            trained_X[:ITER_IND_LOC, :],
            trained_Y[:ITER_IND_LOC, :],
            GX[:ITER_IND_LOC] if GX is not None else None,
            X_pen.to(**tkwargs),
            Function,
            GPU_DEVICE,
            tr_lb,
            tr_ub,
            state,
            weights,
            batch_size,
            GPU_DEVICE,
            tkwargs,
        )

        if ACQ is None and CONS is None:
            print("PFN output NANs")
            return trained_X, MAX_ARR

        X_pen = X_pen.permute(1, 0, 2)
        ACQ = ACQ.permute(1, 0)

        best_candidate_indices = torch.argmax(ACQ, dim=1)

        num_batches = X_pen.shape[0]
        row_indices = torch.arange(num_batches)

        best_candidate = X_pen[
            row_indices.to("cpu"), best_candidate_indices.to("cpu"), :
        ].detach()

        # Update timings
        TOTAL_TIME += time.monotonic() - start_time
        TIME_ARR[iter_] = TOTAL_TIME

        # Evaluate new points
        ITER_IND_LOC = INIT_LOC + N_CANDIDATES
        trained_X[INIT_LOC:ITER_IND_LOC, :] = best_candidate
        g_, y_ = Function.evaluate(best_candidate)
        trained_Y[INIT_LOC:ITER_IND_LOC, :] = y_
        if GX is not None:
            GX[INIT_LOC:ITER_IND_LOC, :] = g_
        INIT_LOC = ITER_IND_LOC

        # Update trust region if needed
        if "TR" in Acquisition:
            state = update_ts_state(state, trained_Y, GX)
            TR_LB_List[iter_, :] = tr_lb
            TR_UB_List[iter_, :] = tr_ub

        # Track best value
        sorted_ind = get_sorted_indices(
            trained_Y[:ITER_IND_LOC, :],
            GX[:ITER_IND_LOC, :] if GX is not None else None,
        )
        MAX_ARR[iter_] = trained_Y[:ITER_IND_LOC, :][sorted_ind[0], :].cpu().detach()

        # Print progress
        if SAVE_DIR is None:
            print(f"GITBO Iter {iter_}) Opt: {MAX_ARR[iter_]} Time: {TOTAL_TIME}")
            if GX is not None:
                print(f"Feasible solutions: {(GX <= 0).all(dim=1).any()}")

    if SAVE_DIR == None:
        print("Done GITBO")
    else:

        save_folder = (
            f"{SAVE_DIR}/GITBO_{Acquisition}/{Function.__class__.__name__}_DIM_{DIM}/"
        )

        # Setup output file Directory
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        now = datetime.now()  # Get current date and time
        timestamp = now.strftime("%Y%m%d_%H%M%S")  # Format it as YYYYMMDD_HHMMSS
        save_file_name = f"{save_folder}/_trial_{Trail_N}_rankR_{rank_r}_iters_{N_iterations}_pend_{N_PENDING}.pt"

        torch.save(
            {
                "TIME_ARR": TIME_ARR.cpu().detach(),
                "MAX_ARR": MAX_ARR.cpu().detach(),
                "trained_X": trained_X.cpu().detach(),
            },
            save_file_name,
        )

        print(f"Save GITBO file at {save_file_name}")

    return trained_X, MAX_ARR


def compute_acquisition_values(
    Acquisition: str,
    DIM: int,
    sobol_engine,
    N_PENDING: int,
    N_CANDIDATES: int,
    PFN_MODEL,  # Not used in v2 but kept for interface compatibility
    trained_X: torch.Tensor,
    trained_Y: torch.Tensor,
    GX: torch.Tensor,
    X_pen: torch.Tensor,
    Function,
    DEVICE: str,
    tr_lb=None,
    tr_ub=None,
    state=None,
    weights=None,
    batch_size=None,
    GPU_DEVICE="cuda:0",
    tkwargs=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute acquisition values using TabPFN v2 model, with memory‑efficient gradient estimation.
    Returns: (acquisition_values [N_PENDING×N_CANDIDATES], constraint_values=None, grad_est [N_PENDING×N_CANDIDATES×DIM])
    """

    # reset stats
    reset_device_memory(torch.device(DEVICE))
    peak_bytes = get_max_memory_allocated(torch.device(DEVICE))

    regressor = VanillaDirectTabPFNRegressor(device=GPU_DEVICE)
    single_eval_pos = trained_X.shape[0]

    # report
    peak_bytes = torch.cuda.max_memory_allocated()
    # print(f"Peak GPU memory used: {peak_bytes/2**20:.2f} MiB")

    # --- build the full X and Y concatenation ---
    X_train = trained_X.unsqueeze(1).expand(
        -1, N_CANDIDATES, -1
    )  # [single_eval_pos, N_CANDIDATES, DIM]
    X_full = torch.cat(
        [X_train, X_pen], dim=0
    )  # [(single_eval_pos+N_PENDING), N_CANDIDATES, DIM]

    Y_pad = torch.zeros(N_PENDING, 1, **tkwargs)  # [N_PENDING, 1]
    Y_full = torch.cat([trained_Y, Y_pad], dim=0).unsqueeze(
        1
    )  # [(single_eval_pos+N_PENDING), 1, 1]
    Y_full = Y_full.expand(
        -1, N_CANDIDATES, -1
    )  # [(single_eval_pos+N_PENDING), N_CANDIDATES, 1]

    # --- split off the candidate portion for gradient tracking ---
    X_train_det = X_full[:single_eval_pos].detach()
    X_cand = (
        X_full[single_eval_pos:].clone().requires_grad_()
    )  # [N_PENDING, N_CANDIDATES, DIM]
    X_concat = torch.cat([X_train_det, X_cand], dim=0).to(GPU_DEVICE)

    # --- get devices ---
    amp_dtype = tkwargs["dtype"]
    amp_device = GPU_DEVICE
    peak_bytes = torch.cuda.max_memory_allocated()

    if Acquisition in ["EI", "TR_EI"]:
        # --- forward + EI under autocast for mixed precision ---
        with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
            out = regressor.forward(X_concat, Y_full, single_eval_pos)
            logits = out["standard"]
            acq = regressor.predict_ei(
                logits, trained_Y.max()
            )  # [ (single_eval_pos+N_PENDING), N_CANDIDATES ]
        # take only the candidate rows
        EI = acq[single_eval_pos:]  # [N_PENDING, N_CANDIDATES]

        # --- gradient only wrt X_cand ---
        (grad_cand,) = torch.autograd.grad(
            EI.sum(), X_cand, retain_graph=False, create_graph=False
        )
        grad_est = -grad_cand.view(N_PENDING, N_CANDIDATES, DIM).detach()

        return EI.to(**tkwargs), None, grad_est.to(**tkwargs)

    elif Acquisition in ["ThompsonSampling", "TR_TS"]:
        # --- forward + mean/variance under autocast ---
        with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
            out = regressor.forward(X_concat, Y_full, single_eval_pos)
            logits = out["standard"]
            output_mean = regressor.predict_mean(
                logits
            )  # [ (single_eval_pos+N_PENDING), N_CANDIDATES ]
            output_variance = regressor.predict_variance(logits)  # same shape

        # print('forward pass done')

        mu_cand = output_mean[single_eval_pos:]  # [N_PENDING, N_CANDIDATES]
        var_cand = output_variance[single_eval_pos:]
        std_cand = torch.clamp(var_cand, min=1e-8).sqrt()

        # --- sample if requested ---
        sample_count = 512
        mu_expanded = mu_cand.unsqueeze(-1).expand(-1, -1, sample_count)
        std_expanded = std_cand.unsqueeze(-1).expand(-1, -1, sample_count)
        sampled_y = torch.normal(mu_expanded, std_expanded).mean(dim=-1)

        # --- gradient only wrt X_cand, using the mean as loss ---
        loss = mu_cand.sum()
        (grad_cand,) = torch.autograd.grad(
            loss, X_cand, retain_graph=False, create_graph=False
        )
        grad_est = -grad_cand.view(N_PENDING, N_CANDIDATES, DIM).detach()

        # print('grad done')

        return sampled_y.to(**tkwargs), None, grad_est.to(**tkwargs)

    else:
        raise ValueError(f"Unknown acquisition type: {Acquisition}")


def get_Xpen(
    Function,
    Acquisition,
    sobol_engine,
    N_PENDING,
    N_CANDIDATES,
    DIM,
    state,
    weights,
    trained_X,
    trained_Y,
    GX=None,
    rank_r=None,
    scale=None,
    PREV_Xpen=None,
    PREV_Eval=None,
    GI_SUBSPACE=False,
    grad_est=None,
    tkwargs=None,
):

    if "TR" in Acquisition:
        if rank_r is not None:
            tr_lb, tr_ub, X_pen = generate_batch_PFN(
                state,
                weights,  # REQUIRE WEIGHT DEFINITION
                trained_X,  # Evaluated points on the domain [0, 1]^d
                trained_Y,  # Function values
                n_candidates=None,  # Number of candidates for Thompson sampling
                C=GX,  # Constraint values (optional)
                tkwargs=tkwargs,
            )
            if grad_est is None:
                total_points = N_CANDIDATES * N_PENDING
                X_pen = sobol_engine.draw(total_points)
                X_pen = X_pen.view(N_PENDING, N_CANDIDATES, DIM)
                return tr_lb, tr_ub, X_pen
            else:
                X_pen, _, _ = sample_dominant_subspace(
                    trained_X,
                    trained_Y,
                    DIM,
                    grad_est,
                    sobol_engine,
                    rank_r=rank_r,
                    n_samples=N_PENDING,
                    N_CANDIDATES=N_CANDIDATES,
                    scale=scale,
                    GI_SUBSPACE=GI_SUBSPACE,
                    tkwargs=tkwargs,
                )
                return tr_lb, tr_ub, X_pen
        else:
            tr_lb, tr_ub, X_pen = generate_batch_PFN(
                state,
                weights,  # REQUIRE WEIGHT DEFINITION
                trained_X,  # Evaluated points on the domain [0, 1]^d
                trained_Y,  # Function values
                n_candidates=None,  # Number of candidates for Thompson sampling
                C=GX,  # Constraint values (optional)
                tkwargs=tkwargs,
            )

            X_pen = X_pen.view(N_PENDING, N_CANDIDATES, DIM)
            # print("X_pen.shape", X_pen.shape)
            return tr_lb, tr_ub, X_pen

    if grad_est is None:
        #   standard_bounds = torch.zeros(2, DIM, **tkwargs)
        #   standard_bounds[1] = 1

        # change it to batch (April 17)
        total_points = N_CANDIDATES * N_PENDING
        X_pen = sobol_engine.draw(total_points)
        X_pen = X_pen.view(N_PENDING, N_CANDIDATES, DIM)
        return None, None, X_pen
    if rank_r is not None:
        X_pen, _, _ = sample_dominant_subspace(
            trained_X,
            trained_Y,
            DIM,
            grad_est,
            sobol_engine,
            rank_r=rank_r,
            n_samples=N_PENDING,
            N_CANDIDATES=N_CANDIDATES,
            scale=scale,
            GI_SUBSPACE=GI_SUBSPACE,
            tkwargs=tkwargs,
        )
        # print(X_pen.shape)
        return None, None, X_pen
    else:
        # Sample pending points
        total_points = N_CANDIDATES * N_PENDING
        X_pen = sobol_engine.draw(total_points)
        X_pen = X_pen.view(N_PENDING, N_CANDIDATES, DIM)
        return None, None, X_pen


def sample_dominant_subspace(
    x,
    y,
    DIM,
    grad_vals,
    SEED,
    rank_r=1,
    n_samples=100,
    N_CANDIDATES=1,
    scale=1.0,
    GI_SUBSPACE=False,
    new_origin=None,
    tkwargs=None,
):
    """
    Sample points in the dominant subspace given input data x and Function.

    Args:
        x: Input points tensor of shape (n, d) where n is number of points and d is dimension
        Function: Function object with evaluate method that can handle batched inputs
        rank_r: Number of dominant directions to keep (default=1)
        n_samples: Number of samples to generate in subspace (default=100)
        scale: Scale factor for sampling range (default=1.0)

    Returns:
        samples: New points sampled in dominant subspace
        U_r: Top r eigenvectors
        eigenvals: Eigenvalues sorted in descending order
    """

    # rng = np.random.RandomState(SEED)

    # Keep as torch tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, **tkwargs)

    # Extract dimensions
    n_points, batch_size, feat_dim = grad_vals.shape

    # Convert to numpy for eigendecomposition
    grad_vals_np = grad_vals.cpu().detach().numpy()
    x_np = x.cpu().detach().numpy()

    # Initialize containers for H_est matrices
    X_pen = torch.zeros((n_samples, batch_size, feat_dim))

    # Calculate H_est for each batch in a vectorized way
    for b in range(batch_size):
        # Get dominant subspace for this batch
        H_est = (grad_vals_np[:, b, :].T @ grad_vals_np[:, b, :]) / n_points

        eigvals, eigvecs = np.linalg.eigh(H_est)
        idx_sorted = np.argsort(eigvals)[::-1]
        eigenvals = eigvals[idx_sorted]
        eigenvecs = eigvecs[:, idx_sorted]

        # Get top r eigenvectors
        U_r = eigenvecs[:, :rank_r]

        # Get mean of input points as origin
        if new_origin is None:
            origin = x_np.mean(axis=0)
        else:
            origin = new_origin.mean(axis=0)

        alpha = np.random.uniform(-scale, scale, size=(n_samples, rank_r))

        # Map back to original space: x = origin + U_r * alpha
        samples = origin + (alpha @ U_r.T)

        # Convert samples back to torch tensor
        samples = torch.tensor(samples, **tkwargs)

        # Clamp to [0, 1] (design space min max)
        samples = torch.clamp(samples, 0.0, 1.0).to(**tkwargs)

        X_pen[:, b, :] = samples.detach().to(**tkwargs)

    return X_pen, U_r, eigenvals

def compute_acquisition_values(
    Acquisition: str,
    DIM: int,
    sobol_engine,
    N_PENDING: int,
    N_CANDIDATES: int,
    PFN_MODEL,  # Not used in v2 but kept for interface compatibility
    trained_X: torch.Tensor,
    trained_Y: torch.Tensor,
    GX: torch.Tensor,
    X_pen: torch.Tensor,
    Function,
    DEVICE: str,
    tr_lb=None,
    tr_ub=None,
    state=None,
    weights=None,
    batch_size=None,
    GPU_DEVICE="cuda:0",
    tkwargs=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute acquisition values using TabPFN v2 model, with memory‑efficient gradient estimation.
    Returns: (acquisition_values [N_PENDING×N_CANDIDATES], constraint_values=None, grad_est [N_PENDING×N_CANDIDATES×DIM])
    """
    
    # reset stats
    reset_device_memory(torch.device(DEVICE))
    peak_bytes = get_max_memory_allocated(torch.device(DEVICE))

    regressor = VanillaDirectTabPFNRegressor(device=GPU_DEVICE)
    single_eval_pos = trained_X.shape[0]
    
    # report
    peak_bytes = torch.cuda.max_memory_allocated()
    # print(f"Peak GPU memory used: {peak_bytes/2**20:.2f} MiB")

    # --- build the full X and Y concatenation ---
    X_train = trained_X.unsqueeze(1).expand(-1, N_CANDIDATES, -1)           # [single_eval_pos, N_CANDIDATES, DIM]
    X_full  = torch.cat([X_train, X_pen], dim=0)                            # [(single_eval_pos+N_PENDING), N_CANDIDATES, DIM]

    Y_pad   = torch.zeros(N_PENDING, 1, **tkwargs)                          # [N_PENDING, 1]
    Y_full  = torch.cat([trained_Y, Y_pad], dim=0).unsqueeze(1)             # [(single_eval_pos+N_PENDING), 1, 1]
    Y_full  = Y_full.expand(-1, N_CANDIDATES, -1)                           # [(single_eval_pos+N_PENDING), N_CANDIDATES, 1]

    # --- split off the candidate portion for gradient tracking ---
    X_train_det = X_full[:single_eval_pos].detach()
    X_cand      = X_full[single_eval_pos:].clone().requires_grad_()         # [N_PENDING, N_CANDIDATES, DIM]
    X_concat    = torch.cat([X_train_det, X_cand], dim=0).to(GPU_DEVICE)

    # --- get devices ---
    amp_dtype = tkwargs["dtype"]
    amp_device = GPU_DEVICE
    peak_bytes = torch.cuda.max_memory_allocated()


    if Acquisition in ['EI', 'TR_EI']:
        # --- forward + EI under autocast for mixed precision ---
        with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
            out    = regressor.forward(X_concat, Y_full, single_eval_pos)
            logits = out["standard"]
            acq    = regressor.predict_ei(logits, trained_Y.max())            # [ (single_eval_pos+N_PENDING), N_CANDIDATES ]
        # take only the candidate rows
        EI      = acq[single_eval_pos:]                                      # [N_PENDING, N_CANDIDATES]

        # --- gradient only wrt X_cand ---
        grad_cand, = torch.autograd.grad(EI.sum(), X_cand, retain_graph=False, create_graph=False)
        grad_est   = -grad_cand.view(N_PENDING, N_CANDIDATES, DIM).detach()

        return EI.to(**tkwargs), None, grad_est.to(**tkwargs)

    elif Acquisition in ['ThompsonSampling', 'TR_TS']:
        # --- forward + mean/variance under autocast ---
        with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
            out             = regressor.forward(X_concat, Y_full, single_eval_pos)
            logits          = out["standard"]
            output_mean     = regressor.predict_mean(logits)                   # [ (single_eval_pos+N_PENDING), N_CANDIDATES ]
            output_variance = regressor.predict_variance(logits)               # same shape

        # print('forward pass done')

        mu_cand  = output_mean[single_eval_pos:]                              # [N_PENDING, N_CANDIDATES]
        var_cand = output_variance[single_eval_pos:]
        std_cand = torch.clamp(var_cand, min=1e-8).sqrt()

        # --- sample if requested ---
        sample_count = 512
        mu_expanded  = mu_cand.unsqueeze(-1).expand(-1, -1, sample_count)
        std_expanded = std_cand.unsqueeze(-1).expand(-1, -1, sample_count)
        sampled_y    = torch.normal(mu_expanded, std_expanded).mean(dim=-1)

        # --- gradient only wrt X_cand, using the mean as loss ---
        loss       = mu_cand.sum()
        grad_cand, = torch.autograd.grad(loss, X_cand, retain_graph=False, create_graph=False)
        grad_est   = -grad_cand.view(N_PENDING, N_CANDIDATES, DIM).detach()
        
        # print('grad done')

        return sampled_y.to(**tkwargs), None, grad_est.to(**tkwargs)

    else:
        raise ValueError(f"Unknown acquisition type: {Acquisition}")

def run_multiple_trials(args, n_trials=5):
    """Run GITBO multiple times and collect results"""
    all_results = []
    all_memory = []

    for trial in range(n_trials):
        print(f"\n{'='*50}")
        print(f"Running Trial {trial + 1}/{n_trials}")
        print(f"{'='*50}\n")

        # Generate a random seed for this trial
        random_seed = np.random.randint(0, 2**31)
        print(f"Trial {trial + 1} random_seed: {random_seed}")

        # # Clear GPU memory before each trial
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        #     torch.cuda.reset_peak_memory_stats()
        reset_device_memory(torch.device(args.GPU_DEVICE))

        # Track memory usage
        memory_usage = []

        # Modified GITBO call to track memory
        def memory_tracking_gitbo(*args, **kwargs):
            # Store original function
            global compute_acquisition_values
            original_compute = compute_acquisition_values

            # Create wrapper to track memory
            def compute_with_memory_tracking(*compute_args, **compute_kwargs):
                result = original_compute(*compute_args, **compute_kwargs)
                if torch.cuda.is_available():
                    # memory_usage.append(
                    #     torch.cuda.max_memory_allocated() / (1024**3)
                    # )  # GB
                    memory_usage.append(get_max_memory_allocated(torch.device(args.GPU_DEVICE)) / (1024**3))
                return result

            # Temporarily replace function
            compute_acquisition_values = compute_with_memory_tracking

            # Run GITBO
            result = GITBO(*args, **kwargs)

            # Restore original function
            compute_acquisition_values = original_compute

            return result

        # Run GITBO with memory tracking
        xx_result, maxx = memory_tracking_gitbo(
            Function,
            random_seed,
            Trail_N=trial,
            N_iterations=args.ITER,
            Acquisition=args.ACQ,
            INITIAL_DIR=args.INITIAL_DIR,
            SAVE_DIR=None,  # Disable individual saving
            N_PENDING=args.N_PENDING,
            N_CANDIDATES=args.N_SAMPLE,
            DEVICE=args.DEVICE,
            GPU_DEVICE=args.GPU_DEVICE,
            GI_SUBSPACE=args.GI_SUBSPACE,
            rank_r=args.RANK_R,
            scale=args.SAMPLE_SCALE,
        )

        all_results.append(maxx.cpu().numpy())
        all_memory.append(memory_usage)

    return all_results, all_memory


def plot_convergence_with_confidence(all_results, args, save_path=None):
    """Plot convergence with confidence intervals using fill_between"""
    plt.figure(figsize=(10, 6))

    # Convert to numpy array for easier manipulation
    results_array = np.array(all_results)
    n_iterations = results_array.shape[1]
    iterations = np.arange(1, n_iterations + 1)

    # Calculate statistics
    mean_results = np.mean(results_array, axis=0)
    std_results = np.std(results_array, axis=0)

    # Calculate confidence intervals (95%)
    confidence_multiplier = 1.96  # for 95% CI
    ci_lower = mean_results - confidence_multiplier * std_results / np.sqrt(
        len(all_results)
    )
    ci_upper = mean_results + confidence_multiplier * std_results / np.sqrt(
        len(all_results)
    )

    # Plot mean line
    plt.plot(iterations, mean_results, "b-", linewidth=2, label="Mean")

    # Plot confidence interval
    plt.fill_between(
        iterations, ci_lower, ci_upper, alpha=0.3, color="blue", label="95% CI"
    )

    # Plot individual trials with transparency
    for i, result in enumerate(all_results):
        plt.plot(iterations, result, "gray", alpha=0.3, linewidth=0.5)

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Best Value Found", fontsize=12)
    plt.title(
        f"{args.FUNC_NAME} - {args.ACQ} (DIM={args.DIM}, {len(all_results)} trials)",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_memory_usage(all_memory, args, save_path=None):
    """Plot GPU memory usage across iterations"""
    plt.figure(figsize=(10, 6))

    # Find the maximum length among all memory traces
    max_len = max(len(mem) for mem in all_memory)

    # Pad shorter sequences with their last value
    padded_memory = []
    for mem in all_memory:
        if len(mem) < max_len:
            padded = mem + [mem[-1]] * (max_len - len(mem))
        else:
            padded = mem
        padded_memory.append(padded)

    # Convert to numpy array
    memory_array = np.array(padded_memory)
    iterations = np.arange(1, max_len + 1)

    # Calculate statistics
    mean_memory = np.mean(memory_array, axis=0)
    std_memory = np.std(memory_array, axis=0)

    # Plot mean line
    plt.plot(iterations, mean_memory, "r-", linewidth=2, label="Mean Memory")

    # Plot min/max range
    plt.fill_between(
        iterations,
        np.min(memory_array, axis=0),
        np.max(memory_array, axis=0),
        alpha=0.3,
        color="red",
        label="Min/Max Range",
    )

    # Plot individual trials
    for i, mem in enumerate(padded_memory):
        plt.plot(iterations[: len(mem)], mem, "gray", alpha=0.3, linewidth=0.5)

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("GPU Memory Usage (GB)", fontsize=12)
    plt.title(
        f"GPU Memory Consumption - {args.FUNC_NAME} (DIM={args.DIM})", fontsize=14
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


class RangeException(Exception):
    pass


class DimensionException(Exception):
    pass


class BenchmarkProblem:
    """
    Base class for Bayesian Optimization benchmark problems.
    """

    def __init__(
        self,
        dim=1,
        num_obj=1,
        num_cons=0,
        bounds=None,
        optimizers=[[]],
        optimum=[[]],
        ref_point=None,
        to_verify=True,
        out_type=torch,
        tags=[],
        is_mixed=False,
        to_print_Xscaled=False,
        is_constrained=False,
        flag="",
    ):
        self.dim = dim
        self.num_obj = num_obj
        self.num_cons = num_cons
        self.bounds = bounds
        self.optimizers = optimizers
        self.optimum = optimum
        self.ref_point = ref_point
        self.to_verify = to_verify
        self.out_type = out_type
        self.tags = tags
        self.is_mixed = is_mixed
        self.to_print_Xscaled = to_print_Xscaled
        self.is_constrained = is_constrained
        self.flag = flag

    def scale(self, X, to_verify):
        """
        (Optionally) verifies that X is in the correct range [0, 1] and has the correct dimensions.
        Converts X to a torch.Tensor if necessary and scales X to the problem's bounds.

        Parameters:
            X (array, np.array, or torch.Tensor): data in range of [0, 1]

        Returns:
            X (Torch.tensor): data scaled to bounds

        """

        if not torch.is_tensor(X):
            X = torch.tensor(X)

        if self.to_verify:
            if X.size(1) != self.dim:
                raise DimensionException("Incorrect X dimensions.")
            if torch.max(X) > 1 or torch.min(X) < 0:
                raise RangeException("Incorrect X range: must be [0, 1].")

        if not torch.is_tensor(self.bounds):
            self.bounds = torch.tensor(self.bounds)

        self.bounds = self.bounds.to(X.device)

        X_scaled = torch.add(
            torch.mul(X, (self.bounds[:, 1] - self.bounds[:, 0])), self.bounds[:, 0]
        )

        return X_scaled

    def cont_to_disc(self, x, disc_values):
        # Convert continuous value to discrete value
        # Input:
        #   x: continuous value in [0, 1]
        #   disc_values: discrete values
        # Output: discrete value
        idx = torch.floor(x * len(disc_values)).long()
        return disc_values[torch.clamp(idx, 0, len(disc_values) - 1)]


class Ackley(BenchmarkProblem):
    r"""
    Eriksson D, Poloczek M (2021) Scalable constrained bayesian optimization.
    In: International Conference on Artificial Intelligence and Statistics, PMLR, pp 730–738
    """

    # N-D objective, 2 constraints, X = n-by-dim

    tags = {"single_objective", "constrained", "continuous", "ND", "extra_imports"}

    def __init__(self, dim=2, is_constrained=False):
        super().__init__(
            dim,
            num_obj=1,
            num_cons=2,
            optimizers=[[0] * dim],
            optimum=[[0]],
            bounds=[[-5, 10]],
            is_constrained=is_constrained,
        )

    def evaluate(self, X, to_verify=True):
        from botorch.test_functions import Ackley as Ackley_imported

        device = torch.device(X.device)
        dtype = torch.float32 if device.type == "mps" else torch.float64

        X = super().scale(X, to_verify)
        n = X.size(0)

        gx = torch.zeros((n, self.num_cons))

        fun = Ackley_imported(dim=self.dim, negate=True).to(dtype=dtype, device=device)
        fun.bounds[0, :].fill_(-5)
        fun.bounds[1, :].fill_(10)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        gx[:, 0] = torch.sum(X, 1)
        gx[:, 1] = torch.norm(X, p=2, dim=1) - 5

        if self.is_constrained:
            return gx, fx
        else:
            return None, fx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process experiment parameters Nature_PFN"
    )
    parser.add_argument("--ITER", type=int, default=5, help="Number of iterations")
    parser.add_argument("--DIM", type=int, default=149, help="Number of dim")
    parser.add_argument(
        "--INITIAL_DIR", type=str, default=None, help="For loading initial data"
    )
    parser.add_argument("--SAVE_DIR", type=str, default=None, help="For saving data")
    parser.add_argument("--FUNC_NAME", type=str, default="Ackley", help="Function")
    parser.add_argument("--ACQ", type=str, default="ThompsonSampling", help="ACQ")
    parser.add_argument("--N_SAMPLE", type=int, default=6, help="N_SAMPLE")
    parser.add_argument("--N_PENDING", type=int, default=1000, help="N_PENDING")
    parser.add_argument("--CONSTRAINED", action="store_true", help="CONSTRAINED")
    parser.add_argument("--TRIAL", type=int, default=0, help="TRIAL")
    parser.add_argument("--DEVICE", type=str, default="mps", help="DEVICE")
    parser.add_argument("--GPU_DEVICE", type=str, default="mps", help="GPU_DEVICE")
    parser.add_argument("--RANK_R", type=int, default=15, help="RANK_R")
    parser.add_argument("--SAMPLE_SCALE", type=float, default=0.2, help="SAMPLE_SCALE")
    parser.add_argument("--GI_SUBSPACE", type=bool, default=True, help="GI_SUBSPACE")
    parser.add_argument(
        "--N_TRIALS", type=int, default=5, help="Number of trials to run"
    )

    args = parser.parse_args()

    print(f"INITIAL_DIR: {args.INITIAL_DIR}, SAVE_DIR: {args.SAVE_DIR}")

    # Assigned Function
    Function = Ackley(dim=args.DIM, is_constrained=args.CONSTRAINED)

    # Run multiple trials
    all_results, all_memory = run_multiple_trials(args, n_trials=args.N_TRIALS)

    # Create save directory if specified
    if args.SAVE_DIR:
        os.makedirs(args.SAVE_DIR, exist_ok=True)
        convergence_path = os.path.join(
            args.SAVE_DIR, f"{args.FUNC_NAME}_convergence_plot.png"
        )
        memory_path = os.path.join(args.SAVE_DIR, f"{args.FUNC_NAME}_memory_plot.png")
    else:
        convergence_path = None
        memory_path = None

    # Plot results
    plot_convergence_with_confidence(all_results, args, save_path=convergence_path)
    plot_memory_usage(all_memory, args, save_path=memory_path)

    # Save raw results if save directory is specified
    if args.SAVE_DIR:
        results_path = os.path.join(args.SAVE_DIR, f"{args.FUNC_NAME}_results.npz")
        np.savez(results_path, results=all_results, memory=all_memory, args=vars(args))
        print(f"\nResults saved to {results_path}")

    print(f"\n{args.FUNC_NAME} GITBO Multiple Trials Done\n")
