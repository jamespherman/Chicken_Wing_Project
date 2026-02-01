"""
Pupil-Luminance Temporal Kernel Module

Models the Pupillary Light Reflex (PLR) as convolution of luminance with an
Erlang gamma impulse response function. Supports per-subject kernel fitting
to account for individual differences in PLR dynamics.

Based on:
- Hoeks & Levelt (1993): Erlang gamma function for PLR modeling
- Knapen et al. (2016): Pupil Response Function estimation

Author: Claude Code
Date: 2026-02-01
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress
import logging

logger = logging.getLogger(__name__)


@dataclass
class PLRKernelParams:
    """Parameters for the Pupillary Light Reflex kernel."""

    t_max_ms: float = 512.0      # Time to peak response (ms)
    n: float = 10.1              # Shape parameter
    duration_ms: float = 2000.0  # Kernel duration (ms)
    is_fitted: bool = False      # Whether params were fitted or canonical
    fit_r_squared: float = 0.0   # R² achieved with these params

    # Physiological bounds (class-level constants)
    T_MAX_MIN: float = field(default=300.0, repr=False)
    T_MAX_MAX: float = field(default=1200.0, repr=False)
    N_MIN: float = field(default=3.0, repr=False)
    N_MAX: float = field(default=25.0, repr=False)


class PupilLuminanceKernel:
    """
    Temporal kernel for pupil-luminance regression with per-subject fitting.

    Models the Pupillary Light Reflex (PLR) as convolution of luminance
    with an Erlang gamma impulse response function:

        h(t) = (t/t_max)^n * exp(n * (1 - t/t_max))

    This form ensures the peak occurs at t = t_max.

    Parameters
    ----------
    sampling_rate_hz : float
        Sampling rate of the eye tracker (default: 90 Hz for Tobii Pro Glasses 3)

    Attributes
    ----------
    params : PLRKernelParams
        Current kernel parameters (fitted or canonical)
    kernel : np.ndarray
        The impulse response function array

    Example
    -------
    >>> kernel_model = PupilLuminanceKernel(sampling_rate_hz=90.0)
    >>> params = kernel_model.fit_to_subject(pupil_data, luminance_data)
    >>> results = kernel_model.fit_regression(pupil_data, luminance_data)
    >>> print(f"R² improved from {results['r_squared_instantaneous']:.3f} to {results['r_squared_convolved']:.3f}")
    """

    # Canonical parameters from Hoeks & Levelt (1993)
    CANONICAL_T_MAX = 512.0
    CANONICAL_N = 10.1

    def __init__(self, sampling_rate_hz: float = 90.0):
        self.sampling_rate = sampling_rate_hz
        self.params: Optional[PLRKernelParams] = None
        self.kernel: Optional[np.ndarray] = None
        self._kernel_time_ms: Optional[np.ndarray] = None

    def create_erlang_kernel(self, t_max_ms: float, n: float,
                              duration_ms: float = 2000.0) -> np.ndarray:
        """
        Create normalized Erlang gamma impulse response function.

        The Erlang gamma function models the temporal dynamics of the
        pupillary light reflex:

            h(t) = (t/t_max)^n * exp(n * (1 - t/t_max))

        This parameterization ensures:
        - Peak response occurs at t = t_max
        - h(t_max) = 1 (before normalization)
        - Smooth rise and decay

        Parameters
        ----------
        t_max_ms : float
            Time to peak response in milliseconds
        n : float
            Shape parameter (higher = sharper peak)
        duration_ms : float
            Total duration of the kernel in milliseconds

        Returns
        -------
        np.ndarray
            Normalized kernel (sums to 1)
        """
        dt_ms = 1000.0 / self.sampling_rate
        t = np.arange(0, duration_ms, dt_ms)

        # Store time array for visualization
        self._kernel_time_ms = t

        # Avoid division by zero at t=0
        t_safe = np.maximum(t, 1e-10)
        t_ratio = t_safe / t_max_ms

        # Erlang gamma function
        # Using log-space computation for numerical stability with large n
        log_h = n * np.log(t_ratio) + n * (1 - t_ratio)
        h = np.exp(log_h)

        # Set h[0] = 0 (no instantaneous response)
        h[0] = 0.0

        # Handle any numerical issues
        h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize to sum to 1 (preserves scale in convolution)
        h_sum = np.sum(h)
        if h_sum > 0:
            h = h / h_sum

        return h

    def convolve_luminance(self, luminance: np.ndarray,
                           kernel: np.ndarray = None) -> np.ndarray:
        """
        Convolve luminance signal with PLR kernel.

        Uses causal convolution: only past luminance affects current pupil.
        Handles edge effects with reflection padding at the start.

        Parameters
        ----------
        luminance : np.ndarray
            Frame luminance time series
        kernel : np.ndarray, optional
            Kernel to use. If None, uses self.kernel

        Returns
        -------
        np.ndarray
            Convolved luminance signal (same length as input)
        """
        if kernel is None:
            kernel = self.kernel

        if kernel is None:
            raise ValueError("No kernel available. Call fit_to_subject() first.")

        # Handle NaN in luminance by interpolating
        luminance_clean = luminance.copy()
        nan_mask = np.isnan(luminance_clean)
        if np.any(nan_mask):
            # Linear interpolation for NaN values
            valid_idx = np.where(~nan_mask)[0]
            if len(valid_idx) > 1:
                luminance_clean[nan_mask] = np.interp(
                    np.where(nan_mask)[0],
                    valid_idx,
                    luminance_clean[valid_idx]
                )
            else:
                # If too few valid points, fill with median
                luminance_clean[nan_mask] = np.nanmedian(luminance)

        # Causal convolution: pad at start only
        pad_len = len(kernel)
        padded = np.pad(luminance_clean, (pad_len, 0), mode='reflect')

        # Convolve
        convolved_full = np.convolve(padded, kernel, mode='full')

        # Extract the portion aligned with original signal
        # The convolution output starts being valid after pad_len samples
        convolved = convolved_full[pad_len : pad_len + len(luminance)]

        return convolved

    def _compute_r_squared(self, pupil: np.ndarray, luminance: np.ndarray,
                           t_max_ms: float, n: float) -> float:
        """
        Compute R² for given kernel parameters.

        Parameters
        ----------
        pupil : np.ndarray
            Pupil diameter time series
        luminance : np.ndarray
            Frame luminance time series
        t_max_ms : float
            Time to peak parameter
        n : float
            Shape parameter

        Returns
        -------
        float
            R² value (0 to 1), or 0 if computation fails
        """
        try:
            kernel = self.create_erlang_kernel(t_max_ms, n)
            conv_lum = self.convolve_luminance(luminance, kernel)

            # Remove NaN values
            valid = ~(np.isnan(pupil) | np.isnan(conv_lum))
            n_valid = np.sum(valid)

            if n_valid < 100:
                return 0.0

            # Linear regression
            slope, intercept, r, p, se = linregress(
                conv_lum[valid], pupil[valid]
            )

            return r ** 2

        except Exception as e:
            logger.debug(f"R² computation failed: {e}")
            return 0.0

    def fit_to_subject(self, pupil: np.ndarray, luminance: np.ndarray,
                       use_canonical_fallback: bool = True) -> PLRKernelParams:
        """
        Fit kernel parameters (t_max, n) to maximize R² for this subject.

        Uses bounded L-BFGS-B optimization with multiple starting points
        to avoid local minima. Falls back to canonical parameters if
        fitting fails.

        Parameters
        ----------
        pupil : np.ndarray
            Pupil diameter time series
        luminance : np.ndarray
            Frame luminance time series (same length as pupil)
        use_canonical_fallback : bool
            If True, use canonical params when fitting fails

        Returns
        -------
        PLRKernelParams
            Fitted (or canonical) kernel parameters
        """
        # Objective: minimize negative R²
        def objective(params):
            t_max, n = params
            r2 = self._compute_r_squared(pupil, luminance, t_max, n)
            return -r2

        # Bounds based on physiological constraints
        bounds = [
            (PLRKernelParams.T_MAX_MIN, PLRKernelParams.T_MAX_MAX),
            (PLRKernelParams.N_MIN, PLRKernelParams.N_MAX)
        ]

        # Multiple starting points to avoid local minima
        starting_points = [
            (512.0, 10.1),   # Canonical (Hoeks & Levelt)
            (400.0, 8.0),    # Fast response hypothesis
            (800.0, 12.0),   # Slow response hypothesis
            (600.0, 6.0),    # Broad kernel hypothesis
            (450.0, 15.0),   # Fast, sharp response
            (700.0, 5.0),    # Slow, broad response
        ]

        best_result = None
        best_r2 = -np.inf

        for x0 in starting_points:
            try:
                result = minimize(
                    objective, x0, method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-6}
                )
                r2 = -result.fun
                if r2 > best_r2:
                    best_r2 = r2
                    best_result = result
            except Exception as e:
                logger.debug(f"Optimization from {x0} failed: {e}")
                continue

        # Check if fitting succeeded
        fitting_succeeded = best_result is not None and best_r2 > 0.001

        if fitting_succeeded:
            t_max_fitted, n_fitted = best_result.x
            self.params = PLRKernelParams(
                t_max_ms=float(t_max_fitted),
                n=float(n_fitted),
                is_fitted=True,
                fit_r_squared=float(best_r2)
            )
            logger.info(
                f"Kernel fitted: t_max={t_max_fitted:.1f}ms, n={n_fitted:.2f}, "
                f"R²={best_r2:.4f}"
            )
        elif use_canonical_fallback:
            # Fall back to canonical parameters
            canonical_r2 = self._compute_r_squared(
                pupil, luminance, self.CANONICAL_T_MAX, self.CANONICAL_N
            )
            self.params = PLRKernelParams(
                t_max_ms=self.CANONICAL_T_MAX,
                n=self.CANONICAL_N,
                is_fitted=False,
                fit_r_squared=float(canonical_r2)
            )
            logger.warning(
                f"Kernel fitting failed, using canonical params. R²={canonical_r2:.4f}"
            )
        else:
            raise ValueError("Kernel fitting failed and fallback disabled")

        # Create the kernel with fitted/canonical params
        self.kernel = self.create_erlang_kernel(
            self.params.t_max_ms, self.params.n
        )

        return self.params

    def fit_regression(self, pupil: np.ndarray, luminance: np.ndarray) -> Dict[str, Any]:
        """
        Fit full pupil-luminance regression with current kernel.

        Computes both instantaneous and convolved regressions for comparison.

        Parameters
        ----------
        pupil : np.ndarray
            Pupil diameter time series
        luminance : np.ndarray
            Frame luminance time series

        Returns
        -------
        dict
            Dictionary containing:
            - Instantaneous regression metrics (baseline)
            - Convolved regression metrics
            - Kernel parameters
            - Improvement metrics
            - Arrays for visualization (convolved_luminance, residuals)
        """
        if self.kernel is None:
            raise ValueError("No kernel available. Call fit_to_subject() first.")

        # === INSTANTANEOUS REGRESSION (baseline) ===
        valid_inst = ~(np.isnan(pupil) | np.isnan(luminance))
        n_valid_inst = np.sum(valid_inst)

        if n_valid_inst < 100:
            return self._empty_results()

        slope_inst, intercept_inst, r_inst, p_inst, _ = linregress(
            luminance[valid_inst], pupil[valid_inst]
        )
        r2_inst = r_inst ** 2

        # Instantaneous residuals
        pred_inst = slope_inst * luminance + intercept_inst
        residuals_inst = pupil - pred_inst

        # === CONVOLVED REGRESSION ===
        conv_lum = self.convolve_luminance(luminance)

        valid_conv = ~(np.isnan(pupil) | np.isnan(conv_lum))
        n_valid_conv = np.sum(valid_conv)

        if n_valid_conv < 100:
            return self._empty_results()

        slope_conv, intercept_conv, r_conv, p_conv, _ = linregress(
            conv_lum[valid_conv], pupil[valid_conv]
        )
        r2_conv = r_conv ** 2

        # Convolved residuals
        pred_conv = slope_conv * conv_lum + intercept_conv
        residuals_conv = pupil - pred_conv

        # === IMPROVEMENT METRICS ===
        r2_improvement = r2_conv - r2_inst
        if r2_inst > 0.0001:
            r2_improvement_pct = (r2_improvement / r2_inst) * 100
        else:
            r2_improvement_pct = np.inf if r2_improvement > 0 else 0.0

        return {
            # Instantaneous (baseline)
            'r_squared_instantaneous': float(r2_inst),
            'slope_instantaneous': float(slope_inst),
            'intercept_instantaneous': float(intercept_inst),
            'p_value_instantaneous': float(p_inst),
            'residual_std_instantaneous': float(np.nanstd(residuals_inst)),

            # Convolved (with fitted kernel)
            'r_squared_convolved': float(r2_conv),
            'slope_convolved': float(slope_conv),
            'intercept_convolved': float(intercept_conv),
            'p_value_convolved': float(p_conv),
            'residual_std_convolved': float(np.nanstd(residuals_conv)),
            'residual_mean_convolved': float(np.nanmean(residuals_conv)),

            # Kernel parameters
            'kernel_t_max_ms': float(self.params.t_max_ms),
            'kernel_n': float(self.params.n),
            'kernel_is_fitted': self.params.is_fitted,

            # Improvement metrics
            'r_squared_improvement': float(r2_improvement),
            'r_squared_improvement_pct': float(r2_improvement_pct),

            # Sample counts
            'n_valid_samples': int(n_valid_conv),

            # Arrays for visualization
            'convolved_luminance': conv_lum,
            'residuals_convolved': residuals_conv,
            'residuals_instantaneous': residuals_inst,
            'predicted_convolved': pred_conv,
            'predicted_instantaneous': pred_inst,
        }

    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results dictionary when computation fails."""
        return {
            'r_squared_instantaneous': 0.0,
            'slope_instantaneous': 0.0,
            'intercept_instantaneous': 0.0,
            'p_value_instantaneous': 1.0,
            'residual_std_instantaneous': np.nan,
            'r_squared_convolved': 0.0,
            'slope_convolved': 0.0,
            'intercept_convolved': 0.0,
            'p_value_convolved': 1.0,
            'residual_std_convolved': np.nan,
            'residual_mean_convolved': np.nan,
            'kernel_t_max_ms': self.CANONICAL_T_MAX,
            'kernel_n': self.CANONICAL_N,
            'kernel_is_fitted': False,
            'r_squared_improvement': 0.0,
            'r_squared_improvement_pct': 0.0,
            'n_valid_samples': 0,
            'convolved_luminance': np.array([]),
            'residuals_convolved': np.array([]),
            'residuals_instantaneous': np.array([]),
            'predicted_convolved': np.array([]),
            'predicted_instantaneous': np.array([]),
        }

    def get_kernel_for_plotting(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get kernel and time array for visualization.

        Returns
        -------
        tuple
            (time_ms, kernel_values) arrays
        """
        if self.kernel is None or self._kernel_time_ms is None:
            raise ValueError("No kernel available. Call fit_to_subject() first.")

        return self._kernel_time_ms.copy(), self.kernel.copy()
