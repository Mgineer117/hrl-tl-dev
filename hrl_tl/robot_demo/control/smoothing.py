"""Real-time causal action smoothing filters for robot demonstrations."""

from __future__ import annotations

import enum
from typing import Final

import numpy as np
from scipy import signal


class SmoothingMethod(enum.StrEnum):
    """Supported smoothing algorithms."""

    NONE = "none"
    EMA = "ema"
    BUTTERWORTH = "butterworth"


_DEFAULT_ALPHA: Final[float] = 0.4
_DEFAULT_CUTOFF_HZ: Final[float] = 1.0
_DEFAULT_SAMPLE_RATE_HZ: Final[float] = 10.0
_DEFAULT_ORDER: Final[int] = 2


class ActionSmoothingFilter:
    """A causal real-time low-pass filter for 2D continuous action vectors."""

    def __init__(
        self,
        method: SmoothingMethod = SmoothingMethod.BUTTERWORTH,
        *,
        alpha: float = _DEFAULT_ALPHA,
        cutoff_hz: float = _DEFAULT_CUTOFF_HZ,
        sample_rate_hz: float = _DEFAULT_SAMPLE_RATE_HZ,
        order: int = _DEFAULT_ORDER,
    ) -> None:
        """Initializes the action smoothing filter.

        Args:
            method: Smoothing algorithm (NONE, EMA, or BUTTERWORTH).
            alpha: Exponential moving average coefficient, must be in (0, 1].
            cutoff_hz: Low-pass cutoff frequency in Hertz.
            sample_rate_hz: Control update or sampling rate in Hertz.
            order: Order of the digital Butterworth filter.

        Raises:
            ValueError: If alpha is out of range or cutoff exceeds Nyquist limit.
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}.")

        self.method: SmoothingMethod = method
        self.alpha: float = alpha
        self.cutoff_hz: float = cutoff_hz
        self.sample_rate_hz: float = sample_rate_hz
        self.order: int = order

        self._butter_b: np.ndarray | None = None
        self._butter_a: np.ndarray | None = None
        if method == SmoothingMethod.BUTTERWORTH:
            nyquist = sample_rate_hz / 2.0
            if cutoff_hz >= nyquist:
                raise ValueError(
                    f"Cutoff {cutoff_hz} Hz must be below Nyquist {nyquist} Hz."
                )
            self._butter_b, self._butter_a = signal.butter(
                order, cutoff_hz, fs=sample_rate_hz, btype="low"
            )

        self._prev_action: np.ndarray | None = None
        self._butter_zi: list[np.ndarray] = []
        self._is_initialized: bool = False

    def reset(self) -> None:
        """Resets filter history and delay states."""
        self._prev_action = None
        self._butter_zi = []
        self._is_initialized = False

    def step(self, action: np.ndarray) -> np.ndarray:
        """Applies causal smoothing to an incoming continuous 2D action array.

        Args:
            action: Instantaneous 2D continuous action array.

        Returns:
            Smoothed continuous action array with the same shape and dtype.
        """
        if self.method == SmoothingMethod.NONE:
            return action.copy()

        if not self._is_initialized:
            self._prev_action = action.copy()
            if self.method == SmoothingMethod.BUTTERWORTH:
                if self._butter_b is None or self._butter_a is None:
                    raise RuntimeError(
                        "Butterworth filter coefficients not initialized."
                    )
                flat_action = action.flatten()
                self._butter_zi = [
                    signal.lfilter_zi(self._butter_b, self._butter_a) * val
                    for val in flat_action
                ]
            self._is_initialized = True
            return action.copy()

        if self.method == SmoothingMethod.EMA:
            if self._prev_action is None:
                raise RuntimeError("Previous action state is missing for EMA.")
            smoothed = (
                self.alpha * action + (1.0 - self.alpha) * self._prev_action
            )
            self._prev_action = smoothed.copy()
            return smoothed

        if self.method == SmoothingMethod.BUTTERWORTH:
            if self._butter_b is None or self._butter_a is None:
                raise RuntimeError(
                    "Butterworth filter coefficients not initialized."
                )
            flat_action = action.flatten()
            smoothed_flat = np.zeros_like(flat_action)
            for i, val in enumerate(flat_action):
                filtered, self._butter_zi[i] = signal.lfilter(
                    self._butter_b,
                    self._butter_a,
                    [val],
                    zi=self._butter_zi[i],
                )
                smoothed_flat[i] = filtered[0]
            return smoothed_flat.reshape(action.shape)

        return action.copy()


def create_smoothing_filter(
    method: str | SmoothingMethod,
    *,
    alpha: float = _DEFAULT_ALPHA,
    cutoff_hz: float = _DEFAULT_CUTOFF_HZ,
    sample_rate_hz: float = _DEFAULT_SAMPLE_RATE_HZ,
    order: int = _DEFAULT_ORDER,
) -> ActionSmoothingFilter | None:
    """Creates an ActionSmoothingFilter or returns None if method is NONE.

    Args:
        method: Desired smoothing algorithm name or enum.
        alpha: Exponential moving average coefficient, must be in (0, 1].
        cutoff_hz: Low-pass cutoff frequency in Hertz.
        sample_rate_hz: Sampling or control frequency in Hertz.
        order: Order of the Butterworth filter.

    Returns:
        Configured ActionSmoothingFilter, or None if method is 'none'.
    """
    parsed = (
        SmoothingMethod(method.lower()) if isinstance(method, str) else method
    )
    if parsed == SmoothingMethod.NONE:
        return None
    return ActionSmoothingFilter(
        method=parsed,
        alpha=alpha,
        cutoff_hz=cutoff_hz,
        sample_rate_hz=sample_rate_hz,
        order=order,
    )
