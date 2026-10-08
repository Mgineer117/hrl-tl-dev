"""Action mode implementations for Fetch environments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class FetchMultiDiscreteAction:
    """Action mode for multi-discrete 3D action space in Fetch.

    Handles summing and blending of discrete displacement actions along
    Cartesian X, Y, and Z axes.
    """

    def __init__(self, num_bins: int = 5) -> None:
        """Initializes the multi-discrete action mode.

        Args:
            num_bins: Number of discrete displacement bins per dimension.
        """
        self.num_bins = num_bins

    def sum_actions(
        self,
        action_1: NDArray,
        action_2: NDArray,
        weight: float = 0.5,
    ) -> NDArray:
        """Sums two multi-discrete 3D actions using weighted linear interpolation.

        Args:
            action_1: First action array of discrete bin indices, shape (..., 3) or (3,).
            action_2: Second action array of discrete bin indices, shape (..., 3) or (3,).
            weight: Weight for the first action in [0, 1].

        Returns:
            The resulting blended action array with integer values clipped to [0, num_bins - 1].
        """
        summed = weight * action_1 + (1.0 - weight) * action_2
        return np.clip(np.round(summed), 0, self.num_bins - 1).astype(np.int64)
