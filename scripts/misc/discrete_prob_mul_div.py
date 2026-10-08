"""
Script to multiply and divide two discrete probability distributions.

For discrete distributions P and Q over the same support:
- Multiplication: R(x) ∝ P(x) * Q(x), then normalize
- Division: R(x) ∝ P(x) / Q(x), then normalize (undefined where Q(x) = 0)
"""

import matplotlib.pyplot as plt
import numpy as np


def normalize(probs):
    """Normalize a probability distribution."""
    total = np.sum(probs)
    if total > 0:
        return probs / total
    return probs


def multiply_distributions(p1, p2):
    """
    Multiply two discrete probability distributions.

    Args:
        p1, p2: Arrays of probabilities for each bin

    Returns:
        p_mult: Normalized product distribution
    """
    p_mult = p1 * p2
    return normalize(p_mult)


def divide_distributions(p1, p2, epsilon=1e-10):
    """
    Divide two discrete probability distributions (P1 / P2).

    Args:
        p1: Numerator distribution
        p2: Denominator distribution
        epsilon: Small value to avoid division by zero

    Returns:
        p_div: Normalized quotient distribution
        valid: Boolean indicating if division is well-defined
    """
    # Check for zeros in denominator
    if np.any(p2 < epsilon):
        print(
            f"Warning: Denominator has near-zero probabilities at indices: {np.where(p2 < epsilon)[0]}"
        )
        # Replace zeros with epsilon to avoid inf
        p2_safe = np.maximum(p2, epsilon)
        p_div = p1 / p2_safe
        return normalize(p_div), False

    p_div = p1 / p2
    return normalize(p_div), True


def plot_distributions(bins, p1, p2, p_mult, p_div=None, div_valid=True):
    """
    Plot the original distributions and the results of multiplication and division.

    Args:
        bins: Array of bin labels/values
        p1, p2: Original probability distributions
        p_mult: Product distribution
        p_div: Quotient distribution (optional)
        div_valid: Whether division was well-defined
    """
    n_plots = 2 if p_div is None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 2:
        axes = [axes[0], axes[1], None]

    width = 0.35
    x = np.arange(len(bins))

    # Plot 1: Original Distributions
    ax = axes[0]
    ax.bar(
        x - width / 2,
        p1,
        width,
        label="Distribution 1",
        color="blue",
        alpha=0.7,
    )
    ax.bar(
        x + width / 2, p2, width, label="Distribution 2", color="red", alpha=0.7
    )
    ax.set_xlabel("Bins", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title("Original Distributions", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(np.max(p1), np.max(p2)) * 1.1)

    # Plot 2: Multiplication
    ax = axes[1]
    ax.bar(
        x - 2 * width / 3,
        p1,
        width * 0.6,
        label="Distribution 1",
        color="blue",
        alpha=0.4,
    )
    ax.bar(x, p2, width * 0.6, label="Distribution 2", color="red", alpha=0.4)
    ax.bar(
        x + 2 * width / 3,
        p_mult,
        width * 0.6,
        label="Product",
        color="green",
        alpha=0.9,
    )
    ax.set_xlabel("Bins", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title("Distribution Multiplication", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(np.max(p1), np.max(p2), np.max(p_mult)) * 1.1)

    # Plot 3: Division (if applicable)
    if p_div is not None and axes[2] is not None:
        ax = axes[2]
        ax.bar(
            x - 2 * width / 3,
            p1,
            width * 0.6,
            label="Distribution 1",
            color="blue",
            alpha=0.4,
        )
        ax.bar(
            x, p2, width * 0.6, label="Distribution 2", color="red", alpha=0.4
        )
        ax.bar(
            x + 2 * width / 3,
            p_div,
            width * 0.6,
            label="Quotient",
            color="magenta",
            alpha=0.9,
        )
        ax.set_xlabel("Bins", fontsize=12)
        ax.set_ylabel("Probability", fontsize=12)

        title = "Distribution Division"
        if not div_valid:
            title += " (with regularization)"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, max(np.max(p1), np.max(p2), np.max(p_div)) * 1.1)

    plt.tight_layout()
    return fig


def main():
    """
    Main function demonstrating discrete distribution multiplication and division.
    """
    import os

    # Define 5 bins
    bins = ["A", "B", "C", "D", "E"]
    n_bins = len(bins)

    # Define two discrete probability distributions
    # Distribution 1: peaked at bin C
    p1 = np.array([0.3, 0.2, 0.1, 0.3, 0.1])

    # Distribution 2: more uniform with slight peak at D
    p2 = np.array([0.10, 0.2, 0.2, 0.35, 0.15])

    # Verify they're normalized
    p1 = normalize(p1)
    p2 = normalize(p2)

    print("=" * 60)
    print("DISCRETE PROBABILITY DISTRIBUTION OPERATIONS")
    print("=" * 60)
    print(f"\nBins: {bins}")
    print(f"Distribution 1: {p1}")
    print(f"Distribution 2: {p2}")
    print(f"Sum P1: {np.sum(p1):.4f}, Sum P2: {np.sum(p2):.4f}")

    # Multiply distributions
    p_mult = multiply_distributions(p1, p2)
    print(f"\nMultiplication Result:")
    print(f"  Product: {p_mult}")
    print(f"  Sum: {np.sum(p_mult):.4f}")

    # Divide distributions
    p_div, div_valid = divide_distributions(p1, p2)
    print(f"\nDivision Result (Distribution 1 / Distribution 2):")
    print(f"  Quotient: {p_div}")
    print(f"  Sum: {np.sum(p_div):.4f}")
    print(f"  Well-defined: {div_valid}")

    # Plot the results
    print("\nGenerating plots...")
    fig = plot_distributions(bins, p1, p2, p_mult, p_div, div_valid)

    # Save the figure
    output_dir = "out/plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "discrete_prob_mul_div.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)
    print("• Multiplication: Emphasizes bins where both distributions agree")
    print("  (high probability in both → even higher in product)")
    print("• Division: Emphasizes bins where P1 is high relative to P2")
    print("  (redistributes mass based on relative probabilities)")
    print("• Both operations preserve normalization (sum to 1)")
    print("=" * 60)

    # Additional example: Division with zeros
    print("\n" + "=" * 60)
    print("EXAMPLE WITH ZERO PROBABILITIES:")
    print("=" * 60)

    p3 = np.array([0.3, 0.3, 0.4, 0.0, 0.0])  # Zeros in last two bins
    p4 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Uniform

    print(f"\nDistribution 3: {p3} (with zeros)")
    print(f"Distribution 4: {p4} (uniform)")

    p_div2, div_valid2 = divide_distributions(p4, p3)
    print(f"\nDivision (Distribution 4 / Distribution 3):")
    print(f"  Quotient: {p_div2}")
    print(f"  Well-defined: {div_valid2}")
    print("  Note: Division by zero bins are regularized")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
