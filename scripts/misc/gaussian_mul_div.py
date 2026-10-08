"""
Script to multiply and divide two Gaussian distributions.

For two Gaussians N(μ₁, σ₁²) and N(μ₂, σ₂²):
- Multiplication: N(μ_new, σ_new²) where:
    σ_new² = 1/(1/σ₁² + 1/σ₂²)
    μ_new = σ_new² * (μ₁/σ₁² + μ₂/σ₂²)
- Division: N(μ_new, σ_new²) where:
    σ_new² = 1/(1/σ₁² - 1/σ₂²)
    μ_new = σ_new² * (μ₁/σ₁² - μ₂/σ₂²)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def multiply_gaussians(mu1, sigma1, mu2, sigma2):
    """
    Multiply two Gaussian distributions.

    Args:
        mu1, sigma1: Mean and standard deviation of first Gaussian
        mu2, sigma2: Mean and standard deviation of second Gaussian

    Returns:
        mu_new, sigma_new: Mean and standard deviation of the product
    """
    var1, var2 = sigma1**2, sigma2**2
    var_new = 1 / (1 / var1 + 1 / var2)
    mu_new = var_new * (mu1 / var1 + mu2 / var2)
    sigma_new = np.sqrt(var_new)
    return mu_new, sigma_new


def divide_gaussians(mu1, sigma1, mu2, sigma2):
    """
    Divide two Gaussian distributions (Gaussian 1 / Gaussian 2).

    Args:
        mu1, sigma1: Mean and standard deviation of numerator Gaussian
        mu2, sigma2: Mean and standard deviation of denominator Gaussian

    Returns:
        mu_new, sigma_new: Mean and standard deviation of the quotient
        or None, None if division is not well-defined
    """
    var1, var2 = sigma1**2, sigma2**2

    # Check if division is well-defined (requires 1/var1 > 1/var2, i.e., var1 < var2)
    if var1 >= var2:
        print(
            f"Warning: Division not well-defined. var1={var1:.4f} >= var2={var2:.4f}"
        )
        print(
            "         For division, numerator must have smaller variance (higher precision) than denominator."
        )
        return None, None

    var_new = 1 / (1 / var1 - 1 / var2)
    mu_new = var_new * (mu1 / var1 - mu2 / var2)
    sigma_new = np.sqrt(var_new)
    return mu_new, sigma_new


def plot_gaussians(
    mu1, sigma1, mu2, sigma2, mu_mult, sigma_mult, mu_div=None, sigma_div=None
):
    """
    Plot the original Gaussians and the results of multiplication and division.
    """
    # Create x-axis range
    x_min = min(mu1 - 4 * sigma1, mu2 - 4 * sigma2)
    x_max = max(mu1 + 4 * sigma1, mu2 + 4 * sigma2)

    if mu_mult is not None:
        x_min = min(x_min, mu_mult - 4 * sigma_mult)
        x_max = max(x_max, mu_mult + 4 * sigma_mult)

    if mu_div is not None:
        x_min = min(x_min, mu_div - 4 * sigma_div)
        x_max = max(x_max, mu_div + 4 * sigma_div)

    x = np.linspace(x_min, x_max, 1000)

    # Create subplots
    n_plots = 2 if mu_div is None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 2:
        axes = [axes[0], axes[1], None]

    # Plot 1: Original Gaussians
    ax = axes[0]
    pdf1 = norm.pdf(x, mu1, sigma1)
    pdf2 = norm.pdf(x, mu2, sigma2)
    ax.plot(x, pdf1, "b-", linewidth=2, label=f"N({mu1:.2f}, {sigma1:.2f}²)")
    ax.plot(x, pdf2, "r-", linewidth=2, label=f"N({mu2:.2f}, {sigma2:.2f}²)")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title("Original Gaussians", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Multiplication
    ax = axes[1]
    pdf1 = norm.pdf(x, mu1, sigma1)
    pdf2 = norm.pdf(x, mu2, sigma2)
    pdf_mult = norm.pdf(x, mu_mult, sigma_mult)
    ax.plot(
        x,
        pdf1,
        "b-",
        linewidth=1.5,
        alpha=0.5,
        label=f"N({mu1:.2f}, {sigma1:.2f}²)",
    )
    ax.plot(
        x,
        pdf2,
        "r-",
        linewidth=1.5,
        alpha=0.5,
        label=f"N({mu2:.2f}, {sigma2:.2f}²)",
    )
    ax.plot(
        x,
        pdf_mult,
        "g-",
        linewidth=2.5,
        label=f"Product: N({mu_mult:.2f}, {sigma_mult:.2f}²)",
    )
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title("Gaussian Multiplication", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 3: Division (if applicable)
    if mu_div is not None and axes[2] is not None:
        ax = axes[2]
        pdf1 = norm.pdf(x, mu1, sigma1)
        pdf2 = norm.pdf(x, mu2, sigma2)
        pdf_div = norm.pdf(x, mu_div, sigma_div)
        ax.plot(
            x,
            pdf1,
            "b-",
            linewidth=1.5,
            alpha=0.5,
            label=f"N({mu1:.2f}, {sigma1:.2f}²)",
        )
        ax.plot(
            x,
            pdf2,
            "r-",
            linewidth=1.5,
            alpha=0.5,
            label=f"N({mu2:.2f}, {sigma2:.2f}²)",
        )
        ax.plot(
            x,
            pdf_div,
            "m-",
            linewidth=2.5,
            label=f"Quotient: N({mu_div:.2f}, {sigma_div:.2f}²)",
        )
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.set_title("Gaussian Division", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """
    Main function demonstrating Gaussian multiplication and division.
    """
    import os

    # Define two Gaussian distributions
    # For division to be well-defined: var1 < var2 (numerator has higher precision)
    mu1, sigma1 = 2.0, 1.0
    mu2, sigma2 = 5.0, 1.5

    print("=" * 60)
    print("GAUSSIAN MULTIPLICATION AND DIVISION")
    print("=" * 60)
    print(f"\nGaussian 1: μ₁ = {mu1:.2f}, σ₁ = {sigma1:.2f}")
    print(f"Gaussian 2: μ₂ = {mu2:.2f}, σ₂ = {sigma2:.2f}")

    # Multiply Gaussians
    mu_mult, sigma_mult = multiply_gaussians(mu1, sigma1, mu2, sigma2)
    print(f"\n{'Multiplication Result:'}")
    print(f"  μ_product = {mu_mult:.4f}")
    print(f"  σ_product = {sigma_mult:.4f}")

    # Divide Gaussians - try both directions
    mu_div, sigma_div = divide_gaussians(mu1, sigma1, mu2, sigma2)
    if mu_div is not None:
        print(f"\n{'Division Result (Gaussian 1 / Gaussian 2):'}")
        print(f"  μ_quotient = {mu_div:.4f}")
        print(f"  σ_quotient = {sigma_div:.4f}")
    else:
        # Try the other direction
        print(
            "\nTrying division in reverse direction (Gaussian 2 / Gaussian 1)..."
        )
        mu_div, sigma_div = divide_gaussians(mu2, sigma2, mu1, sigma1)
        if mu_div is not None:
            print("Division Result (Gaussian 2 / Gaussian 1):")
            print(f"  μ_quotient = {mu_div:.4f}")
            print(f"  σ_quotient = {sigma_div:.4f}")
        else:
            print("Division is not well-defined in either direction.")

    # Plot the results
    print("\nGenerating plots...")
    fig = plot_gaussians(
        mu1, sigma1, mu2, sigma2, mu_mult, sigma_mult, mu_div, sigma_div
    )

    # Save the figure
    output_dir = "out/plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gaussian_mul_div.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)
    print("• Multiplication: Result has smaller variance (more peaked)")
    print("• Division: Result has larger variance (more spread out)")
    print("• Division requires variance of numerator < variance of denominator")
    print("  (numerator must have higher precision than denominator)")
    print("=" * 60)


if __name__ == "__main__":
    main()
