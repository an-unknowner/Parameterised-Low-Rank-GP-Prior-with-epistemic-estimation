"""Small standalone plotting utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def softplus(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Numerically stable Softplus: log(1 + exp(beta*x)) / beta."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    return np.logaddexp(0.0, beta * x) / beta


def plot_softplus(
        x_min: float = -10.0,
        x_max: float = 10.0,
        beta: float = 1.0,
        save_path: str | Path | None = None):
    """Plot the Softplus function and return the Matplotlib figure."""
    x = np.linspace(x_min, x_max, 1000)
    y = softplus(x, beta=beta)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color="tab:blue", linewidth=2,
            label=rf"$\mathrm{{softplus}}(x)=\log(1+e^{{{beta:g}x}})/{beta:g}$")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("softplus(x)")
    ax.set_title("Softplus Function")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parent / "res" / "softplus.png"
    plot_softplus(save_path=output_path)
    print(f"Softplus figure saved to: {output_path}")
    plt.show()
