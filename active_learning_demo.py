"""
Simple Active Learning demo with a Gaussian Process (gpytorch, CPU).

We have a known 1D target function. Starting from a couple of random points,
we repeatedly:
  1. Fit a GP to the currently-labelled points.
  2. Look at the GP's predictive uncertainty (std) over a dense grid.
  3. Query the *most uncertain* location and add it to the training set.

Each iteration is plotted (true function, GP mean + 2-sigma band, sampled
points, and the next query point), and all frames are stitched into a GIF.

Everything runs on CPU.
"""

import os

import torch
import gpytorch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# --- global plot style: black background + large fonts --------------------
plt.style.use("dark_background")
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "legend.fontsize": 16,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "savefig.facecolor": "black",
    "lines.linewidth": 2.5,
})

# --- reproducibility & CPU only -------------------------------------------
torch.manual_seed(0)
np.random.seed(0)
DEVICE = torch.device("cpu")

OUT_DIR = "frames"
os.makedirs(OUT_DIR, exist_ok=True)


# --- the (known) function we want to learn --------------------------------
def true_function(x):
    """A smooth-ish 1D function on [0, 1]."""
    return torch.sin(6.0 * x) + 0.5 * torch.cos(14.0 * x)


# --- the GP model ---------------------------------------------------------
class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_gp(train_x, train_y, n_iter=100, lr=0.1):
    """Fit a fresh GP to the given data and return (model, likelihood)."""
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
    model = ExactGP(train_x, train_y, likelihood).to(DEVICE)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    return model, likelihood


def predict(model, likelihood, x):
    """Return predictive mean and std over x."""
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x))
    return pred.mean, pred.stddev


# --- active learning loop -------------------------------------------------
def main():
    # dense grid: candidate query locations + plotting resolution
    grid = torch.linspace(0.0, 1.0, 400, device=DEVICE)
    true_y = true_function(grid)

    # start from 2 random labelled points
    n_start = 2
    n_iterations = 12
    train_x = torch.rand(n_start, device=DEVICE)
    train_y = true_function(train_x)

    frame_paths = []

    for it in range(n_iterations):
        model, likelihood = fit_gp(train_x, train_y)
        mean, std = predict(model, likelihood, grid)

        # acquisition = maximum predictive uncertainty
        next_idx = int(torch.argmax(std).item())
        next_x = grid[next_idx]

        # --- plot this iteration ------------------------------------------
        fig, (ax, ax_u) = plt.subplots(
            2, 1, figsize=(12, 9), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        g = grid.cpu().numpy()
        m = mean.cpu().numpy()
        s = std.cpu().numpy()

        ax.plot(g, true_y.cpu().numpy(), "w--", lw=2.5, label="true function")
        ax.plot(g, m, "C0", lw=3, label="GP mean")
        ax.fill_between(g, m - 2 * s, m + 2 * s, color="C0", alpha=0.25,
                        label="GP ±2σ")
        ax.scatter(train_x.cpu().numpy(), train_y.cpu().numpy(),
                   c="C3", s=110, zorder=5, label=f"samples (n={len(train_x)})")
        ax.axvline(next_x.item(), color="C2", ls=":", lw=2.5,
                   label="next query")
        ax.set_ylabel("y")
        ax.set_title(f"Active learning iteration {it + 1}/{n_iterations}")
        ax.legend(loc="upper right")
        ax.set_ylim(-2.5, 2.5)

        # uncertainty panel
        ax_u.plot(g, s, "C0", lw=2.5)
        ax_u.fill_between(g, 0, s, color="C0", alpha=0.25)
        ax_u.axvline(next_x.item(), color="C2", ls=":", lw=2.5)
        ax_u.scatter([next_x.item()], [s[next_idx]], c="C2", s=90, zorder=5)
        ax_u.set_ylabel("GP std")
        ax_u.set_xlabel("x")
        ax_u.set_title("predictive uncertainty (acquisition)")

        fig.tight_layout()
        path = os.path.join(OUT_DIR, f"iter_{it:02d}.png")
        fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
        plt.close(fig)
        frame_paths.append(path)

        print(f"iter {it + 1:2d}: n={len(train_x):2d}  "
              f"max std={s[next_idx]:.3f}  -> query x={next_x.item():.3f}")

        # --- label the queried point and add it ---------------------------
        train_x = torch.cat([train_x, next_x.unsqueeze(0)])
        train_y = torch.cat([train_y, true_function(next_x).unsqueeze(0)])

    # --- build the GIF ----------------------------------------------------
    gif_path = "active_learning.gif"
    frames = [imageio.imread(p) for p in frame_paths]
    # duration is per-frame in milliseconds -> 500 ms = 0.5 s per iteration
    imageio.mimsave(gif_path, frames, duration=500, loop=0)
    print(f"\nSaved {len(frame_paths)} frames in '{OUT_DIR}/' and GIF '{gif_path}'")


if __name__ == "__main__":
    main()
