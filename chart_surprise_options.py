import matplotlib.pyplot as plt
import numpy as np


def generate_naive_surprise_chart():
    # Generate probability values from 0.01 to 0.99
    p = np.linspace(0.01, 0.99, 100)

    # Calculate 1 - p(x)
    naive_entropy = 1 - p

    plt.xkcd()
    plt.rcParams['font.family'] = "xkcd"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    # Plot without markers
    ax.plot(p, naive_entropy)

    ax.set_xlabel("P(X)")
    ax.set_ylabel("1 - P(X)")
    ax.legend()

    fig.tight_layout()
    fig.savefig("naive_surprise.png", bbox_inches="tight")
    plt.close(fig)


def generate_real_surprise_chart():
    # Generate probability values from 0.01 to 0.99
    p = np.linspace(0.01, 0.99, 100)

    # Calculate 1 - p(x)
    naive_entropy = - np.log(p)

    plt.xkcd()
    plt.rcParams['font.family'] = "xkcd"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    # Plot without markers
    ax.plot(p, naive_entropy)

    ax.set_xlabel("P(X)")
    ax.set_ylabel("- LOG(P(X))")
    ax.legend()

    fig.tight_layout()
    fig.savefig("real_surprise.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    generate_naive_surprise_chart()
    generate_real_surprise_chart()

