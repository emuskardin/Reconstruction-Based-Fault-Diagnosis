import seaborn as sns
import scipy.stats as stats
from matplotlib import pyplot as plt

def plot_reconstruction_loss(save_path, nominal_losses, mean_loss, standard_dev, percentile_95):
    # Plot histogram with KDE
    plt.figure(figsize=(10, 5))
    sns.histplot(nominal_losses, bins=30, kde=True, color="blue", alpha=0.6)

    # Add vertical lines for thresholds
    plt.axvline(mean_loss + 3 * standard_dev, color='red', linestyle='dashed', label="3σ threshold")
    plt.axvline(percentile_95, color='green', linestyle='dashed', label="95th percentile")

    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f'{save_path}/histogram_kde.png')

    # Q-Q plot for normality check
    plt.figure(figsize=(6, 6))
    stats.probplot(nominal_losses, dist="norm", plot=plt)
    plt.title("Q-Q Plot for Normality Check")
    plt.savefig(f'{save_path}/q_plot.png')
