"""
Produces plots of beta distributions

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta
import itertools

# observed data
y1, n1 = 5, 10
y2, n2 = 50, 100

data = [
    (5, 10),
    (500, 1000)
]

prior_params = [
    (1, 1),
    (30, 70)
]

# parameters for the beta distributions
alpha1, beta1 = 1, 1
alpha2, beta2 = 300, 700

params = [
    (y1+alpha1, n1-y1+beta1), (y2+alpha1, n2-y2+beta1),
    (y1+alpha2, n1-y1+beta2), (y2+alpha2, n2-y2+beta2)
]

# plot the beta distributions
sns.set(style="darkgrid")

x = np.linspace(0, 1, 100)

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

for p, ax in zip(itertools.product(prior_params, data), itertools.chain(*axes)):
    prior_alpha, prior_beta = p[0]
    y, n = p[1]
    
    alpha = y + prior_alpha
    beta_ = n - y + prior_beta
    
    # plot prior distribution
    sns.lineplot(
        ax=ax,
        x=x, 
        y=beta.pdf(x, prior_alpha, prior_beta), 
        label=f'Prior Beta({prior_alpha}, {prior_beta})',
    )
    
    # plot posterior distribution
    sns.lineplot(
        ax=ax,
        x=x, 
        y=beta.pdf(x, alpha, beta_), 
        label=f'Posterior Beta({y} + {prior_alpha}, {n - y} + {prior_beta})'
    )
    
    # annotate max density
    mode = (alpha - 1) / (alpha + beta_ - 2)
    ax.axvline(mode, color='grey', linestyle='--', label=f'Mode at {mode:.2f}')

    ax.text(
        mode, 
        max(beta.pdf(x, alpha, beta_)), 
        f'MAP: {mode:.2f}', 
        color='grey', 
        fontsize=10,
        horizontalalignment='right'
    )

    # add axis labels
    ax.set_xlabel(r'$\theta$ (probability of success)')
    ax.set_ylabel(r'Posterior Density')

# Adding title and labels
plt.suptitle(r'Posterior Distributions for $\theta$', fontsize=14)

plt.tight_layout()

plt.savefig('posterior_beta_distributions.png', dpi=300)
