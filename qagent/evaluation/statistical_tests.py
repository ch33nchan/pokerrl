import numpy as np
from scipy import stats
from typing import Tuple

def run_t_test(sample1: np.ndarray, sample2: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, bool]:
    """
    Performs an independent two-sample t-test to compare the means of two samples.

    Args:
        sample1 (np.ndarray): A sample of observations (e.g., rewards) from the first group.
        sample2 (np.ndarray): A sample of observations from the second group.
        alpha (float): The significance level for the test.

    Returns:
        A tuple containing:
        - t_stat (float): The computed t-statistic.
        - p_value (float): The two-tailed p-value.
        - significant (bool): True if the result is statistically significant (p < alpha).
    """
    if len(sample1) < 2 or len(sample2) < 2:
        return np.nan, np.nan, False
        
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=False)  # Welch's t-test
    significant = p_value < alpha
    return t_stat, p_value, significant

def run_anova(samples: list[np.ndarray], alpha: float = 0.05) -> Tuple[float, float, bool]:
    """
    Performs a one-way ANOVA test to compare the means of three or more samples.

    Args:
        samples (list[np.ndarray]): A list of samples (e.g., rewards from different agents).
        alpha (float): The significance level for the test.

    Returns:
        A tuple containing:
        - f_stat (float): The computed F-statistic.
        - p_value (float): The p-value.
        - significant (bool): True if the result is statistically significant (p < alpha).
    """
    if len(samples) < 3:
        raise ValueError("ANOVA requires at least three sample groups.")
    
    f_stat, p_value = stats.f_oneway(*samples)
    significant = p_value < alpha
    return f_stat, p_value, significant

def bootstrap_ci(sample: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float]:
    """
    Calculates the confidence interval for the mean of a sample using bootstrapping.

    Args:
        sample (np.ndarray): The sample of observations.
        n_bootstrap (int): The number of bootstrap samples to generate.
        ci (float): The desired confidence interval (e.g., 0.95 for 95% CI).

    Returns:
        A tuple containing the lower and upper bounds of the confidence interval.
    """
    bootstrap_means = np.array([
        np.mean(np.random.choice(sample, size=len(sample), replace=True))
        for _ in range(n_bootstrap)
    ])
    
    lower_percentile = (1.0 - ci) / 2.0 * 100
    upper_percentile = (1.0 + ci) / 2.0 * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return lower_bound, upper_bound

def get_effect_size_cohens_d(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """
    Calculates Cohen's d as a measure of effect size between two samples.

    Args:
        sample1 (np.ndarray): Sample from the first group.
        sample2 (np.ndarray): Sample from the second group.

    Returns:
        float: The calculated Cohen's d value.
    """
    n1, n2 = len(sample1), len(sample2)
    if n1 < 2 or n2 < 2:
        return np.nan
        
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return np.nan
        
    cohens_d = (mean1 - mean2) / pooled_std
    return cohens_d
