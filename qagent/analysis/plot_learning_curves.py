import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def plot_learning_curves_ci(results_file, agent_names, title, output_filename):
    """
    Plots learning curves with 95% confidence intervals for multiple agents.
    Handles missing data gracefully and provides clear visual feedback.

    Args:
        results_file (str): Path to the JSON file containing the results.
        agent_names (dict): Dictionary mapping agent keys to display names.
        title (str): The title for the plot.
        output_filename (str): Path to save the output plot image.
    """
    with open(results_file, "r", encoding="utf-8") as f:
        all_run_data = json.load(f)

    plt.figure(figsize=(14, 8))
    plotted_any = False
    skipped_agents = []

    for agent_key, agent_display_name in agent_names.items():
        if agent_key not in all_run_data:
            skipped_agents.append(f"{agent_display_name} (no data)")
            continue
            
        agent_runs = all_run_data[agent_key]
        if not agent_runs or all(len(run) == 0 for run in agent_runs):
            skipped_agents.append(f"{agent_display_name} (no exploitability data)") 
            continue
            
        reference_run = next((run for run in agent_runs if run), None)
        if not reference_run:
            skipped_agents.append(f"{agent_display_name} (invalid data)")
            continue

        reference_iterations = [item[0] for item in reference_run]
        all_runs = []
        for run_data in agent_runs:
            run_dict = dict(run_data)
            aligned = [run_dict.get(it, np.nan) for it in reference_iterations]
            all_runs.append(aligned)
            
        df = pd.DataFrame(all_runs, columns=reference_iterations).astype(float)
        df.columns = df.columns.astype(int)
        
        if df.dropna(how="all").empty:
            skipped_agents.append(f"{agent_display_name} (all NaN)")
            continue
            
        mean = df.mean(axis=0)
        std = df.std(axis=0)
        ci = 1.96 * (std / np.sqrt(len(df)))
        
        plt.plot(
            mean.index,
            mean,
            label=agent_display_name,
            linewidth=2,
            marker="o",
            markersize=6,
        )

        if len(mean.index) > 1:
            plt.fill_between(mean.index, mean - ci, mean + ci, alpha=0.15)
        plotted_any = True

    if not plotted_any:
        plt.text(0.5, 0.5, "No exploitability data available", 
                ha="center", va="center", fontsize=14, color="red")
        plt.axis("off")
    else:
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel("Training Iterations", fontsize=12)
        plt.ylabel("Exploitability (mbb/g)", fontsize=12)
        plt.legend(title="Agent Architecture", framealpha=0.9)
        plt.grid(True, linestyle="--", alpha=0.5)
        
        if skipped_agents:
            plt.figtext(0.5, 0.01, "Skipped: " + ", ".join(skipped_agents), 
                       ha="center", fontsize=10, color="gray")
        
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_filename}")
    plt.close()