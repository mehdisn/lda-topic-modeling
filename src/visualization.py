import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import gensim
import gensim.corpora as corpora


class Visualization:
    def __init__(self, results_path: Path):
        self.results_path = results_path
        
        if not self.results_path.exists():
            self.results_path.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Visualization output folder: {self.results_path}")

    def plot_coherence_comparison(self, df_coherence: pd.DataFrame):
        logging.info("Creating coherence comparison chart...")
        
        # Group stats by number of topics
        grouped = df_coherence.groupby("Num Topics").agg({"Coherence Value": ["mean", "std"]}).reset_index()
        grouped.columns = ["Num Topics", "Mean Coherence", "Std Coherence"]  # flatten multi-index
        
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")
        
        # Plot all runs (individual lines)
        sns.lineplot(
            data=df_coherence,
            x="Num Topics",
            y="Coherence Value",
            hue="Run",
            linewidth=1.5,
            alpha=0.85,
            palette="Set2"
        )
        
        # Overlay the average line
        plt.plot(
            grouped["Num Topics"], 
            grouped["Mean Coherence"], 
            color="black",
            linewidth=2.3,
            linestyle="--",
            label="Average Coherence"
        )
        
        # Shade the std dev
        plt.fill_between(
            grouped["Num Topics"],
            grouped["Mean Coherence"] - grouped["Std Coherence"],
            grouped["Mean Coherence"] + grouped["Std Coherence"],
            color="gray",
            alpha=0.18,
            label="±1 Std Dev"
        )
        
        plt.xlabel("Number of Topics", fontsize=13)
        plt.ylabel("Coherence Value", fontsize=13)
        plt.title("Coherence Scores Across Runs", fontsize=15, fontweight="bold")
        
        plt.xticks(
            range(min(df_coherence["Num Topics"]), max(df_coherence["Num Topics"]) + 1, 2)
        )
        
        plt.legend(title="Legend", fontsize=11)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        
        save_file = self.results_path / "coherence_comparison.png"
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved coherence plot to {save_file}")

    def visualize_lda(self, lda_model: gensim.models.LdaModel, corpus: list, id2word: corpora.Dictionary):
        logging.info("Building pyLDAvis output...")
        
        vis_data = gensimvis.prepare(lda_model, corpus, id2word, mds="pcoa")
        
        save_html_path = self.results_path / "lda_best_model_vis.html"
        pyLDAvis.save_html(vis_data, str(save_html_path))  # ✅ now from the main module
        
        logging.info(f"pyLDAvis HTML saved at {save_html_path}")
