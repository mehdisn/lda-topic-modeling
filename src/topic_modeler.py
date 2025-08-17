import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from typing import Tuple, List, Dict
from tqdm import tqdm
from pathlib import Path
import os


class TopicModeler:
    def __init__(self, mallet_path: str, results_path: Path):
        self.mallet_path = mallet_path
        self.results_path = results_path

        if not self.results_path.exists():
            self.results_path.mkdir(parents=True, exist_ok=True)

        print(f"Initialized TopicModeler with MALLET at: {self.mallet_path}")

    def run_experiments(self, corpus: list, id2word: corpora.Dictionary, texts: list,
                        start: int, limit: int, step: int, num_runs: int) -> Tuple[List[Dict], pd.DataFrame]:
        all_scores = []
        models_data = []

        topic_range = range(start, limit, step)

        # Outer loop (runs)
        for run_idx in tqdm(range(1, num_runs + 1), desc="Runs", position=0):
            # Inner loop (topic counts)
            for k_topics in tqdm(topic_range, desc=f"Run {run_idx}", position=1, leave=False):
                lda_model = gensim.models.wrappers.LdaMallet(
                    mallet_path=self.mallet_path,
                    corpus=corpus,
                    num_topics=k_topics,
                    id2word=id2word,
                    iterations=1000,
                    alpha=50 / k_topics,
                    random_seed=run_idx,
                    prefix=os.devnull  # suppress MALLET logs
                )

                save_file = self.results_path / f"lda_mallet_run{run_idx}_topics{k_topics}"
                lda_model.save(str(save_file))

                # Evaluate model coherence
                cm = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
                score = cm.get_coherence()

                models_data.append({
                    "run": run_idx,
                    "num_topics": k_topics,
                    "model": lda_model,
                    "coherence": score
                })

                all_scores.append({
                    "Run": f"Run {run_idx}",
                    "Num Topics": k_topics,
                    "Coherence Value": score
                })

        scores_df = pd.DataFrame(all_scores)
        return models_data, scores_df

    def format_topics_sentences(self, ldamodel: gensim.models.LdaModel, corpus: list) -> pd.DataFrame:
        topic_data = []
        for i, row_list in enumerate(ldamodel[corpus]):
            row = sorted(row_list, key=lambda x: x[1], reverse=True)
            dominant_topic, prop_topic = row[0]
            wp = ldamodel.show_topic(dominant_topic)
            topic_keywords = ", ".join([word for word, prop in wp])
            topic_data.append({
                'Dominant_Topic': int(dominant_topic),
                'Topic_Perc_Contrib': round(prop_topic, 4),
                'Keywords': topic_keywords
            })
        return pd.DataFrame(topic_data)
