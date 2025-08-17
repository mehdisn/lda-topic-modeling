import os
import logging
import pandas as pd
import gensim
from pathlib import Path

from src.data_preprocessor import DataPreprocessor
from src.topic_modeler import TopicModeler
from src.visualization import Visualization
import src.config as config

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_pipeline():
    os.environ["MALLET_HOME"] = str(config.MALLET_HOME)
    
    # --- Step 1: Data Processing ---
    preproc = DataPreprocessor()
    
    logging.info(f"Loading original CSV from {config.INPUT_CSV_PATH}")
    df_raw = pd.read_csv(config.INPUT_CSV_PATH)
    
    processed_texts = preproc.load_and_clean_data(
        csv_path=config.INPUT_CSV_PATH,
        text_col=config.TEXT_COLUMN,
        title_col=config.TITLE_COLUMN,
        save_clean_path=config.CLEAN_TEXT_PATH
    )
    
    texts_out, id2word, corpus = preproc.tokenize_and_lemmatize(processed_texts)

    # --- Step 2: Topic Modeling ---
    modeler = TopicModeler(
        mallet_path=config.MALLET_BIN_PATH, 
        results_path=config.RESULTS_DIR
    )
    all_model_info, coherence_df = modeler.run_experiments(
        corpus=corpus,
        id2word=id2word,
        texts=texts_out,
        start=config.TOPIC_START,
        limit=config.TOPIC_LIMIT,
        step=config.TOPIC_STEP,
        num_runs=config.NUM_RUNS
    )
    
    best_model = max(all_model_info, key=lambda m: m["coherence"])
    mallet_best = best_model["model"]
    
    logging.info(f"=== Best Model Found ===")
    logging.info(f"Run: {best_model['run']}")
    logging.info(f"Topics: {best_model['num_topics']}")
    logging.info(f"Coherence: {best_model['coherence']:.4f}")
    
    lda_best = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(mallet_best)
    
    # --- Step 3: Visualization ---
    vis = Visualization(results_path=config.RESULTS_DIR)
    vis.plot_coherence_comparison(coherence_df)
    vis.visualize_lda(lda_best, corpus, id2word)

    # --- Step 4: Final Output ---
    # **IMPROVEMENT**: Simpler, more robust final DataFrame creation
    df_topic_assignments = modeler.format_topics_sentences(
        ldamodel=lda_best,
        corpus=corpus
    )

    # Join the topic assignments back to the processed dataframe
    # This is a direct 1-to-1 match based on row order
    df_final = preproc.df.join(df_topic_assignments)
    
    output_csv = config.RESULTS_DIR / "final_topics_dataset.csv"
    df_final.to_csv(output_csv, index=False)
    logging.info(f"Pipeline complete — results saved at {output_csv}")

if __name__ == "__main__":
    if not config.INPUT_CSV_PATH.is_file():
        logging.error(f"No CSV found at: {config.INPUT_CSV_PATH}")
    elif not Path(config.MALLET_BIN_PATH).is_file():
        logging.error(f"MALLET binary missing at: {config.MALLET_BIN_PATH}")
    else:
        run_pipeline()
