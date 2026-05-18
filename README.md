# LDA Topic Modeling with MALLET & Gensim

This is an end-to-end pipeline for discovering topics in text data using Latent Dirichlet Allocation (LDA). It iterates through different topic counts, calculates coherence scores (c_v), and outputs the best-performing model along with a pyLDAvis dashboard.

## What it does
1. **Preprocesses text:** Removes HTML, code snippets, punctuation, and URLs. Tokenizes, builds bigrams/trigrams, and lemmatizes using spaCy.
2. **Trains models:** Runs MALLET LDA across a configurable range of topic numbers.
3. **Evaluates:** Calculates coherence scores for each run to find the optimal number of topics.
4. **Visualizes:** Generates a coherence comparison plot and an interactive HTML pyLDAvis dashboard.
5. **Outputs:** Merges the dominant topic for each document back into a final CSV.

## Usage

1. Place your dataset as data.csv inside the data/ folder.
2. Open src/config.py. Update TEXT_COLUMN and TITLE_COLUMN to match your CSV headers. You can also adjust the topic search range (TOPIC_START, TOPIC_LIMIT, TOPIC_STEP, NUM_RUNS) here.
3. Run
   ```bash
   python main.py
   ```

## Results

1. **final_topics_dataset.csv:** Your original data with the dominant topic appended.
2. **coherence_comparison.png:** A graph showing coherence scores across different topic counts.
3. **lda_best_model_vis.html:** Interactive pyLDAvis dashboard for the best model.
