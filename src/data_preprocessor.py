import re
import logging
import pandas as pd
import nltk
import spacy
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from typing import List, Tuple

# Download NLTK stopwords if they aren't there
try:
    STOPWORDS = nltk.corpus.stopwords.words('english')
except LookupError:
    logging.info("Stopwords not found. Grabbing them now...")
    nltk.download('stopwords')
    STOPWORDS = nltk.corpus.stopwords.words('english')

# Load Spacy model (disable stuff we don’t use)
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    logging.info("Spacy model missing. Downloading en_core_web_sm...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


class DataPreprocessor:
    def __init__(self):
        self.df = None
        self.stopwords = STOPWORDS.copy()
        self.stopwords.extend(["from", "subject", "re", "edu", "use"])
        
        logging.info("DataPreprocessor ready to roll.")

    def load_and_clean_data(self, csv_path: str, text_col: str, title_col: str, save_clean_path: str = None) -> List[str]:
        logging.info(f"Reading CSV from {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        self.df = self.df.drop_duplicates(subset=[text_col, title_col])
        
        self.df[title_col] = self.df[title_col].fillna("")
        self.df[text_col] = self.df[text_col].fillna("")
        combined_text = (self.df[title_col] + " " + self.df[text_col]).tolist()
        # combined_text = self.df[text_col].values.tolist()

        # Clean HTML, code, punctuation, numbers
        cleaned_text = []
        for t in combined_text:
            t = re.sub(r"<pre><code>.*?</code></pre>", "", t, flags=re.DOTALL)  # code blocks
            t = re.sub(r"<code>.*?</code>", "", t, flags=re.DOTALL)  # inline code
            t = re.sub(r"<.*?>", "", t)  # HTML tags
            t = re.sub(r"http\S+", "", t)  # URLs
            t = re.sub(r"\s+", " ", t)  # spaces
            t = re.sub(r"\'", "", t)  # apostrophes
            t = re.sub(r"[\d]", "", t)  # numbers
            t = re.sub(r"[^\w\s]", "", t)  # punctuation
            cleaned_text.append(t.strip())

        self.df["processed_text"] = cleaned_text
        self.df = self.df.drop_duplicates(subset=["processed_text"])
        
        if save_clean_path:
            save_clean_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_clean_path, "w", encoding="utf-8") as f:
                for doc in cleaned_text:
                    f.write(doc + "\n")
            logging.info(f"Saved cleaned text for EDA at: {save_clean_path}")

        logging.info(f"Finished cleaning. Final dataset size: {self.df.shape}")
        
        return self.df["processed_text"].tolist()

    def tokenize_and_lemmatize(self, texts: List[str]):
        logging.info("Starting tokenization + lemmatization...")
        
        tokenized_words = [simple_preprocess(str(doc), deacc=True) for doc in texts]

        bigram_model = gensim.models.Phrases(tokenized_words, min_count=3, threshold=10)
        trigram_model = gensim.models.Phrases(bigram_model[tokenized_words], min_count=2, threshold=5)
        bigram_mod = gensim.models.phrases.Phraser(bigram_model)
        trigram_mod = gensim.models.phrases.Phraser(trigram_model)

        no_stopwords = [[w for w in doc if w not in self.stopwords] for doc in tokenized_words]
        
        with_bigrams = [bigram_mod[doc] for doc in no_stopwords]
        with_trigrams = [trigram_mod[bigram_mod[doc]] for doc in no_stopwords]
        
        allowed_tags = {"NOUN", "ADJ", "VERB", "ADV", "PROPN"}
        lemmatized_texts = []
        for sent in with_trigrams:
            doc = nlp(" ".join(sent))
            lemmatized_texts.append([tok.lemma_ for tok in doc if tok.pos_ in allowed_tags])
        
        id2word = corpora.Dictionary(lemmatized_texts)
        id2word.filter_extremes(no_below=5, no_above=0.5)

        corpus = [id2word.doc2bow(text) for text in lemmatized_texts]
        
        logging.info("Tokenization + lemmatization done.")
        logging.info(f"Dictionary size after filtering: {len(id2word)} unique tokens")

        return lemmatized_texts, id2word, corpus
