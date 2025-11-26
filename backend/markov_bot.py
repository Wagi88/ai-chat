# markov_bot.py
import os
import random
import re
from collections import defaultdict, deque
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

class MarkovModel:
    def __init__(self, order=2):
        self.order = order
        self.model = defaultdict(list)

    def _tokenize(self, text):
        # simple tokenization keep punctuation as tokens
        tokens = re.findall(r"\w+|[^\s\w]", text, flags=re.UNICODE)
        return tokens

    def feed(self, text):
        tokens = self._tokenize(text)
        dq = deque(maxlen=self.order)
        for t in tokens:
            if len(dq) == self.order:
                key = tuple(dq)
                self.model[key].append(t)
            dq.append(t)

    def generate(self, seed=None, max_words=80):
        if seed:
            tokens = self._tokenize(seed)
            dq = deque(tokens[-self.order:], maxlen=self.order)
        else:
            # pick a random starting key
            if not self.model:
                return ""
            key = random.choice(list(self.model.keys()))
            dq = deque(list(key), maxlen=self.order)

        out = list(dq)
        for _ in range(max_words):
            key = tuple(dq)
            options = self.model.get(key)
            if not options:
                break
            nxt = random.choice(options)
            out.append(nxt)
            dq.append(nxt)
        return " ".join(out)

def load_corpus_from_folder(folder):
    texts = []
    for fname in os.listdir(folder):
        if fname.lower().endswith('.txt'):
            with open(os.path.join(folder, fname), 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read())
    return texts

class HybridBot:
    def __init__(self, order=2):
        self.markov = MarkovModel(order=order)
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.corpus = []
        self.doc_vectors = None

    def build(self, folder):
        texts = load_corpus_from_folder(folder)
        if not texts:
            raise ValueError("No .txt files in folder.")
        self.corpus = texts
        for t in texts:
            self.markov.feed(t)
        # build TF-IDF for retrieval
        self.doc_vectors = self.vectorizer.fit_transform(texts)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'order': self.markov.order,
                'model': dict(self.markov.model),
                'vectorizer': self.vectorizer,
                'corpus': self.corpus,
                'doc_vectors': self.doc_vectors
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.markov.order = data['order']
        self.markov.model = defaultdict(list, data['model'])
        self.vectorizer = data['vectorizer']
        self.corpus = data['corpus']
        self.doc_vectors = data['doc_vectors']

    def respond(self, user_text, top_k=3):
        # retrieval step
        q_vec = self.vectorizer.transform([user_text])
        sims = cosine_similarity(q_vec, self.doc_vectors).flatten()
        best_idx = sims.argsort()[-top_k:][::-1]
        # gather candidate sentences from top documents
        candidates = []
        for idx in best_idx:
            doc = self.corpus[idx]
            sents = re.split(r'(?<=[.!?])\s+', doc)
            candidates.extend(sents[-10:])  # use last few sentences as context
        # pick the best sentence by simple heuristic (cosine on sentence-level)
        if candidates:
            cvecs = self.vectorizer.transform(candidates)
            cs = cosine_similarity(q_vec, cvecs).flatten()
            chosen = candidates[cs.argmax()]
            # combine with Markov continuation
            markup = self.markov.generate(seed=chosen, max_words=40)
            # small cleanup
            response = re.sub(r'\s([.,!?;:])', r'\1', markup)
            return response
        # fallback to pure Markov
        return self.markov.generate(seed=user_text, max_words=40)
