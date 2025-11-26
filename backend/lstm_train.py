# lstm_train.py
import os, json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

def load_corpus(folder):
    texts = []
    for fname in os.listdir(folder):
        if fname.lower().endswith('.txt'):
            with open(os.path.join(folder, fname), 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read().lower())
    return "\n".join(texts)

def make_sequences(corpus, tokenizer, seq_len=20):
    tokens = tokenizer.texts_to_sequences([corpus])[0]
    sequences = []
    for i in range(seq_len, len(tokens)):
        seq = tokens[i-seq_len:i+1]  # input tokens + target
        sequences.append(seq)
    sequences = np.array(sequences)
    X, y = sequences[:,:-1], sequences[:,-1]
    return X, y

def build_model(vocab_size, seq_len, embed_dim=100, lstm_units=256):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim, input_length=seq_len))
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model

def train(folder='data', seq_len=20, epochs=20, batch_size=128, out='lstm_model'):
    corpus = load_corpus(folder)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([corpus])
    vocab_size = len(tokenizer.word_index) + 1
    X, y = make_sequences(corpus, tokenizer, seq_len=seq_len)
    model = build_model(vocab_size, seq_len)
    checkpoint = ModelCheckpoint(out + '.h5', monitor='loss', save_best_only=True, verbose=1)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
    # save tokenizer
    with open(out + '_tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer.to_json()))
    print("Saved model and tokenizer.")

if __name__ == '__main__':
    train(folder='data', seq_len=20, epochs=10, batch_size=128, out='lstm_model')
