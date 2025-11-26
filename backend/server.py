# server.py
from flask import Flask, request, jsonify, send_from_directory
import os, json
from markov_bot import HybridBot
# Optional imports for LSTM
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    import numpy as np
    HAS_TF = True
except Exception:
    HAS_TF = False

app = Flask(__name__, static_folder='../frontend', static_url_path='')

MODEL_TYPE = os.environ.get('MODEL_TYPE', 'markov')  # 'markov' or 'lstm'
MODEL_PATH = os.environ.get('MODEL_PATH', 'backend/model.pkl')

bot = None
lstm = None
tokenizer = None
lstm_model = None

if MODEL_TYPE == 'markov':
    bot = HybridBot(order=2)
    if os.path.exists(MODEL_PATH):
        bot.load(MODEL_PATH)
    else:
        print("No markov model saved yet. Build with markov_bot.HybridBot.build()")

if MODEL_TYPE == 'lstm' and HAS_TF:
    # expects files lstm_model.h5 and lstm_model_tokenizer.json
    try:
        lstm_model = load_model('backend/lstm_model.h5')
        with open('backend/lstm_model_tokenizer.json','r',encoding='utf-8') as f:
            tok_json = f.read()
        tokenizer = tokenizer_from_json(tok_json)
    except Exception as e:
        print("Unable to load LSTM model:", e)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    msg = data.get('message','').strip()
    if not msg:
        return jsonify({'reply': "Say something."})

    if MODEL_TYPE == 'markov':
        reply = bot.respond(msg)
        return jsonify({'reply': reply})

    if MODEL_TYPE == 'lstm' and HAS_TF and lstm_model:
        # simple next-word generation using sliding window
        seq_len = lstm_model.input_shape[1]
        # tokenize input
        toks = tokenizer.texts_to_sequences([msg])[0]
        # pad/truncate
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        toks = pad_sequences([toks], maxlen=seq_len, truncating='pre')
        # generate words
        inv_map = {v:k for k,v in tokenizer.word_index.items()}
        out_words = []
        for _ in range(30):
            preds = lstm_model.predict(toks, verbose=0)[0]
            idx = np.argmax(preds)
            word = inv_map.get(idx)
            if not word: break
            out_words.append(word)
            # append and slide
            toks = np.append(toks[:,1:], [[idx]], axis=1)
        reply = " ".join(out_words)
        return jsonify({'reply': reply})

    return jsonify({'reply': "Model not ready. Check server logs."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
