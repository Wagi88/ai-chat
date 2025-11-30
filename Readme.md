# 🧠 Termux AI — Local Student Chatbot  
Built and trained inside Termux by students of the National University of Samoa  

A lightweight, fully local AI chatbot — trained with plain text, no external APIs.  
Uses Python for backend logic and HTML/CSS/JS for the hacker-style frontend.  

---

# 🚀 Features  
• No API dependencies — 100% offline chatbot you can train yourself  
• Python backend with a hybrid Markov + TF-IDF model  
• Beautiful chat UI built with HTML/CSS/JS (dark hacker theme)  
• Custom training — feed it your own .txt files to make it learn  
• Runs in Termux, localhost, or any Python environment  
• Designed for NUS students — learn AI, code, and cybersecurity in one project  

---

# 🧩 Project Structure  

#Update & install python, git, build tools
pkg update && pkg upgrade -y
pkg install python git clang fftw libjpeg-turbo -y   # libs helpful for pip wheels

#Create venv
python -m venv ~/ai-chat-venv
source ~/ai-chat-venv/bin/activate

#Install lightweight dependencies for Markov app
pip install flask numpy scikit-learn

#If you want to try LSTM (heavy):
#pip install tensorflow    # may be large and fail on some Termux/Android setups
#pip install flask numpy tensorflow scikit-learn



ai-chat/  
├── backend/  
│   ├── markov_bot.py      - Core AI model (Markov + TF-IDF hybrid)  
│   ├── server.py          - Flask API for frontend chat  
│   ├── lstm_train.py      - (Optional) Neural LSTM model trainer  
│   ├── data/              - Folder for your training text files  
│   │   └── day1.txt  
│   └── model.pkl          - Saved trained model  
└── frontend/  
    └── index.html         - Chat UI (hacker/developer style)  



cat <<EOL > README.md
# AI Chat

A simple AI chatbot project with a frontend and backend structure.  
This project is designed to run locally in **Termux** using Python and HTML/JS.

## Folder Structure

\`\`\`
ai-chat/
  backend/
    server.py       # Main backend server (Flask)
    markov_bot.py   # Simple AI bot logic (Markov chain)
    lstm_train.py   # LSTM training script (optional)
  frontend/
    index.html      # Chat interface (HTML/CSS/JS)
\`\`\`

## Requirements

- Python 3
- Flask (\`pip install flask\`)
- Optional: TensorFlow or PyTorch for LSTM training

## Setup in Termux

1. Navigate to your project folder:
\`\`\`bash
cd ~/ai-chat
\`\`\`

2. Install dependencies:
\`\`\`bash
pkg install python -y
pip install flask
\`\`\`

3. Run the server:
\`\`\`bash
python3 backend/server.py
\`\`\`

4. Open \`frontend/index.html\` in a browser to use the chat interface (or serve via Flask).

## Usage

- \`server.py\` runs the backend server.  
- \`markov_bot.py\` contains simple AI chat logic.  
- \`lstm_train.py\` is for training an AI model using text data.  
- \`index.html\` is the frontend interface where you can chat with your AI bot.

## Notes

- Designed for local use in Termux.  
- Fully customizable AI logic and UI.  
- Ideal for learning how AI chatbots work end-to-end.


# 🧠 Training Your Model  

1. Create your first training file  
mkdir -p backend/data  
nano backend/data/day1.txt  

Add your own sentences, developer text, or notes.  
Example content:  
Today marks day one of training my AI system inside Termux.  
I started building the chatbot from scratch using Python and HTML.  

2. Train your AI model  
cd ~/ai-chat  
python - <<'PY'  
from backend.markov_bot import HybridBot  
b = HybridBot(order=2)  
b.build('backend/data')  
b.save('backend/model.pkl')  
print("✅ Training complete — model saved as backend/model.pkl")  
PY  

---

# 💬 Run the Chat Server  

cd backend  
export MODEL_TYPE=markov  
export MODEL_PATH=backend/model.pkl  
python server.py  

Then open your browser and go to:  
http://127.0.0.1:5050/  

You’ll see your local AI chatbot in action 😎  

---

# 🧠 How It Learns  

This AI uses two simple but powerful systems:  
• Markov chains — to learn how words connect naturally  
• TF-IDF retrieval — to find relevant ideas in your training data  

Each .txt file you add teaches it new vocabulary, tone, and logic.  
No categories or Q&A formats are required — it learns directly from your writing style.  

---

# 🎨 UI Design  

• Theme: Dark Hacker / Cyber Developer  
• Colors: Blue, White, and Green (NUS student colors)  
• Tech look inspired by GitHub and Termux styling  

---

#  Future Upgrades  

☑ Add LSTM neural model support  
☑ Add conversation memory system  
☑ Add option to retrain directly from the UI  
☑ Add encryption for local model files  

---

# 🧑‍💻 Created By  

Wagi88 

“Learning AI from scratch — one line of Python at a time.”  

Built proudly inside Termux, for students and developers everywhere.  

---

# 🛡️ License  

MIT License — free to modify, share, and build on.  
Just give credit and keep learning 🧠✨
