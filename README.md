# 🧠 Public AI & Tech Quiz

A comprehensive, interactive quiz application covering AI, Machine Learning, LLMs, RAG, Embeddings, and more.

## Features

✨ **100+ Questions** across 14 topics  
📊 **Real-time Scoring** with topic-wise breakdown  
📧 **Email Results** - Share your performance  
📥 **CSV Export** - Download detailed results  
🎯 **Customizable** - Filter by topic and question type  
🎲 **Shuffle Mode** - Randomize question order  
🧑‍💻 **Public & Free** - No personal context, open to all  

## Topics Covered

- AI Fundamentals
- LLMs & Transformers
- RAG & Vector Search
- Embeddings
- Vector Databases
- APIs & Integration
- Data & Preprocessing
- Model Evaluation
- NLP & Text Processing
- Production & Deployment
- Prompt Engineering
- Agents & Tools
- Ethics & Safety
- Emerging Trends

## Installation

### Local Setup

```bash
# Clone or download this repository
cd quiz_app_public

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run quiz_app_public.py
```

The app will open at `http://localhost:8501`

### Docker Setup

```bash
docker build -t public-quiz .
docker run -p 8501:8501 public-quiz
```

## Usage

1. **Select Topics**: Choose which AI/ML topics to focus on
2. **Adjust Settings**: Set number of questions, question type, shuffle preference
3. **Answer Questions**: Multiple choice or fill-in-the-blank
4. **View Results**: Get instant feedback with explanations
5. **Share Results**: Enter email to publish and download results

## Project Structure

```
quiz_app_public/
├── quiz_app_public.py      # Main Streamlit app
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── Dockerfile              # Docker configuration (optional)
├── .gitignore             # Git ignore rules
└── results/               # Saved quiz results (auto-created)
    └── quiz_results.csv   # Results log
```

## Results Storage

When you publish results with your email:
- A CSV log is saved in the `results/` folder
- You can download individual quiz results as CSV
- Results include: timestamp, email, score, topic breakdown

## Configuration

### Environment Variables (Optional)

```bash
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true
```

### Streamlit Config

Edit `~/.streamlit/config.toml` to customize:

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#F7F7F7"
secondaryBackgroundColor = "#E8E8E8"
textColor = "#262730"
font = "sans serif"
```

## API Integration (Advanced)

To integrate with external systems:

```python
# Send results to webhook
import requests

def send_results_webhook(email, score, total, webhook_url):
    payload = {
        "email": email,
        "score": score,
        "total": total,
        "percentage": round(score/total*100),
        "timestamp": datetime.now().isoformat()
    }
    requests.post(webhook_url, json=payload)
```

## Troubleshooting

### Streamlit not found
```bash
pip install streamlit --upgrade
```

### Port 8501 already in use
```bash
streamlit run quiz_app_public.py --server.port 8502
```

### Results folder not found
The app auto-creates it on first result save. Ensure write permissions in the app directory.

## Deployment Options

### Streamlit Cloud (Recommended for beginners)

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repo
4. Deploy with one click

### Heroku

```bash
git push heroku main
```

(Requires `Procfile` and `setup.sh`)

### AWS/GCP/Azure

- Use containers (Docker)
- Deploy to App Engine, Cloud Run, or App Service
- Set up environment variables for scaling

## Contributing

Pull requests welcome! Areas to improve:

- Add more questions
- Improve UI/UX
- Add database backend for results
- Implement user authentication
- Add progress saving

## License

MIT License - Free to use, modify, distribute

## Support

- 📧 Email: support@quizapp.com
- 🐛 Report bugs: GitHub Issues
- 💡 Suggest features: GitHub Discussions

## Changelog

### v1.0 (2024-05-17)
- ✅ 100+ questions across 14 topics
- ✅ Email results publishing
- ✅ CSV export functionality
- ✅ Real-time scoring & feedback
- ✅ Topic-wise breakdown

---

**Happy Learning! 🚀**
