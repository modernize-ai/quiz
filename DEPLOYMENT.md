# Deployment & Scaling Guide

## Quick Deployment

### Option 1: Streamlit Cloud (Easiest)

1. **Push to GitHub**
```bash
git add .
git commit -m "Initial public quiz app"
git push origin main
```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repo
   - Set main file: `quiz_app_public.py`
   - Deploy!

**URL Format:** `https://[username]-public-quiz.streamlit.app`

### Option 2: Docker on VPS

```bash
# Build image
docker build -t public-quiz:latest .

# Run container
docker run -d \
  --name quiz-app \
  -p 8501:8501 \
  -v $(pwd)/results:/app/results \
  public-quiz:latest

# View logs
docker logs -f quiz-app
```

### Option 3: Docker Compose

```bash
docker-compose up -d
```

Access at `http://localhost:8501`

## Production Checklist

- [ ] Results directory has write permissions
- [ ] Email validation is working
- [ ] CSV export functions properly
- [ ] Streamlit caching configured
- [ ] Environment variables set
- [ ] HTTPS enabled (via reverse proxy)
- [ ] Backups of results/ directory scheduled

## Performance Optimization

### Streamlit Caching

```python
@st.cache_data
def load_questions():
    return QUESTIONS

@st.cache_resource
def init_session():
    # Initialize session state
    pass
```

### Database Backend (Advanced)

For production, replace CSV with database:

```python
import sqlite3

def save_results_db(email, score, total, pct):
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO results (email, score, total, percentage, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (email, score, total, pct, datetime.now()))
    conn.commit()
    conn.close()
```

## Scaling

### For 1000+ Users/Day

1. **Database**: Migrate from CSV to PostgreSQL
2. **Caching**: Use Redis for session management
3. **Load Balancer**: Nginx or AWS ELB
4. **CDN**: Cloudflare for static assets
5. **Monitoring**: Prometheus + Grafana

### Infrastructure (AWS Example)

```yaml
- ECS Fargate: 3 containers (auto-scaling)
- RDS PostgreSQL: Results storage
- CloudFront: CDN
- Route53: DNS
- CloudWatch: Monitoring
```

## Security

### Environment Variables

```bash
# .env
ADMIN_PASSWORD=secure_password_here
DATABASE_URL=postgresql://user:pass@host/db
EMAIL_API_KEY=your_key
```

### Input Validation

```python
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
```

## Monitoring & Alerts

### Health Check Endpoint

```python
@st.cache_data
def health_check():
    try:
        # Verify database connection
        # Verify file permissions
        return {"status": "healthy"}
    except:
        return {"status": "unhealthy"}
```

## Backup Strategy

```bash
#!/bin/bash
# Daily backup
tar -czf backups/results_$(date +%Y%m%d).tar.gz results/
aws s3 cp backups/ s3://my-quiz-backups/
```

## API Integration

To integrate with external systems:

```python
import requests

def send_results_webhook(email, score, total):
    webhook_url = os.getenv("WEBHOOK_URL")
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

### App crashes on deploy
- Check logs: `docker logs quiz-app`
- Verify environment variables set
- Test locally first

### Results not saving
- Check write permissions on results/ directory
- Verify disk space available
- Check for locked files

### Slow performance
- Enable Streamlit caching
- Reduce question count for initial load
- Use database instead of CSV

---

**Need help?** Check Streamlit docs: https://docs.streamlit.io
