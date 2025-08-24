# Respiratory Disease Classification System (Backend Scaffold)

## Quickstart

1. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create environment file
```bash
cp .env.example .env
```

4. Run development server
```bash
FLASK_ENV=development FLASK_DEBUG=1 python wsgi.py
```

5. Health check
```bash
curl http://localhost:5000/health
```

## Notes
- Default DB is SQLite in `instance/app.db`. Override with `DATABASE_URL`.
- Heavy ML packages are listed but not required to boot the API.