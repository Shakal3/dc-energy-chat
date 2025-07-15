# DC Energy Chat - OpenInfra Power Forecast Assistant

A Flask-based chat API that ingests OpenInfra power telemetry data and provides cost forecasting through natural language interactions.

## 🚀 Quick Start Guide

### 1. **Environment Setup**
```bash
# Copy environment template and customize
copy env.example .env

# Edit .env with your settings (database passwords, API keys, etc.)
# The defaults work for local development
```

### 2. **Install Dependencies**
```bash
# Activate your virtual environment
.venv\Scripts\activate  # Windows
# or source .venv/bin/activate  # Linux/Mac

# Install Python packages
pip install -r requirements.txt
```

### 3. **Test the Application**
```bash
# Run tests to verify everything works
pytest -v

# Start Flask development server
python -m chat_api.app
# ➜ API available at http://localhost:5000
```

### 4. **Test the Chat Endpoint**
```bash
# Test the /chat endpoint
curl -X POST http://localhost:5000/chat -H "Content-Type: application/json" -d "{\"query\":\"What are my energy costs?\"}"

# Expected response:
# {"ok": true, "query_received": "What are my energy costs?", "response": "Chat functionality coming soon!", ...}
```

## 🐳 **Docker Setup (Full Stack)**

### 1. **Start with Docker Compose**
```bash
# Build and start all services (API + TimescaleDB + DCIM placeholder)
docker compose up -d

# Check logs
docker compose logs -f

# Stop when done
docker compose down
```

### 2. **Docker Troubleshooting**
Since your Docker is in `S:\Docker\WSL\DockerDesktopWSL`, make sure:
- Docker Desktop is running
- WSL integration is enabled
- Your project folder is accessible from WSL

## 📊 **OpenInfra Integration Guide**

### What is OpenInfra?
OpenInfra (Open Infrastructure) is a data center infrastructure monitoring platform. For this project, we're expecting CSV telemetry data with power consumption metrics.

### 1. **Sample OpenInfra CSV Format**
Create sample data in `data/openinfra/sample_telemetry.csv`:
```csv
timestamp,device_id,location,data_center,power_kw,voltage_v,current_a
2024-01-01T00:00:00Z,server-001,rack-a1,dc-north,2.5,220,11.4
2024-01-01T00:05:00Z,server-001,rack-a1,dc-north,2.7,220,12.3
2024-01-01T00:10:00Z,server-002,rack-a2,dc-north,3.1,220,14.1
```

### 2. **Load OpenInfra Data**
```python
from ingestion.openinfra_csv_loader import OpenInfraCSVLoader

# Load CSV data
loader = OpenInfraCSVLoader(data_dir="./data/openinfra")
df = loader.load_csv("sample_telemetry.csv")
print(df.head())
```

### 3. **Real OpenInfra Setup**
If you have access to actual OpenInfra APIs:
1. Update `OPENDCIM_API_URL` and `OPENDCIM_API_KEY` in `.env`
2. Implement API integration in `ingestion/openinfra_csv_loader.py`
3. Set up automated data sync to TimescaleDB

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Chat API      │    │   TimescaleDB   │    │   OpenInfra     │
│   (Flask)       │───▶│   (Time-series) │◀───│   (Data Source) │
│   Port 5000     │    │   Port 5432     │    │   CSV/API       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│  Forecast       │    │  Cost Engine    │
│  Engine         │    │  (ARIMA/ML)     │
└─────────────────┘    └─────────────────┘
```

## 📁 **Project Structure**
```
dc-energy-chat/
├─ chat_api/           # Flask API and routes
├─ forecast/           # Cost forecasting engine
├─ ingestion/          # Data loading from OpenInfra
├─ tests/              # Unit and integration tests
├─ docker/             # Docker configurations
├─ data/               # CSV files and datasets
├─ logs/               # Application logs
└─ .vscode/            # Cursor/VSCode tasks
```

## 🛠️ **Development Workflow**

### Using Cursor Command Runner
Press `Ctrl+Shift+P` → "Tasks: Run Task" → Select:
- **`run-api`** - Start Flask development server
- **`run-tests`** - Run pytest suite
- **`docker-up`** - Start Docker containers
- **`docker-down`** - Stop Docker containers
- **`install-deps`** - Install Python dependencies

### Next Implementation Steps
1. **Implement Real Forecasting** in `forecast/cost_engine.py`
2. **Add AI Chat Logic** in `chat_api/routes.py`
3. **Connect OpenInfra Data** to TimescaleDB
4. **Add Authentication** and rate limiting

## 🔧 **Troubleshooting**

### Common Issues:
1. **Import Errors**: Make sure `PYTHONPATH` includes project root
2. **Docker Connection**: Verify Docker Desktop is running
3. **Database Connection**: Check TimescaleDB is started (`docker compose ps`)
4. **Port Conflicts**: Change ports in `docker-compose.yml` if needed

### Debug Commands:
```bash
# Check if services are running
docker compose ps

# View container logs
docker compose logs api
docker compose logs timescale

# Test database connection
docker compose exec timescale psql -U dcenergy -d dcenergy_db -c "\dt"
```

## 📈 **What's Working Now**
✅ Flask API with `/chat` endpoint  
✅ TimescaleDB time-series database  
✅ Docker containerization  
✅ Unit test framework  
✅ OpenInfra CSV loader structure  
✅ Cost forecasting engine skeleton  

## 🚧 **Coming Next**
🔄 Real AI-powered chat responses  
🔄 ARIMA/ML forecasting models  
🔄 OpenInfra API integration  
🔄 Interactive web UI  
🔄 Production deployment configs  

---

**Ready to start developing!** 🚀 Run `pytest` and `python -m chat_api.app` to verify everything works.