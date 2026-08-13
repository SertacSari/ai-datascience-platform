# BasitAnaliz

BasitAnaliz is a local data analysis web application built for the CS395 project. It lets a user register, upload tabular datasets, inspect data quality, clean common dataset issues, create and run analysis jobs, view saved ML results, and generate local plain-language AI explanations.

The project is designed as a local end-product prototype rather than a cloud-deployed analytics platform. Classification, regression, forecasting, deterministic result guidance, and local Gemma/Ollama explanations are implemented. Report/PDF generation and advanced dashboard polishing remain future work.

## Current Features

- User registration and login with FastAPI authentication
- HttpOnly cookie-based session handling
- CSV, XLS, and XLSX upload
- Dataset preview table
- Cleaning report with missing values, duplicate rows, and column type signals
- Dataset cleaning endpoint
- Analysis job creation for:
  - classification
  - regression
  - forecasting
- Forecasting job validation with a required date column
- ML readiness checks before job creation and training
- Classification, regression, and forecasting training with scikit-learn
- Saved classification results:
  - accuracy
  - precision
  - recall
  - F1 score
  - class distribution
  - confusion matrix
  - classification report
- Saved regression and forecasting results:
  - MAE
  - RMSE
  - R2 score
  - prediction sample
  - deterministic quality level, warnings, and recommended actions
- Local Gemma/Ollama AI explanation layer using safe summarized result data
- React dashboard for upload, cleaning, job creation, job running, saved result viewing, and cached AI explanations

## Tech Stack

### Backend

- Python
- FastAPI
- SQLAlchemy
- PostgreSQL
- Pydantic
- pandas
- scikit-learn
- httpx
- python-jose
- passlib/bcrypt
- pytest

### Frontend

- React
- Vite
- react-router-dom
- Fetch API
- Plain CSS

## Project Structure

```text
backend/
  app/
    models/       Database models
    routers/      FastAPI route handlers
    schemas/      Pydantic request/response schemas
    services/     Business logic, dataset logic, and ML training logic
  migrations/     SQL migration files
  tests/          Backend test suite
  uploads/        Local uploaded files, ignored except .gitkeep

frontend/
  src/
    api/          Frontend API client
    components/   Reusable UI components
    context/      Session and dashboard state
    lib/          Small frontend helper data/functions
    pages/        Login and dashboard pages

data/
  mock_datasets/  Small sample datasets for local testing
```

## Local Setup

### 1. Clone the project

```bash
git clone <repository-url>
cd cs395-project
```

### 2. Backend environment

Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

Create the backend environment file:

```bash
cp backend/.env.example backend/.env
```

Fill in at least these values in `backend/.env`:

```text
DATABASE_URL=postgresql://USER:PASSWORD@localhost:5432/DATABASE_NAME
SECRET_KEY=replace-with-a-local-secret
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
AUTH_COOKIE_NAME=basitanaliz_access_token
AUTH_COOKIE_SECURE=false
UPLOAD_DIR=uploads
MAX_UPLOAD_SIZE_MB=50
MAX_DATAFRAME_MEMORY_MB=200
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=gemma3:4b
AI_EXPLANATION_ENABLED=false
AI_EXPLANATION_TIMEOUT_SECONDS=30
```

Create database tables and apply migrations:

```bash
cd backend
python create_tables.py
set -a
source .env
set +a
psql "$DATABASE_URL" -f migrations/001_add_cleaned_file_path.sql
psql "$DATABASE_URL" -f migrations/002_harden_analysis_jobs.sql
psql "$DATABASE_URL" -f migrations/003_unique_model_result_per_analysis.sql
psql "$DATABASE_URL" -f migrations/004_unique_ai_explanation_per_analysis.sql
```

For local AI explanations, install Ollama and download the model:

```bash
ollama pull gemma3:4b
```

Then set `AI_EXPLANATION_ENABLED=true` in `backend/.env` and keep Ollama running locally while using the explanation feature.

Run the backend:

```bash
python -m uvicorn app.main:app --reload
```

The backend runs at:

```text
http://localhost:8000
```

### 3. Frontend setup

In a new terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at:

```text
http://localhost:5173
```

## Testing

Run backend tests from the project root:

```bash
PYTHONDONTWRITEBYTECODE=1 ./venv/bin/python -m pytest backend/tests -q
```

Build the frontend:

```bash
cd frontend
npm run build
```

The latest checked state passed:

```text
Backend tests: 108 passed
Frontend build: passed
```

## Manual Smoke Test

Use the included sample datasets:

```text
data/mock_datasets/customer_churn_classification.csv
data/mock_datasets/house_price_regression.csv
data/mock_datasets/daily_sales_forecasting.csv
```

Recommended flow:

1. Register or log in.
2. Upload the churn classification CSV.
3. Confirm preview and cleaning cards show backend data.
4. Clean one dataset.
5. Create and run classification, regression, and forecasting jobs.
6. Confirm each job becomes `completed`.
7. Confirm metrics, deterministic guidance, and result tables are shown.
8. Generate a local AI explanation for one completed result.
9. Refresh the page.
10. Click `View result` on the completed job.
11. Confirm the saved result and cached AI explanation load again.

## Security Notes

- The frontend does not store the JWT in `localStorage` or `sessionStorage`.
- The backend sends the auth token through an HttpOnly cookie.
- The login response does not expose `access_token` in JSON.
- Bearer token authentication is still supported as a backend fallback for compatibility.
- Local uploaded files are ignored by Git.
- `.env` files are ignored by Git.
- Gemma/Ollama receives only summarized ML result facts, not full datasets, raw CSV rows, upload paths, tokens, passwords, or secrets.

## Current Limitations

- The saved model artifact is not persisted yet; the app currently persists model results and metrics.
- Report generation is not part of the current completed flow.
- The app is intended to run locally; it is not cloud-deployed.
- The dataset overview chart is still a placeholder and should be replaced or removed in a final polish pass.

## Planned Next Work

- Improve final frontend layout symmetry and visual polish.
- Add smart task/target/date-column recommendations.
- Replace the placeholder dataset chart with real dataset insight charts.
- Add report/PDF generation if required for the final deliverable.
- Add local DevOps/CI practice, such as automated test and build checks.
