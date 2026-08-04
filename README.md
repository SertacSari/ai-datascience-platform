# BasitAnaliz

BasitAnaliz is a local data analysis web application built for the CS395 project. It lets a user register, upload tabular datasets, inspect data quality, clean common dataset issues, create analysis jobs, and run a first real classification training flow.

The project is currently focused on a clear end-to-end vertical slice rather than a full production analytics platform. Classification training is implemented. Regression training, forecasting training, report generation, and AI explanations are planned later.

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
- Classification ML readiness checks before training
- Classification model training with scikit-learn
- Saved classification results:
  - accuracy
  - precision
  - recall
  - F1 score
  - class distribution
  - confusion matrix
  - classification report
- React dashboard for upload, cleaning, job creation, job running, and saved result viewing

## Tech Stack

### Backend

- Python
- FastAPI
- SQLAlchemy
- PostgreSQL
- Pydantic
- pandas
- scikit-learn
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
```

Create database tables:

```bash
cd backend
python create_tables.py
```

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
Backend tests: 62 passed
Frontend build: passed
```

## Manual Smoke Test

Use the included sample dataset:

```text
data/mock_datasets/customer_churn_classification.csv
```

Recommended flow:

1. Register or log in.
2. Upload the churn classification CSV.
3. Confirm preview and cleaning cards show backend data.
4. Create a classification job.
5. Choose `churned` as the target column.
6. Run the job.
7. Confirm the job becomes `completed`.
8. Confirm classification metrics and tables are shown.
9. Refresh the page.
10. Click `View result` on the completed job.
11. Confirm the saved result loads again.

## Security Notes

- The frontend does not store the JWT in `localStorage` or `sessionStorage`.
- The backend sends the auth token through an HttpOnly cookie.
- The login response does not expose `access_token` in JSON.
- Bearer token authentication is still supported as a backend fallback for compatibility.
- Local uploaded files are ignored by Git.
- `.env` files are ignored by Git.

## Current Limitations

- Classification training is implemented; regression and forecasting training are not implemented yet.
- The saved model artifact is not persisted yet; the app currently persists model results and metrics.
- AI explanation generation is not active yet.
- Report generation is not part of the current completed flow.
- The app is intended to run locally during this project phase.

## Planned Next Work

- Add regression training.
- Add forecasting training.
- Improve result history and report views.
- Add an explanation layer that can translate technical ML results into plain-language summaries.
- Add local DevOps/CI practice, such as automated test and build checks.
