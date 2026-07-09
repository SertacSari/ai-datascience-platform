# Swagger Manual Test Checklist

Run the API from the project root, then open `http://127.0.0.1:8000/docs`.

```bash
cd backend
../venv/bin/python -m uvicorn app.main:app --reload
```

Use a small CSV with at least two rows, a numeric column, a categorical target
with at least two classes, and a few missing or duplicate values.

1. **Register** — `POST /auth/register`
   - Create a user with a unique username and email.
   - Confirm the response contains no password or password hash.
2. **Login** — `POST /auth/login`
   - Enter the registered email in the OAuth2 `username` field.
   - Confirm an access token is returned.
3. **Authorize and inspect the current user**
   - Use Swagger's **Authorize** button with the same email and password.
   - Call `GET /auth/me` and confirm it returns the registered user.
4. **Upload** — `POST /datasets/upload`
   - Upload the CSV and confirm filename, row count, and column count.
5. **Preview** — `GET /datasets/{dataset_id}/preview`
   - Confirm rows, columns, dtypes, missing-value counts, duplicate count, and
     summary statistics are returned.
6. **Cleaning report** — `GET /datasets/{dataset_id}/cleaning-report`
   - Confirm issues are reported without creating or changing a cleaned file.
7. **Clean** — `POST /datasets/{dataset_id}/clean`
   - Confirm duplicates are removed and a cleaned copy is created without
     overwriting the original upload.
8. **Create an analysis job** — `POST /analysis/jobs`
   - Submit `dataset_id`, `task_type`, and `target_column`.
   - For forecasting, also submit `config_json.date_column`.
   - Confirm the response status is `created`; no model training should run.
9. **List jobs** — `GET /analysis/jobs`
   - Confirm only the authorized user's jobs are returned.
10. **Get one job** — `GET /analysis/jobs/{job_id}`
    - Confirm the created job is returned.

For ownership checks, register a second user and confirm the first user's
dataset and job identifiers return `404` when requested as the second user.
