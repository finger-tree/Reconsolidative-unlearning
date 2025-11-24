# Repo-specific instructions for AI coding agents

These notes help an AI coding agent become productive quickly in this repository.

1) Big picture
- Purpose: a minimal FastAPI service that serves an Iris classifier (scikit-learn RandomForest) and is Docker-ready.
- Main components:
  - `main.py` — FastAPI app and inference endpoint (`/predict`). Model is loaded at module import.
  - `train_model.py` — trains and saves the scikit-learn model to `app/iris_model.pkl`.
  - `app/` — contains the serialized model (`iris_model.pkl`).
  - `Dockerfile`, `compose.yaml` — containerization and compose service (exposes port 8000).

2) Runtime / workflows (concrete commands)
- Train model locally: `python train_model.py` (creates `app/iris_model.pkl`).
- Run locally (development): `uvicorn main:app --reload --host 0.0.0.0 --port 8000`.
- Run production container: `docker build -t iris-predictor .` then `docker run -p 8000:8000 iris-predictor`.
- Or use compose: `docker compose up --build` (service is `server`, mapped to host 8000).

3) Important implementation notes and gotchas
- Model loading: `main.py` does `joblib.load(model_path)` at import time. Agents must remember:
  - After re-training, restart the server to pick up the new model (or change `MODEL_PATH` before starting).
  - `MODEL_PATH` env var can override `app/iris_model.pkl`.
- Input validation: `IrisInput` is a Pydantic model (fields: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`). Keep parameter names and types consistent.
- Prediction output: returns `{"prediction": <int>, "species": <string>}` where mapping is defined inline in `main.py`.

4) Patterns & conventions used in this repo
- Lightweight single-file FastAPI app (no router splitting). New endpoints should follow the same pattern and keep business logic small and testable.
- Model artifact location: `app/iris_model.pkl` — follow this path for tooling unless `MODEL_PATH` is explicitly used.
- Training script should be idempotent and safe to run locally (it creates `app/` if missing).

5) Integration points and dependencies
- External packages used (see `requirements.txt`): `fastapi`, `uvicorn`, `scikit-learn`, `joblib`, `numpy`, `pydantic`.
- Docker integration: `Dockerfile` installs `requirements.txt` and runs `uvicorn main:app`. Image expects the model file to be present in the image (created by running `train_model.py` before build, or by copying in a model into `app/` in the build context).

6) Helpful examples for code changes
- If adding a new endpoint that uses the model, follow `main.py`'s simple pattern:
  - Accept a Pydantic input model
  - Convert to `numpy.array` with shape `(1, 4)`
  - Call `model.predict(...)` and map to human-readable output
- To add hot-reload of models without restart, implement a simple reload helper (example: check `MODEL_PATH` and `joblib.load()` inside the endpoint or use an admin POST endpoint to reload). Note: current code loads at import time.

7) Where to look first when changing behavior
- `main.py` — inference behavior, request/response shapes, env var usage.
- `train_model.py` — how the model was trained and persisted; mirrors preprocessing expectations (no scaling used here).
- `Dockerfile` and `compose.yaml` — container start command and ports.

8) Testing & debugging tips
- No test suite present — run the training script and start the server locally to sanity-check changes.
- Use `curl` or `http` to test the `/predict` endpoint:
  - Example: `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'`

9) When merging or refactoring
- Preserve the simple structure: keep model handling and API surface minimal and explicit. If extracting modules, update `Dockerfile` and import paths accordingly.

If anything here is unclear or you'd like more detail (examples for a reload endpoint, unit-test scaffolding, or CI integration), tell me which area to expand.
