### How to launch (Docker, new, good):

This runs everything in dev mode, frontend with hot-reload on localhost:5173, backend on localhost:8000, NO hot-reload.
All backend changes require to stop docker and execute step 3.

1) `cd` into the project directory
2) Do have the `.env` file in `backend/`, with the necessary variables like in `backend/.env.example`.
3) `docker compose -f docker-compose.dev.yaml up --build`

### How to launch (Local, old, bad):
Backend:
1) Install `backend/requirements.txt`
2) Install pytorch
3) Create `backend/.env` with hf_token (like `backend/.env.example`), groq token if necessary, `local_model` variable (true/false)
4) `cd backend`
5) `python main.py`

Frontend:
1) `cd frontend`
2) `npm install`
3) `npm run dev`
