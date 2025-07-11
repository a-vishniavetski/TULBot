### How to launch (Docker, new, good):

This runs everything in dev mode, frontend with hot-reload on localhost:5173, backend on localhost:8000, NO hot-reload.
All backend changes require to stop docker and execute step 3.

1) `cd` into the project directory
2) Do have the `.env` file in `backend/`, with the necessary variables like in `backend/.env.example`.
3) `docker compose -f docker-compose.dev.yaml up --build`


### How to test (TO-DO)

Example test queries:
```
test_queries = [
    {
    "subject_name": "Mechanika płynów",
    "test_query": "Na jakim przedmiocie powiedzą mi o elementarnych przepływach laminarnych a turbulentnych?"
    },
    {
    "subject_name": "Physics (Fizyka)",
    "test_query": "Na jakim przedmiocie powiedzą mi o rodzaje i źródłach sił w przyrodzie, zasadach dynamiki Newtona, a formułowaniu i rozwiązywaniu równań ruchu?"
    },
```

Example qdrant results:
```
TEST QUERY (Top k = 3)
Query: Na jakim przedmiocie powiedzą mi o elementarnych przepływach laminarnych a turbulentnych?
Target: Mechanika płynów
Result: Fizyka Score: 0.7656182
Result: Algebra Score: 0.7646966
Result: Matematyka dyskretna Score: 0.76191163

TEST QUERY (Top k = 3)
Query: Na jakim przedmiocie powiedzą mi o rodzaje i źródłach sił w przyrodzie, zasadach dynamiki Newtona, a formułowaniu i rozwiązywaniu równań ruchu?
Target: Physics (Fizyka)
Result: Fizyka Score: 0.8235302
Result: Matematyka dyskretna Score: 0.801771
Result: Modern Physics (Fizyka współczesna) Score: 0.7988901
```



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
