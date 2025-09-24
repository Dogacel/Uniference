import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Store worker states and events
workers = {}

class RegisterRequest(BaseModel):
    worker_id: str

class TaskRequest(BaseModel):
    worker_id: str
    input_data: str

@app.post("/register")
async def register_worker(req: RegisterRequest):
    if req.worker_id in workers:
        raise HTTPException(status_code=400, detail="Worker already registered")
    # Each worker gets its own event and result placeholder
    workers[req.worker_id] = {
        "event": asyncio.Event(),
        "result": None,
        "input_data": None
    }
    return {"status": "registered"}

@app.post("/wait_for_task")
async def wait_for_task(req: RegisterRequest):
    if req.worker_id not in workers:
        raise HTTPException(status_code=404, detail="Worker not registered")
    worker = workers[req.worker_id]
    await worker["event"].wait()
    worker["event"].clear()
    return {"input_data": worker["input_data"]}

@app.post("/send_task")
async def send_task(req: TaskRequest):
    if req.worker_id not in workers:
        raise HTTPException(status_code=404, detail="Worker not registered")
    worker = workers[req.worker_id]
    worker["input_data"] = req.input_data
    worker["event"].set()
    return {"status": "task sent"}

# Example: worker calls /wait_for_task and blocks until /send_task is called for it.