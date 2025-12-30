from pydantic import BaseModel
from typing import List, Dict, Optional

class Message(BaseModel):
    role: str
    content: str

class TrajectoryMetadata(BaseModel):
    seed_id: str
    assistant_model: str
    user_model: str
    timestamp: str
    total_turns: int
    max_turns: int

class Trajectory(BaseModel):
    metadata: TrajectoryMetadata
    conversation: List[Message]
