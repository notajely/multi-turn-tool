from pydantic import BaseModel
from typing import List, Dict, Optional

class Message(BaseModel):
    role: str
    content: str

class TrajectoryMetadata(BaseModel):
    seed_id: str
    mut_name: str
    prober_name: str
    timestamp: str
    total_turns: int
    max_turns: int

class Trajectory(BaseModel):
    metadata: TrajectoryMetadata
    conversation: List[Message]
