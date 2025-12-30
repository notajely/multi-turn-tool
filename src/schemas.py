from pydantic import BaseModel
from typing import List, Dict, Optional

class Message(BaseModel):
    role: str
    content: str

class TrajectoryMetadata(BaseModel):
    seed_id: str
    user_model: str
    assistant_model: str
    timestamp: str
    total_turns: int
    max_turns: int
    user_strategies: List[str] = []
    user_profile: Optional[str] = None

class Trajectory(BaseModel):
    metadata: TrajectoryMetadata
    conversation: List[Message]
