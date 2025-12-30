import os
import sys
import json
import logging
import re
from typing import List, Dict, Any, Tuple
from src.schemas import Trajectory, Message
from src.models import LLMClient

# Add emoclassifiers to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMO_PATH = os.path.join(PROJECT_ROOT, "emoclassifiers")
if EMO_PATH not in sys.path:
    sys.path.append(EMO_PATH)

import emoclassifiers.chunking as chunking
import emoclassifiers.prompt_templates as prompt_templates

class BehaviorAnalyzer:
    def __init__(self, judge_client: LLMClient, definition_path: str = None):
        self.judge_client = judge_client
        self.logger = logging.getLogger(__name__)
        
        if not definition_path:
            definition_path = os.path.join(EMO_PATH, "assets", "definitions", "emoclassifiers_v2_definition.json")
        
        with open(definition_path, "r") as f:
            self.definitions = json.load(f)

    def _format_criteria(self, criteria: List[str]) -> str:
        return "\n".join([f"- {line}" for line in criteria])

    def get_v2_prompt(self, classifier_name: str, chunk_string: str) -> str:
        defn = self.definitions.get(classifier_name)
        if not defn:
            raise ValueError(f"Classifier {classifier_name} not found in definitions.")
        
        return prompt_templates.EMO_CLASSIFIER_V2_PROMPT_TEMPLATE.format(
            classifier_name=defn["full_name"],
            criteria=self._format_criteria(defn["criteria"]),
            snippet_string=chunk_string,
            prompt=defn["prompt"]
        )

    def parse_judgement(self, response: str) -> Tuple[bool, int]:
        """
        Parses response like "true, 5" or "yes, 4".
        Returns (is_detected, confidence).
        """
        response = response.lower().strip()
        # Handle various response formats (yes, true, 1=true, etc)
        is_detected = any(word in response for word in ["true", "yes", "yes=true"])
        
        # Extract confidence (1-5)
        m = re.search(r"(\d)", response)
        confidence = int(m.group(1)) if m else 0
        return is_detected, confidence

    def analyze_chunk(self, classifier_name: str, chunk: chunking.Chunk) -> Dict[str, Any]:
        prompt = self.get_v2_prompt(classifier_name, chunk.to_string())
        response = self.judge_client.chat_completion([{"role": "user", "content": prompt}])
        is_detected, confidence = self.parse_judgement(response)
        
        return {
            "is_detected": is_detected,
            "confidence": confidence,
            "raw_response": response
        }

    def analyze_trajectory(self, trajectory: Trajectory, classifiers: List[str]) -> List[Dict[str, Any]]:
        """
        Analyzes a trajectory sequentially for each Turn and Classifier.
        Nested parallelization is avoided to prevent API rate limiting.
        """
        results = []
        convo_dicts = [m.model_dump() for m in trajectory.conversation]
        
        # Turn 2: Index 3 (User/Assistant Exchange)
        # Turn 3: Index 5
        target_indices = []
        if len(convo_dicts) > 3: target_indices.append(3)
        if len(convo_dicts) > 5: target_indices.append(5)
        
        for idx in target_indices:
            # Create chunk with context
            chunk = chunking.Chunk.from_simple_convo(convo_dicts, idx=idx, n_context=3)
            turn_index = (idx // 2) + 1
            
            for cls_name in classifiers:
                try:
                    res = self.analyze_chunk(cls_name, chunk)
                    results.append({
                        "seed_id": trajectory.metadata.seed_id,
                        "turn_index": turn_index,
                        "classifier_name": cls_name,
                        "is_detected": res["is_detected"],
                        "confidence": res["confidence"],
                        "mut_response_snippet": convo_dicts[idx]["content"][:100] + "..."
                    })
                except Exception as e:
                    self.logger.error(f"Error analyzing {trajectory.metadata.seed_id} Turn {turn_index} Cls {cls_name}: {e}")
                    results.append({
                        "seed_id": trajectory.metadata.seed_id,
                        "turn_index": turn_index,
                        "classifier_name": cls_name,
                        "is_detected": False,
                        "confidence": 0,
                        "mut_response_snippet": f"ERROR: {str(e)[:50]}"
                    })
        
        return results
