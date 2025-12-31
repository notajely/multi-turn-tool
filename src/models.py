import os
import time
import logging
import random
from openai import OpenAI

class LLMClient:
    PROVIDERS = {
        "dashscope": "DASHSCOPE",
        "whale": "WHALE",
        "volcano": "VOLCANO",
        "idealab": "IDEALAB",
        "openrouter": "OPENROUTER"
    }

    def __init__(self, model_name: str, channel: str = None, api_key: str = None, base_url: str = None):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Determine provider from channel or try to infer (simplified)
        self.provider = channel.lower() if channel else self._infer_provider(model_name)
        
        # Load API key and base URL from env if not provided
        prefix = self.PROVIDERS.get(self.provider, "OPENAI")
        self.api_key = api_key or os.getenv(f"{prefix}_API_KEY")
        self.base_url = base_url or os.getenv(f"{prefix}_BASE_URL")
        
        if not self.api_key:
            raise ValueError(f"API key for provider '{self.provider}' (prefix {prefix}) not found in environment.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _infer_provider(self, model_name: str) -> str:
        # Simple heuristic if channel is not provided
        model_name = model_name.lower()
        if "qwen" in model_name or "deepseek" in model_name:
            return "dashscope"
        if "doubao" in model_name:
            return "volcano"
        if "claude" in model_name or "gpt" in model_name or "gemini" in model_name:
            return "idealab"
        return "openai"

    def chat_completion(self, messages: list, system_prompt: str = None, max_retries: int = 5):
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=full_messages,
                    temperature=0.8,
                    presence_penalty=0.6,
                    frequency_penalty=0.6,
                    timeout=30
                )
                
                # Handle non-standard responses
                if isinstance(response, str):
                    return response
                if isinstance(response, dict):
                    return response.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                return response.choices[0].message.content
            except Exception as e:
                # Check for rate limit specifically
                status_str = str(e)
                is_rate_limit = "429" in status_str or "limit" in status_str.lower()
                
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.random()
                if is_rate_limit:
                    wait_time += 5 # Wait longer for rate limits
                
                self.logger.error(f"Error calling API ({self.model_name}) (Attempt {attempt+1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise e
