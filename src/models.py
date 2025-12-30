import os
import time
import logging
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

    def chat_completion(self, messages: list, system_prompt: str = None, max_retries: int = 3):
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        for attempt in range(max_retries):
            try:
                # Handle special model names or requirements here if needed
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=full_messages,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                self.logger.error(f"Error calling API ({self.model_name}) (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise e
