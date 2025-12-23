# app/llm/client.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # small and open-source


class LocalLLMClient:
    def __init__(self, model_name: str = BASE_MODEL_NAME):
        print(f"[LocalLLMClient] Loading LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )
        # Give enough room for full prompt (evidence + question + "Answer:")
        self.max_input_tokens = 1536   # back up from 1024
        # Keep answers short for speed
        self.max_new_tokens = 96

    def __call__(self, prompt: str) -> str:
        """
        Direct call to model.generate with manual truncation and max_new_tokens.
        Avoids pipeline length validation issues.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        output_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )
        return output_text
