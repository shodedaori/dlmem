import torch
from transformers import AutoTokenizer, AutoModel

from src.models.base_model import BaseModel


class DreamModel(BaseModel):
    """
    Wrapper for Dream-v0-Instruct-7B (diffusion language model).

    Reference: https://github.com/DreamLM/Dream
    Model: Dream-org/Dream-v0-Instruct-7B
    """

    name = "dream"

    def __init__(
        self,
        model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B",
        max_new_tokens: int = 512,
        steps: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        alg: str = "entropy",
        device: str | None = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.steps = steps
        self.temperature = temperature
        self.top_p = top_p
        self.alg = alg

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.dtype = torch.bfloat16 if self.device != "cpu" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        self.model.eval().to(self.device)

    def generate(self, prompt: str, **kwargs) -> tuple[str, dict]:
        # prompt is a plain string; wrap as a user message for the chat template
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.device)

        with torch.no_grad():
            output = self.model.diffusion_generate(
                input_ids,
                max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
                steps=kwargs.get("steps", self.steps),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                alg=kwargs.get("alg", self.alg),
            )

        # Dream implementations may return either a generation object with
        # `.sequences` or the token tensor directly, depending on the version.
        if hasattr(output, "sequences"):
            sequences = output.sequences
        else:
            sequences = output

        if not isinstance(sequences, torch.Tensor):
            raise TypeError(
                f"Unexpected diffusion_generate return type: {type(output)!r}"
            )

        generated_ids = sequences[0][input_ids.shape[1]:]
        prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        meta = {
            "prompt_tokens": input_ids.shape[1],
            "generation_tokens": generated_ids.shape[0],
        }
        return prediction.strip(), meta
