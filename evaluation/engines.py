from typing import Optional
from tqdm import tqdm

class BaseEngine:
    def generate(self, prompts: list[str]) -> list[str]:
        raise NotImplementedError

class VLLMEngine(BaseEngine):
    def __init__(self, ckpt_path: str, **kwargs):
        from vllm import LLM

        self.llm = LLM(model=ckpt_path, **kwargs)

    def generate(self, prompts: list[str], temperature: float = 0.0, max_tokens: Optional[int] = None, stop: Optional[str | list[str]] = None) -> list[str]:
        from vllm import SamplingParams

        outputs = self.llm.generate(prompts, SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop
        ))
        return [out.outputs[0].text for out in outputs]


class MLXQwenHRMEngine(BaseEngine):
    def __init__(
        self,
        ckpt_path: str,
        mode: str = "hrm",
        chat_template: bool = False,
        h_cycles: int = 1,
        l_cycles: int = 1,
        alpha_l: float = 0.03,
        alpha_h: float = 0.05,
        beta_l: float = 0.01,
        beta_h: float = 0.01,
        update_mix_l: float = 1.0,
        update_mix_h: float = 1.0,
        refined_delta_scale: float = 1.0,
        logit_blend: float = 0.2,
        logit_fusion: str = "blend",
        split_index: Optional[int] = None,
        revision: Optional[str] = None,
        **_: object,
    ):
        import mlx.core as mx
        from transformers import AutoTokenizer

        from mlx_hrm_text.qwen_hrm import QwenHrmForCausalLM

        self.mx = mx
        self.chat_template = chat_template
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=True, trust_remote_code=True)
        self.model = QwenHrmForCausalLM.from_pretrained(ckpt_path, revision=revision)

        if split_index is not None:
            self.model.model.split_index = split_index
        if mode == "base":
            self.model.model.H_cycles = 0
            self.model.model.L_cycles = 1
            self.model.logit_blend = 0.0
            self.model.logit_fusion = "blend"
        else:
            self.model.model.H_cycles = h_cycles
            self.model.model.L_cycles = l_cycles
            self.model.model.alpha_l = alpha_l
            self.model.model.alpha_h = alpha_h
            self.model.model.beta_l = beta_l
            self.model.model.beta_h = beta_h
            self.model.model.update_mix_l = update_mix_l
            self.model.model.update_mix_h = update_mix_h
            self.model.model.refined_delta_scale = refined_delta_scale
            self.model.logit_blend = logit_blend
            self.model.logit_fusion = logit_fusion
        mx.eval(self.model.parameters())

    def _format_prompt(self, prompt: str) -> str:
        if not self.chat_template:
            return prompt
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def generate(
        self,
        prompts: list[str],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[str | list[str]] = None,
        **_: object,
    ) -> list[str]:
        from mlx_hrm_text.generate_qwen_hrm import generate

        outputs = []
        eos_token_id = self.tokenizer.eos_token_id
        if max_tokens is None:
            max_tokens = 128
        for prompt in tqdm(prompts, desc="generation"):
            text = generate(
                self.model,
                self.tokenizer,
                self._format_prompt(prompt),
                max_tokens=max_tokens,
                temperature=temperature,
                eos_token_id=eos_token_id,
            )
            if stop:
                stops = [stop] if isinstance(stop, str) else stop
                for marker in stops:
                    text = text.split(marker)[0]
            outputs.append(text)
        return outputs

class SimpleEngine(BaseEngine):
    def __init__(self, ckpt_path: str, ckpt_epoch: Optional[int] = None, ckpt_use_ema: bool = True):
        from simple_inference_engine import inference_load_checkpoint

        self.ckpt = inference_load_checkpoint(ckpt_path, ckpt_epoch, ckpt_use_ema)

    def generate(self, prompts: list[str], batch_size: int = 100, max_context: int = 1024, max_tokens: Optional[int] = None, temperature: float = 0.0, condition: str = "direct") -> list[str]:
        from simple_inference_engine import inference_generate

        if max_tokens is None:
            max_tokens = max_context

        # Launch generation
        engine_prompts = [(i, (condition, p.strip())) for i, p in enumerate(prompts)]
        outputs = [""] * len(engine_prompts)

        pbar = tqdm(total=len(outputs), desc="generation")
        for gen_id, generated_text in inference_generate(
            self.ckpt, iter(engine_prompts), max_context, max_tokens, batch_size, temperature
        ):
            outputs[gen_id] = generated_text
            pbar.update()
        pbar.close()

        return outputs
