from __future__ import annotations

import inspect
import os
import re
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model_resolve import MISSING_WEIGHTS_MESSAGE, resolve_model_source
from .story_memory import CONTINUATION_PATTERN, retrieve_memories, save_turn

SPECIAL_TOKEN_RE = re.compile(
    r"</?s>|<\|endoftext\|>|<BOS>|<EOS>|<PAD>|"
    r"<superhero>|<action>|<drama>|<thriller>|<horror>|<sci_fi>",
    re.IGNORECASE,
)
GENRE_TAG_RE = re.compile(
    r"<(?:superhero|action|drama|thriller|horror|sci_fi)>", re.IGNORECASE
)


class TextGenerationRequest(BaseModel):
    prompt: str
    conversation_id: str = ""
    story_id: str = ""
    genre: Literal["action", "drama"] = "drama"
    max_length: int = 180
    top_k: int = 40
    top_p: float = 0.88
    temperature: float = 0.65
    repetition_penalty: float = 1.12


app = FastAPI(title="Narrative Hub")


def prompt_format_for(model, tokenizer) -> str:
    configured = getattr(model.config, "story_prompt_format", None)
    if configured:
        return configured
    if getattr(tokenizer, "chat_template", None):
        return "chat_template"
    return "legacy"


def sync_loaded_special_tokens(model, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.bos_token == "<BOS>" or tokenizer.eos_token == "<EOS>":
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        if not getattr(model.config, "story_prompt_format", None):
            model.config.story_prompt_format = "idea_story_v1"
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            generation_config.bos_token_id = tokenizer.bos_token_id
            generation_config.eos_token_id = tokenizer.eos_token_id
            generation_config.pad_token_id = tokenizer.pad_token_id


def eos_token_ids(tokenizer, model):
    token_id = tokenizer.eos_token_id
    if token_id is None:
        token_id = getattr(model.config, "eos_token_id", None)
    return token_id


def clean_generated_text(text: str) -> str:
    cleaned = SPECIAL_TOKEN_RE.sub(" ", text)
    cleaned = GENRE_TAG_RE.sub(" ", cleaned)
    cleaned = re.split(r"\n\s*(?:Idea|User|System)\s*:", cleaned, maxsplit=1)[0]
    cleaned = re.sub(r"^(?:Story|story)\s*:\s*", "", cleaned.strip())
    cleaned = cleaned.replace("\n", " ")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -\n\t")
    return cleaned


def build_idea_story_prompt(request: TextGenerationRequest, memories):
    premise = memories[0]["prompt"] if memories else request.prompt.strip()
    continuing = bool(CONTINUATION_PATTERN.search(request.prompt))
    story_parts = [memory["response"][-900:] for memory in memories]
    if memories and request.prompt.strip() and not continuing:
        if request.prompt.strip().lower() not in premise.lower():
            story_parts.append(request.prompt.strip())
    story_history = " ".join(part for part in story_parts if part).strip()
    prompt = f"<BOS> <{request.genre}> Idea: {premise}\nStory:"
    if story_history:
        prompt = f"{prompt} {story_history}"
    return prompt


def load_generator():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name_or_path = resolve_model_source()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    sync_loaded_special_tokens(model, tokenizer)
    model.eval()
    return tokenizer, model, model_name_or_path


try:
    model_source = resolve_model_source()
    load_error = None
except FileNotFoundError as exc:
    model_source = ""
    load_error = str(exc)

tokenizer = None
model = None
PROMPT_FORMAT = "idea_story_v1"


def ensure_generator():
    global tokenizer, model, model_source, PROMPT_FORMAT, load_error
    if model is not None and tokenizer is not None:
        return
    if load_error:
        raise HTTPException(status_code=503, detail=load_error)
    tokenizer, model, model_source = load_generator()
    PROMPT_FORMAT = prompt_format_for(model, tokenizer)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok" if load_error is None else "missing_weights",
        "model": model_source or None,
        "prompt_format": PROMPT_FORMAT,
        "eos_token": getattr(tokenizer, "eos_token", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "warning": load_error,
    }


@app.post("/generate")
@app.post("/generate/")
async def generate_text(request: TextGenerationRequest):
    try:
        ensure_generator()
        assert model is not None and tokenizer is not None
        import torch

        if not request.prompt or not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
        memories = retrieve_memories(
            request.conversation_id,
            request.prompt,
            story_id=request.story_id,
        )
        story_id = memories[-1]["story_id"] if memories else uuid4().hex
        max_new_tokens = max(48, min(256, request.max_length))
        context_window = min(
            int(getattr(model.config, "n_positions", tokenizer.model_max_length) or 1024),
            int(tokenizer.model_max_length or 1024),
        )
        max_input_tokens = max(32, context_window - max_new_tokens)

        if PROMPT_FORMAT == "idea_story_v1":
            selected_memories = list(memories)
            while True:
                prompt = build_idea_story_prompt(request, selected_memories)
                prompt_length = len(
                    tokenizer(prompt, add_special_tokens=False)["input_ids"]
                )
                if prompt_length <= max_input_tokens or not selected_memories:
                    break
                selected_memories.pop(0)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            )
        elif getattr(tokenizer, "chat_template", None):
            system_message = (
                "You are a creative story generator. Write an original, coherent story "
                "in the requested genre. Follow the user's premise and keep its names, "
                "events, and relationships consistent. Do not introduce unrelated "
                "franchises or plot summaries. If earlier story context is supplied, "
                "continue it rather than restarting."
            )
            if memories:
                story_history = "\n\n".join(
                    f"User direction: {memory['prompt']}\n"
                    f"Story so far: {memory['response'][-1200:]}"
                    for memory in memories
                )
                user_message = (
                    f"Genre: {request.genre}\nEarlier context:\n{story_history}\n\n"
                    f"Continue from this latest direction: {request.prompt}"
                )
            else:
                user_message = (
                    f"Genre: {request.genre}\nStory premise: {request.prompt}\n"
                    "Write the opening of this story."
                )
            chat_kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
                "return_dict": True,
            }
            if "enable_thinking" in inspect.signature(
                tokenizer.apply_chat_template
            ).parameters:
                chat_kwargs["enable_thinking"] = False
            inputs = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                **chat_kwargs,
            )
        else:
            if memories:
                premise = memories[0]["prompt"]
                story_history = " ".join(
                    memory["response"][-900:] for memory in memories
                )
                prompt = (
                    f"<BOS> <{request.genre}> {premise} {story_history} {request.prompt}"
                )
            else:
                prompt = f"<BOS> <{request.genre}> {request.prompt}"
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            )

        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        with torch.inference_mode():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                min_new_tokens=24,
                do_sample=True,
                top_k=request.top_k,
                top_p=request.top_p,
                temperature=max(0.3, request.temperature),
                repetition_penalty=max(1.05, request.repetition_penalty),
                no_repeat_ngram_size=3,
                num_return_sequences=1,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_ids(tokenizer, model),
            )
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        generated_text = clean_generated_text(generated_text)
        if not generated_text:
            generated_text = (
                "The story stalled. Try a more specific premise, or regenerate."
            )
        save_turn(request.conversation_id, story_id, request.prompt, generated_text)
        return {"generated_text": generated_text, "story_id": story_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
