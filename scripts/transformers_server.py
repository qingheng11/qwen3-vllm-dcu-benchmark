import json
import os
import queue
import threading
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.generation.streamers import BaseStreamer


MODEL_PATH = os.environ.get("MODEL_PATH")
if not MODEL_PATH:
    raise RuntimeError(
        "请先设置 MODEL_PATH=/path/to/Qwen3-1.7B"
    )

SERVED_MODEL_NAME = os.environ.get(
    "SERVED_MODEL_NAME", "qwen3-1.7b"
)
DEVICE = os.environ.get("DEVICE", "cuda:0")

app = FastAPI()
generation_lock = threading.Lock()

print("正在加载Tokenizer……", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
)

print("正在加载Transformers模型……", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
    dtype=torch.bfloat16,
).to(DEVICE)

model.eval()
print("Transformers模型加载完成", flush=True)


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = True
    ignore_eos: bool = False


class IncrementalStreamer(BaseStreamer):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.queue = queue.Queue()
        self.skip_prompt = True
        self.token_ids = []
        self.decoded_text = ""

    def put(self, value):
        if self.skip_prompt:
            self.skip_prompt = False
            return

        token_ids = value.detach().cpu().reshape(-1).tolist()
        self.token_ids.extend(token_ids)

        text = self.tokenizer.decode(
            self.token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        delta = text[len(self.decoded_text):]
        self.decoded_text = text

        if delta:
            self.queue.put(("text", delta))

    def end(self):
        self.queue.put(("end", None))


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": SERVED_MODEL_NAME,
            "object": "model",
            "owned_by": "transformers",
            "root": MODEL_PATH,
        }],
    }


@app.post("/v1/completions")
def completions(request: CompletionRequest):
    if request.model != SERVED_MODEL_NAME:
        return JSONResponse(
            status_code=404,
            content={"error": "model not found"},
        )

    if isinstance(request.prompt, list):
        if request.prompt and isinstance(request.prompt[0], int):
            prompt = tokenizer.decode(request.prompt)
        else:
            prompt = str(request.prompt[0])
    else:
        prompt = request.prompt

    request_id = f"cmpl-{uuid.uuid4().hex}"

    def generate_stream():
        streamer = IncrementalStreamer(tokenizer)

        def worker():
            try:
                with generation_lock:
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        add_special_tokens=True,
                    ).to(DEVICE)

                    kwargs = {
                        **inputs,
                        "streamer": streamer,
                        "max_new_tokens": request.max_tokens,
                        "pad_token_id": tokenizer.eos_token_id,
                        "do_sample": request.temperature > 0,
                    }

                    if request.ignore_eos:
                        kwargs["min_new_tokens"] = request.max_tokens

                    if request.temperature > 0:
                        kwargs["temperature"] = request.temperature

                    with torch.inference_mode():
                        model.generate(**kwargs)

            except Exception as exc:
                streamer.queue.put(("error", repr(exc)))
                streamer.queue.put(("end", None))

        threading.Thread(
            target=worker,
            daemon=True,
        ).start()

        while True:
            kind, value = streamer.queue.get()

            if kind == "text":
                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": SERVED_MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "text": value,
                        "logprobs": None,
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            elif kind == "error":
                error = {
                    "error": {
                        "message": value,
                        "type": "server_error",
                    }
                }
                yield f"data: {json.dumps(error)}\n\n"

            elif kind == "end":
                final_chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": SERVED_MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "text": "",
                        "logprobs": None,
                        "finish_reason": "length",
                    }],
                }

                yield (
                    f"data: {json.dumps(final_chunk)}\n\n"
                    "data: [DONE]\n\n"
                )
                break

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )
