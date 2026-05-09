# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone policy proxy server used inside Harbor/OpenSandbox sandboxes."""

import argparse
import copy
import importlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import re
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


_TOKEN_ID_TEXT_RE = re.compile(r"^token_id:(?P<token_id>\d+)$")

_OPENAI_PROMPT_OVERFLOW_RE = re.compile(
    r"maximum context length is\s+(?P<max>\d+)\s+tokens\..*?"
    r"request has\s+(?P<input>\d+)\s+input tokens",
    re.IGNORECASE | re.DOTALL,
)

_VLLM_PROMPT_OVERFLOW_RE = re.compile(
    r"passed\s+(?P<input>\d+)\s+input tokens.*?"
    r"context length is only\s+(?P<max>\d+)\s+tokens",
    re.IGNORECASE | re.DOTALL,
)


def _join_url(base_url, path):
    if base_url.rstrip("/").endswith("/v1") and path.startswith("/v1/"):
        path = path[3:]
    return base_url.rstrip("/") + path


def _observability_events_path():
    return os.environ.get("NEMO_RL_SANDBOX_OBSERVABILITY_EVENTS_PATH")


def _write_observability_event(name, attributes, *, timestamp_unix_s=None):
    path = _observability_events_path()
    if not path:
        return
    event = {
        "schema_version": 1,
        "timestamp_unix_s": time.time()
        if timestamp_unix_s is None
        else timestamp_unix_s,
        "monotonic_s": None,
        "event_type": "span_end",
        "name": name,
        "attributes": attributes,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n")


def _tokenize_urls(base_url):
    urls = [_join_url(base_url, "/tokenize")]
    stripped = base_url.rstrip("/")
    if stripped.endswith("/v1"):
        urls.append(stripped[:-3] + "/tokenize")
    return urls


def _completion_text(response):
    choices = response.get("choices") or []
    if not choices:
        return ""
    return _choice_completion_text(choices[0])


def _choice_completion_text(choice):
    message = choice.get("message") or {}
    if "content" in message:
        return message.get("content") or ""
    return choice.get("text") or ""


def _finish_reason(response):
    choices = response.get("choices") or []
    if not choices:
        return "stop"
    return choices[0].get("finish_reason") or "stop"


def _dict_or_attr(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_logprobs(choice):
    message = _dict_or_attr(choice, "message") or {}
    generation_log_probs = _dict_or_attr(message, "generation_log_probs")
    if isinstance(generation_log_probs, list):
        return generation_log_probs

    logprobs = _dict_or_attr(choice, "logprobs")
    if not logprobs:
        return None
    content = _dict_or_attr(logprobs, "content")
    if isinstance(content, list):
        values = []
        for token_logprob in content:
            logprob = _dict_or_attr(token_logprob, "logprob")
            if logprob is not None:
                values.append(logprob)
        return values
    token_logprobs = _dict_or_attr(logprobs, "token_logprobs")
    if isinstance(token_logprobs, list):
        return token_logprobs
    return None


def _extract_completion_token_ids(choice):
    provider_fields = _dict_or_attr(choice, "provider_specific_fields") or {}
    message = _dict_or_attr(choice, "message") or {}
    for source, key in (
        (message, "generation_token_ids"),
        (message, "completion_token_ids"),
        (message, "token_ids"),
        (choice, "generation_token_ids"),
        (choice, "completion_token_ids"),
        (choice, "token_ids"),
        (provider_fields, "generation_token_ids"),
        (provider_fields, "completion_token_ids"),
        (provider_fields, "token_ids"),
    ):
        token_ids = _dict_or_attr(source, key)
        if isinstance(token_ids, list):
            return token_ids
    return None


def _extract_prompt_token_ids(response, choice):
    message = _dict_or_attr(choice, "message") or {}
    for source, key in (
        (message, "prompt_token_ids"),
        (message, "input_token_ids"),
        (response, "prompt_token_ids"),
        (response, "input_token_ids"),
    ):
        token_ids = _dict_or_attr(source, key)
        if isinstance(token_ids, list):
            return token_ids
    return None


def _chat_response_training_fields(response):
    choices = response.get("choices") or []
    if not choices:
        return {}

    choice = choices[0]
    prompt_token_ids = _extract_prompt_token_ids(response, choice)
    completion_token_ids = _extract_completion_token_ids(choice)
    logprobs = _extract_logprobs(choice)

    fields = {}
    if prompt_token_ids is not None:
        fields["prompt_token_ids"] = prompt_token_ids
    if completion_token_ids is not None:
        fields["generation_token_ids"] = completion_token_ids
    if logprobs is not None:
        fields["generation_log_probs"] = logprobs
    return fields


def _extract_provider_extra(choice):
    provider_fields = _dict_or_attr(choice, "provider_specific_fields") or {}
    if not hasattr(provider_fields, "items"):
        return None
    extra = {key: value for key, value in provider_fields.items() if key != "token_ids"}
    return extra or None


def _responses_response_body(response_body):
    return response_body.get("object") == "response" or (
        "output" in response_body and "choices" not in response_body
    )


def _responses_completion_text(response_body):
    if not _responses_response_body(response_body):
        return _completion_text(response_body)

    output_text = response_body.get("output_text")
    if isinstance(output_text, str):
        return output_text

    text_parts = []
    for item in response_body.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text" and isinstance(
                content.get("text"), str
            ):
                text_parts.append(content["text"])
    return "".join(text_parts)


def _responses_logprobs_and_token_ids(response_body):
    logprobs = []
    token_ids = []
    logprob_items = []
    has_token_ids = True
    for item in response_body.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            for logprob_item in content.get("logprobs") or []:
                if not isinstance(logprob_item, dict):
                    continue
                logprob = logprob_item.get("logprob")
                if logprob is None:
                    continue
                logprob_items.append(logprob_item)
                logprobs.append(logprob)
                token = logprob_item.get("token")
                match = (
                    _TOKEN_ID_TEXT_RE.match(token) if isinstance(token, str) else None
                )
                if match is None:
                    has_token_ids = False
                else:
                    token_ids.append(int(match.group("token_id")))

    if not logprobs:
        return None, None, []
    return logprobs, token_ids if has_token_ids else None, logprob_items


def _logprob_item_text(logprob_item):
    token = logprob_item.get("token")
    if isinstance(token, str):
        return token

    token_bytes = logprob_item.get("bytes")
    if not isinstance(token_bytes, list):
        return None
    try:
        return bytes(token_bytes).decode("utf-8")
    except (TypeError, ValueError, UnicodeDecodeError):
        return None


def _token_ids_from_logprob_items(target_base_url, model, logprob_items):
    token_ids = []
    cache = {}
    for logprob_item in logprob_items:
        token_text = _logprob_item_text(logprob_item)
        if token_text is None:
            return None
        if token_text not in cache:
            piece_token_ids = _tokenize_text(
                target_base_url,
                model,
                token_text,
                add_special_tokens=False,
            )
            if piece_token_ids is None or len(piece_token_ids) != 1:
                return None
            cache[token_text] = piece_token_ids[0]
        token_ids.append(cache[token_text])
    return token_ids


def _tokenize_payload(target_base_url, body):
    last_error = None
    for url in _tokenize_urls(target_base_url):
        request = Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            last_error = e
            if e.code == 404:
                continue
        except Exception as e:
            last_error = e
        else:
            tokens = payload.get("tokens")
            return tokens if isinstance(tokens, list) else None

    if last_error is not None:
        print(f"Policy proxy tokenize fallback failed: {last_error}", file=sys.stderr)
    return None


def _tokenize_responses_prompt(target_base_url, request_body, model):
    chat_body = _responses_to_chat_completion_body(request_body)
    body = {
        "model": model,
        "messages": chat_body.get("messages") or [],
        "add_generation_prompt": True,
    }
    if chat_body.get("tools") is not None:
        body["tools"] = chat_body["tools"]
    return _tokenize_payload(target_base_url, body)


def _record_responses_trace(trace_path, request_body, response_body, target_base_url):
    logprobs, completion_token_ids, logprob_items = _responses_logprobs_and_token_ids(
        response_body
    )
    model = response_body.get("model") or request_body.get("model")
    if completion_token_ids is None and target_base_url:
        completion_token_ids = _token_ids_from_logprob_items(
            target_base_url, model, logprob_items
        )
    if completion_token_ids is None and target_base_url:
        completion_token_ids = _tokenize_text(
            target_base_url,
            model,
            _responses_completion_text(response_body),
            add_special_tokens=False,
        )

    prompt_token_ids = response_body.get("prompt_token_ids")
    if prompt_token_ids is None and target_base_url:
        prompt_token_ids = _tokenize_responses_prompt(
            target_base_url,
            request_body,
            model,
        )

    if not prompt_token_ids or completion_token_ids is None or logprobs is None:
        print(
            "Policy proxy could not record trainable Responses trace: "
            "missing prompt tokens, completion tokens, or logprobs",
            file=sys.stderr,
        )
        return
    if len(completion_token_ids) != len(logprobs):
        print(
            "Policy proxy could not record trainable Responses trace: "
            f"completion token/logprob length mismatch "
            f"({len(completion_token_ids)} != {len(logprobs)})",
            file=sys.stderr,
        )
        return

    record = {
        "prompt_token_ids": prompt_token_ids,
        "completion_token_ids": completion_token_ids,
        "logprobs": logprobs,
        "request_model": request_body.get("model"),
        "response_model": response_body.get("model"),
        "extra": {"upstream_api": "responses"},
    }
    Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def _record_trace(trace_path, request_body, response_body, target_base_url=None):
    if _responses_response_body(response_body):
        _record_responses_trace(trace_path, request_body, response_body, target_base_url)
        return

    choices = response_body.get("choices") or []
    if not choices:
        return
    choice = choices[0]
    record = {
        "prompt_token_ids": response_body.get("prompt_token_ids"),
        "completion_token_ids": _extract_completion_token_ids(choice),
        "logprobs": _extract_logprobs(choice),
        "request_model": request_body.get("model"),
        "response_model": response_body.get("model"),
    }
    extra = _extract_provider_extra(choice)
    if extra is not None:
        record["extra"] = extra
    Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def _tokenize_text(target_base_url, model, text, *, add_special_tokens=False):
    if not text:
        return None
    return _tokenize_payload(
        target_base_url,
        {
            "model": model,
            "prompt": text,
            "add_special_tokens": add_special_tokens,
        },
    )


def _ensure_response_token_ids(target_base_url, request_body, response_body):
    choices = response_body.get("choices") or []
    for choice in choices:
        if _extract_completion_token_ids(choice) is not None:
            continue
        token_ids = _tokenize_text(
            target_base_url,
            response_body.get("model") or request_body.get("model"),
            _choice_completion_text(choice),
        )
        if token_ids is not None:
            choice["token_ids"] = token_ids


def _litellm_model_name(model, provider):
    if not provider or not model:
        return model
    prefix = provider.rstrip("/") + "/"
    if model.startswith(prefix):
        return model
    return prefix + model


def _api_key_from_headers(headers):
    authorization = headers.get("authorization") or headers.get("Authorization")
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1]
    return os.environ.get("OPENAI_API_KEY") or "sandbox-proxy"


def _response_to_dict(response):
    if isinstance(response, dict):
        response_body = dict(response)
    elif hasattr(response, "to_dict"):
        response_body = response.to_dict()
    elif hasattr(response, "model_dump"):
        response_body = response.model_dump()
    else:
        response_body = json.loads(json.dumps(response))

    prompt_token_ids = getattr(response, "prompt_token_ids", None)
    if prompt_token_ids is not None and response_body.get("prompt_token_ids") is None:
        response_body["prompt_token_ids"] = prompt_token_ids
    return response_body


def _chat_completion_path(path):
    return path.split("?", 1)[0].endswith("/chat/completions")


def _responses_path(path):
    return path.split("?", 1)[0].endswith("/responses")


def _anthropic_messages_path(path):
    return path.split("?", 1)[0].endswith("/messages")


def _litellm_available():
    try:
        importlib.import_module("litellm")
    except Exception:
        return False
    return True


def _backend_for_request(backend_mode, path):
    if not _chat_completion_path(path):
        return "stdlib"
    if backend_mode == "auto":
        return "litellm" if _litellm_available() else "stdlib"
    return backend_mode


def _forward_litellm(target_base_url, request_body, headers, litellm_provider):
    litellm = importlib.import_module("litellm")
    completion_kwargs = dict(request_body)
    completion_kwargs.pop("return_token_ids", None)
    completion_kwargs["model"] = _litellm_model_name(
        completion_kwargs.get("model"), litellm_provider
    )
    completion_kwargs["api_base"] = target_base_url
    completion_kwargs["api_key"] = _api_key_from_headers(headers)
    completion_kwargs["stream"] = False
    completion_kwargs["logprobs"] = True

    extra_body = completion_kwargs.get("extra_body")
    if isinstance(extra_body, dict):
        extra_body = dict(extra_body)
    else:
        extra_body = {}
    for key in ("top_k", "chat_template_kwargs"):
        if key in completion_kwargs:
            extra_body[key] = completion_kwargs.pop(key)
    extra_body["return_token_ids"] = True
    completion_kwargs["extra_body"] = extra_body

    response = litellm.completion(**completion_kwargs)
    response_body = _response_to_dict(response)
    return (
        200,
        {"content-type": "application/json"},
        json.dumps(response_body, separators=(",", ":")).encode("utf-8"),
    )


def _forward_stdlib(method, upstream_url, upstream_body, headers):
    try:
        request = Request(
            upstream_url,
            data=upstream_body,
            headers=headers,
            method=method,
        )
        with urlopen(request, timeout=600) as response:
            response_body = response.read()
            status = response.status
            response_headers = response.headers
    except HTTPError as e:
        response_body = e.read()
        status = e.code
        response_headers = e.headers
    return status, response_headers, response_body


def _response_error_message(response_body):
    decoded = response_body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError:
        return decoded
    if not isinstance(payload, dict):
        return decoded

    error = payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    message = payload.get("message")
    if isinstance(message, str):
        return message
    return decoded


def _chat_usage(response):
    usage = response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    return {
        "input_tokens": prompt_tokens or 0,
        "output_tokens": completion_tokens or 0,
        "total_tokens": total_tokens or (prompt_tokens or 0) + (completion_tokens or 0),
    }


def _response_error_type(status, parsed_response):
    if 200 <= status < 300:
        return None
    if isinstance(parsed_response, dict):
        error = parsed_response.get("error")
        if isinstance(error, dict):
            return error.get("type") or error.get("code") or "upstream_error"
        if isinstance(error, str):
            return error
    return "upstream_error"


def _prompt_overflow_truncate_tokens(response_body):
    message = _response_error_message(response_body)
    match = _OPENAI_PROMPT_OVERFLOW_RE.search(message)
    if match is None:
        match = _VLLM_PROMPT_OVERFLOW_RE.search(message)
    if match is None:
        return None

    max_model_len = int(match.group("max"))
    input_tokens = int(match.group("input"))
    if max_model_len <= 1 or input_tokens < max_model_len:
        return None
    return max_model_len - 1


def _prompt_overflow_retry_body(request_body, response_body):
    if "truncate_prompt_tokens" in request_body:
        return None

    truncate_prompt_tokens = _prompt_overflow_truncate_tokens(response_body)
    if truncate_prompt_tokens is None:
        return None

    retry_body = dict(request_body)
    retry_body["truncate_prompt_tokens"] = truncate_prompt_tokens
    return (
        json.dumps(retry_body).encode("utf-8"),
        truncate_prompt_tokens,
    )


def _forward_stdlib_with_overflow_retry(
    method, upstream_url, upstream_body, headers, request_body
):
    status, response_headers, response_body = _forward_stdlib(
        method, upstream_url, upstream_body, headers
    )
    if status != 400 or not isinstance(request_body, dict):
        return status, response_headers, response_body

    retry = _prompt_overflow_retry_body(request_body, response_body)
    if retry is None:
        return status, response_headers, response_body

    retry_upstream_body, truncate_prompt_tokens = retry
    print(
        "Policy proxy retrying vLLM prompt overflow with "
        f"truncate_prompt_tokens={truncate_prompt_tokens}",
        file=sys.stderr,
    )
    return _forward_stdlib(method, upstream_url, retry_upstream_body, headers)


def _content_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        return _content_text(content.get("text") or content.get("content"))
    return str(content)


def _responses_messages(input_value):
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    if not isinstance(input_value, list):
        return []

    messages = []
    for item in input_value:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call_output":
            output = item.get("output")
            if not isinstance(output, str):
                output = _content_text(item.get("content"))
            if not output:
                continue
            call_id = item.get("call_id")
            if isinstance(call_id, str) and call_id:
                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": output}
                )
            else:
                messages.append({"role": "user", "content": output})
            continue
        role = item.get("role") or "user"
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
        content = _content_text(item.get("content"))
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _responses_tools_to_chat_tools(tools):
    if not isinstance(tools, list):
        return None
    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and "function" not in tool:
            name = tool.get("name")
            if not name:
                continue
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    },
                }
            )
        elif tool.get("type") == "function":
            converted.append(tool)
    return converted or None


def _responses_tool_choice_to_chat_tool_choice(tool_choice, tool_names):
    if isinstance(tool_choice, str):
        return tool_choice
    if (
        isinstance(tool_choice, dict)
        and tool_choice.get("type") == "function"
        and "function" not in tool_choice
        and tool_choice.get("name") in tool_names
    ):
        return {
            "type": "function",
            "function": {"name": tool_choice.get("name")},
        }
    if (
        isinstance(tool_choice, dict)
        and tool_choice.get("type") == "function"
        and isinstance(tool_choice.get("function"), dict)
        and tool_choice["function"].get("name") in tool_names
    ):
        return tool_choice
    return None


def _responses_to_chat_completion_body(request_body, prior_messages=None):
    if prior_messages is not None:
        messages = copy.deepcopy(prior_messages)
    else:
        messages = []
        instructions = request_body.get("instructions")
        if isinstance(instructions, str) and instructions:
            messages.append({"role": "system", "content": instructions})
    messages.extend(_responses_messages(request_body.get("input")))

    chat_body = {
        "model": request_body.get("model"),
        "messages": messages,
    }
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
        "chat_template_kwargs",
    ):
        if key in request_body:
            chat_body[key] = request_body[key]
    if "max_output_tokens" in request_body:
        chat_body["max_tokens"] = request_body["max_output_tokens"]
    tools = _responses_tools_to_chat_tools(request_body.get("tools"))
    if tools is not None:
        chat_body["tools"] = tools
        if "tool_choice" in request_body:
            tool_names = {
                tool["function"]["name"]
                for tool in tools
                if isinstance(tool.get("function"), dict)
                and tool["function"].get("name")
            }
            tool_choice = _responses_tool_choice_to_chat_tool_choice(
                request_body["tool_choice"],
                tool_names,
            )
            if tool_choice is not None:
                chat_body["tool_choice"] = tool_choice
    return chat_body


def _apply_generation_overrides(request_body, generation_overrides, force):
    if not generation_overrides:
        return request_body
    for key, value in generation_overrides.items():
        if key == "chat_template_kwargs" and isinstance(value, dict):
            existing = request_body.get(key)
            if isinstance(existing, dict):
                merged = dict(existing)
            else:
                merged = {}
            for child_key, child_value in value.items():
                if force or child_key not in merged:
                    merged[child_key] = child_value
            if force or key not in request_body or merged:
                request_body[key] = merged
            continue
        if force or key not in request_body:
            request_body[key] = value
    return request_body


def _responses_native_upstream_body(
    request_body, generation_overrides=None, force_generation_overrides=False
):
    upstream_body = dict(request_body)
    include = upstream_body.get("include")
    if isinstance(include, list):
        include = list(include)
    else:
        include = []
    if "message.output_text.logprobs" not in include:
        include.append("message.output_text.logprobs")
    upstream_body["include"] = include
    upstream_body.setdefault("top_logprobs", 0)
    upstream_body["stream"] = False
    upstream_body.setdefault("store", False)
    _apply_generation_overrides(
        upstream_body, generation_overrides, force_generation_overrides
    )
    return upstream_body


def _anthropic_system_text(system):
    if isinstance(system, str):
        return system
    return _content_text(system)


def _anthropic_messages(request_body):
    messages = []
    system = _anthropic_system_text(request_body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})

    for message in request_body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"user", "assistant"}:
            role = "user"
        content = _content_text(message.get("content"))
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _anthropic_tools_to_chat_tools(tools):
    if not isinstance(tools, list):
        return None
    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name:
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
        )
    return converted or None


def _anthropic_to_chat_completion_body(request_body):
    chat_body = {
        "model": request_body.get("model"),
        "messages": _anthropic_messages(request_body),
    }
    if "temperature" in request_body:
        chat_body["temperature"] = request_body["temperature"]
    if "top_p" in request_body:
        chat_body["top_p"] = request_body["top_p"]
    if "max_tokens" in request_body:
        chat_body["max_tokens"] = request_body["max_tokens"]
    tools = _anthropic_tools_to_chat_tools(request_body.get("tools"))
    if tools is not None:
        chat_body["tools"] = tools
    return chat_body


def _anthropic_usage(response):
    usage = response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
    }


def _anthropic_message_object(response, message_id, text, stop_reason):
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": response.get("model") or "",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": _anthropic_usage(response),
    }


def _anthropic_sse_chunks(response):
    created = response.get("created") or int(time.time())
    message_id = response.get("id") or f"msg_proxy_{created}"
    text = _completion_text(response)
    stop_reason = "end_turn"
    output_tokens = _anthropic_usage(response)["output_tokens"]
    return [
        {
            "type": "message_start",
            "message": _anthropic_message_object(response, message_id, "", None),
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
        {
            "type": "content_block_stop",
            "index": 0,
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
        {"type": "message_stop"},
    ]


def _responses_usage(response):
    usage = response.get("usage") or {}
    prompt_tokens = usage.get("input_tokens")
    if prompt_tokens is None:
        prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("output_tokens")
    if completion_tokens is None:
        completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    return {
        "input_tokens": prompt_tokens or 0,
        "output_tokens": completion_tokens or 0,
        "total_tokens": total_tokens or (prompt_tokens or 0) + (completion_tokens or 0),
    }


def _responses_proxy_response_id(response):
    return response.get("id") or f"resp_proxy_{response.get('created') or int(time.time())}"


def _chat_response_tool_calls(response):
    choices = response.get("choices") or []
    if not choices:
        return []
    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    return tool_calls if isinstance(tool_calls, list) else []


def _chat_response_assistant_message(response):
    choices = response.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    assistant_message = {"role": "assistant", "content": content}
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        assistant_message["tool_calls"] = copy.deepcopy(tool_calls)
    if not content and "tool_calls" not in assistant_message:
        return None
    return assistant_message


def _responses_function_call_item(tool_call, index):
    function = tool_call.get("function") if isinstance(tool_call, dict) else None
    if not isinstance(function, dict):
        function = {}
    call_id = (
        tool_call.get("id") if isinstance(tool_call, dict) else None
    ) or f"call_proxy_{index}"
    return {
        "id": f"fc_proxy_{index}",
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": function.get("name") or "",
        "arguments": function.get("arguments") or "",
    }


def _responses_output_item(item_id, text, status, *, training_fields=None):
    item = {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
            }
        ],
    }
    if training_fields:
        item.update(training_fields)
    return item


def _responses_output_items(response, item_id, text, status):
    if status != "completed":
        return []
    if _responses_response_body(response):
        output = response.get("output") or []
        return output if isinstance(output, list) else []

    output = []
    if text:
        output.append(
            _responses_output_item(
                item_id,
                text,
                status,
                training_fields=_chat_response_training_fields(response),
            )
        )
    output.extend(
        _responses_function_call_item(tool_call, index)
        for index, tool_call in enumerate(_chat_response_tool_calls(response))
        if isinstance(tool_call, dict)
    )
    return output


def _responses_response_object(response, response_id, item_id, text, status):
    body = {
        "id": response_id,
        "object": "response",
        "created_at": response.get("created_at")
        or response.get("created")
        or int(time.time()),
        "status": status,
        "model": response.get("model") or "",
        "output": _responses_output_items(response, item_id, text, status),
    }
    usage = _responses_usage(response)
    if usage is not None:
        body["usage"] = usage
    return body


def _responses_sse_chunks(response):
    created = response.get("created_at") or response.get("created") or int(time.time())
    response_id = _responses_proxy_response_id(response)
    item_id = f"msg_proxy_{created}"
    text = _responses_completion_text(response)
    completed_items = _responses_output_items(response, item_id, text, "completed")
    stream_item = (
        completed_items[0]
        if completed_items
        else {
            "id": item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [],
        }
    )
    chunks = [
        {
            "type": "response.created",
            "response": _responses_response_object(
                response, response_id, item_id, "", "in_progress"
            ),
        },
        {
            "type": "response.output_item.added",
            "response_id": response_id,
            "output_index": 0,
            "item": dict(stream_item, status="in_progress"),
        },
    ]
    if text:
        chunks.append(
            {
                "type": "response.output_text.delta",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            }
        )
    chunks.extend(
        [
            {
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": 0,
                "item": stream_item,
            },
            {
                "type": "response.completed",
                "response": _responses_response_object(
                    response, response_id, item_id, text, "completed"
                ),
            },
        ]
    )
    return chunks


def _stream_chat_response(handler, response):
    created = response.get("created") or int(time.time())
    model = response.get("model") or ""
    response_id = response.get("id") or f"chatcmpl-proxy-{created}"
    text = _completion_text(response)
    finish_reason = _finish_reason(response)
    chunks = [
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        },
    ]
    _write_sse(handler, chunks)


def _stream_completion_response(handler, response):
    created = response.get("created") or int(time.time())
    model = response.get("model") or ""
    response_id = response.get("id") or f"cmpl-proxy-{created}"
    text = _completion_text(response)
    finish_reason = _finish_reason(response)
    chunks = [
        {
            "id": response_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": None}],
        },
        {
            "id": response_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "text": "", "finish_reason": finish_reason}],
        },
    ]
    _write_sse(handler, chunks)


def _stream_responses_response(handler, response):
    _write_sse(handler, _responses_sse_chunks(response))


def _stream_anthropic_response(handler, response):
    _write_anthropic_sse(handler, _anthropic_sse_chunks(response))


def _write_sse(handler, chunks):
    handler.send_response(200)
    handler.send_header("content-type", "text/event-stream")
    handler.send_header("cache-control", "no-cache")
    handler.end_headers()
    for chunk in chunks:
        chunk_data = json.dumps(chunk, separators=(",", ":"))
        handler.wfile.write(f"data: {chunk_data}\n\n".encode("utf-8"))
    handler.wfile.write(b"data: [DONE]\n\n")


def _write_anthropic_sse(handler, chunks):
    handler.send_response(200)
    handler.send_header("content-type", "text/event-stream")
    handler.send_header("cache-control", "no-cache")
    handler.end_headers()
    for chunk in chunks:
        chunk_data = json.dumps(chunk, separators=(",", ":"))
        handler.wfile.write(f"event: {chunk['type']}\n".encode("utf-8"))
        handler.wfile.write(f"data: {chunk_data}\n\n".encode("utf-8"))


def _record_model_call_observability(
    *,
    status,
    requested_path,
    upstream_path,
    backend,
    wants_stream,
    request_body,
    parsed_response,
    started_at_unix_s,
    duration_s,
):
    if not _observability_events_path():
        return
    usage = None
    if isinstance(parsed_response, dict):
        if _responses_response_body(parsed_response):
            usage = _responses_usage(parsed_response)
        else:
            usage = _chat_usage(parsed_response)
    attributes = {
        "phase": "llm",
        "status": "ok" if 200 <= status < 300 else "error",
        "http_status": status,
        "duration_s": duration_s,
        "requested_api": "responses"
        if _responses_path(requested_path)
        else "anthropic"
        if _anthropic_messages_path(requested_path)
        else "chat",
        "upstream_api": "responses" if upstream_path == "/v1/responses" else "chat",
        "backend": backend,
        "stream": wants_stream,
    }
    if isinstance(request_body, dict) and request_body.get("model") is not None:
        attributes["model"] = str(request_body["model"])
    if usage is not None:
        attributes["prompt_tokens"] = usage["input_tokens"]
        attributes["completion_tokens"] = usage["output_tokens"]
        attributes["total_tokens"] = usage["total_tokens"]
    error_type = _response_error_type(status, parsed_response)
    if error_type is not None:
        attributes["error_type"] = error_type
    _write_observability_event(
        "llm.request",
        attributes,
        timestamp_unix_s=started_at_unix_s + duration_s,
    )


class ProxyHandler(BaseHTTPRequestHandler):
    target_base_url = None
    trace_path = None
    backend_mode = "stdlib"
    litellm_provider = "openai"
    upstream_model_name = None
    responses_upstream_api = "chat"
    generation_overrides = None
    force_generation_overrides = False
    force_logprobs = True
    force_token_ids = True
    responses_chat_histories = {}
    responses_chat_history_lock = threading.RLock()

    def do_GET(self):
        self._forward(raw_body=None)

    def do_POST(self):
        content_length = int(self.headers.get("content-length", "0"))
        raw_body = self.rfile.read(content_length) if content_length else b"{}"
        self._forward(raw_body=raw_body)

    def _forward(self, raw_body):
        upstream_path = self.path
        requested_path = self.path
        started_at_unix_s = time.time()
        started_at_monotonic_s = time.monotonic()
        responses_request = _responses_path(self.path)
        anthropic_request = _anthropic_messages_path(self.path)
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "accept-encoding"}
        }

        request_body = None
        upstream_body = raw_body
        wants_stream = False
        responses_chat_upstream = False
        responses_chat_messages = None
        if raw_body:
            try:
                request_body = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                request_body = None
            if isinstance(request_body, dict):
                trace_request_body = dict(request_body)
                wants_stream = bool(request_body.get("stream"))
                if (
                    responses_request
                    and self.responses_upstream_api == "responses"
                ):
                    request_body = _responses_native_upstream_body(
                        request_body,
                        self.generation_overrides,
                        self.force_generation_overrides,
                    )
                    upstream_path = "/v1/responses"
                elif responses_request:
                    previous_response_id = request_body.get("previous_response_id")
                    prior_messages = None
                    if isinstance(previous_response_id, str) and previous_response_id:
                        with self.responses_chat_history_lock:
                            prior_messages = self.responses_chat_histories.get(
                                previous_response_id
                            )
                    request_body = _responses_to_chat_completion_body(
                        request_body, prior_messages=prior_messages
                    )
                    responses_chat_upstream = True
                    responses_chat_messages = copy.deepcopy(
                        request_body.get("messages") or []
                    )
                    upstream_path = "/v1/chat/completions"
                elif anthropic_request:
                    request_body = _anthropic_to_chat_completion_body(request_body)
                    upstream_path = "/v1/chat/completions"
                if upstream_path != "/v1/responses":
                    _apply_generation_overrides(
                        request_body,
                        self.generation_overrides,
                        self.force_generation_overrides,
                    )
                    request_body["stream"] = False
                    if self.force_logprobs:
                        request_body["logprobs"] = True
                    if self.force_token_ids:
                        request_body["return_token_ids"] = True
                if self.upstream_model_name:
                    request_body["model"] = self.upstream_model_name
                upstream_body = json.dumps(request_body).encode("utf-8")
                headers["content-type"] = "application/json"
            else:
                trace_request_body = None
        else:
            trace_request_body = None

        upstream_url = _join_url(self.target_base_url, upstream_path)
        backend = _backend_for_request(self.backend_mode, upstream_path)
        if backend == "litellm" and isinstance(request_body, dict):
            try:
                status, response_headers, response_body = _forward_litellm(
                    self.target_base_url,
                    request_body,
                    headers,
                    self.litellm_provider,
                )
            except Exception as e:
                print(
                    f"LiteLLM policy proxy backend failed, falling back to stdlib: {e}",
                    file=sys.stderr,
                )
                status, response_headers, response_body = (
                    _forward_stdlib_with_overflow_retry(
                        self.command,
                        upstream_url,
                        upstream_body,
                        headers,
                        request_body,
                    )
                )
        else:
            status, response_headers, response_body = (
                _forward_stdlib_with_overflow_retry(
                    self.command,
                    upstream_url,
                    upstream_body,
                    headers,
                    request_body,
                )
            )

        parsed_response = None
        if request_body is not None:
            try:
                parsed_response = json.loads(response_body.decode("utf-8"))
            except json.JSONDecodeError:
                parsed_response = None
            if isinstance(parsed_response, dict):
                if (
                    200 <= status < 300
                    and responses_chat_upstream
                    and responses_chat_messages is not None
                ):
                    assistant_message = _chat_response_assistant_message(
                        parsed_response
                    )
                    response_id = _responses_proxy_response_id(parsed_response)
                    if assistant_message is not None and response_id:
                        with self.responses_chat_history_lock:
                            self.responses_chat_histories[response_id] = [
                                *responses_chat_messages,
                                assistant_message,
                            ]
                if not _responses_response_body(parsed_response):
                    _ensure_response_token_ids(
                        self.target_base_url,
                        request_body,
                        parsed_response,
                    )
                _record_trace(
                    self.trace_path,
                    trace_request_body or request_body,
                    parsed_response,
                    self.target_base_url,
                )
        _record_model_call_observability(
            status=status,
            requested_path=requested_path,
            upstream_path=upstream_path,
            backend=backend,
            wants_stream=wants_stream,
            request_body=request_body,
            parsed_response=parsed_response,
            started_at_unix_s=started_at_unix_s,
            duration_s=time.monotonic() - started_at_monotonic_s,
        )

        if 200 <= status < 300 and wants_stream and isinstance(parsed_response, dict):
            if anthropic_request:
                _stream_anthropic_response(self, parsed_response)
            elif responses_request:
                _stream_responses_response(self, parsed_response)
            elif self.path.endswith("/chat/completions"):
                _stream_chat_response(self, parsed_response)
            else:
                _stream_completion_response(self, parsed_response)
            return

        if 200 <= status < 300 and responses_request and isinstance(
            parsed_response, dict
        ) and not _responses_response_body(parsed_response):
            created = parsed_response.get("created") or int(time.time())
            response_id = _responses_proxy_response_id(parsed_response)
            item_id = f"msg_proxy_{created}"
            response_body = json.dumps(
                _responses_response_object(
                    parsed_response,
                    response_id,
                    item_id,
                    _responses_completion_text(parsed_response),
                    "completed",
                ),
                separators=(",", ":"),
            ).encode("utf-8")
        elif 200 <= status < 300 and anthropic_request and isinstance(
            parsed_response, dict
        ):
            created = parsed_response.get("created") or int(time.time())
            message_id = parsed_response.get("id") or f"msg_proxy_{created}"
            response_body = json.dumps(
                _anthropic_message_object(
                    parsed_response,
                    message_id,
                    _completion_text(parsed_response),
                    "end_turn",
                ),
                separators=(",", ":"),
            ).encode("utf-8")

        self.send_response(status)
        content_type = response_headers.get("content-type", "application/json")
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format, *args):
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-base-url", required=True)
    parser.add_argument("--trace-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--backend",
        choices=("stdlib", "litellm", "auto"),
        default=os.environ.get("NEMO_RL_POLICY_PROXY_BACKEND", "stdlib"),
    )
    parser.add_argument(
        "--litellm-provider",
        default=os.environ.get("NEMO_RL_POLICY_PROXY_LITELLM_PROVIDER", "openai"),
    )
    parser.add_argument("--upstream-model-name", default=None)
    parser.add_argument("--generation-temperature", type=float, default=None)
    parser.add_argument("--generation-top-p", type=float, default=None)
    parser.add_argument("--generation-top-k", type=int, default=None)
    parser.add_argument("--generation-chat-template-kwargs-json", default=None)
    parser.add_argument("--force-generation-params", action="store_true")
    parser.add_argument(
        "--responses-upstream-api",
        choices=("chat", "responses"),
        default=os.environ.get(
            "NEMO_RL_POLICY_PROXY_RESPONSES_UPSTREAM_API", "chat"
        ),
    )
    args = parser.parse_args()

    generation_overrides = {}
    if args.generation_temperature is not None:
        generation_overrides["temperature"] = args.generation_temperature
    if args.generation_top_p is not None:
        generation_overrides["top_p"] = args.generation_top_p
    if args.generation_top_k is not None:
        generation_overrides["top_k"] = args.generation_top_k
    if args.generation_chat_template_kwargs_json:
        try:
            chat_template_kwargs = json.loads(
                args.generation_chat_template_kwargs_json
            )
        except json.JSONDecodeError as e:
            parser.error(f"invalid --generation-chat-template-kwargs-json: {e}")
        if not isinstance(chat_template_kwargs, dict):
            parser.error("--generation-chat-template-kwargs-json must be a JSON object")
        generation_overrides["chat_template_kwargs"] = chat_template_kwargs

    ProxyHandler.target_base_url = args.target_base_url
    ProxyHandler.trace_path = args.trace_path
    ProxyHandler.backend_mode = args.backend
    ProxyHandler.litellm_provider = args.litellm_provider
    ProxyHandler.upstream_model_name = args.upstream_model_name
    ProxyHandler.responses_upstream_api = args.responses_upstream_api
    ProxyHandler.generation_overrides = generation_overrides
    ProxyHandler.force_generation_overrides = args.force_generation_params
    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
