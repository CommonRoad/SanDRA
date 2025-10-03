import os
import signal
from json import JSONDecodeError
from typing import Any

import openai
from openai import OpenAI

try:
    from ollama import chat
    import ollama
    _client = ollama.Client()
except Exception as e:
    print(f"Failed to initialize Ollama client: {e}")
    chat = None  # or define a fallback/mock function

import json

from sandra.config import SanDRAConfiguration, PROJECT_ROOT

def ollama_client():
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Required but can be anything for Ollama
    )
    return client


def get_structured_response_online(
    user_prompt: str,
    system_prompt: str,
    schema: dict[str, Any],
    config: SanDRAConfiguration,
    save_dir: str = None,
    temperature: float = 0.6,
) -> dict[str, Any]:
    client = OpenAI(api_key=config.api_key)
    name = schema["title"]
    schema_dict = schema

    response_text_format = {
        "format": {
            "type": "json_schema",
            "name": name,
            "schema": schema_dict,
            "strict": True,
        }
    }
    response = client.responses.create(
        input=user_prompt,
        instructions=system_prompt,
        model=config.model_name,
        temperature=temperature,
        text=response_text_format,
    )

    content = response.output_text
    content_json = json.loads(content)
    if save_dir:
        prompt_json = {
            "system": system_prompt,
            "user": user_prompt,
            "schema": json.dumps(schema_dict),
        }
        save_path = os.path.join(PROJECT_ROOT, "outputs", save_dir)
        save_json = {
            "input": prompt_json,
            "output": content_json,
        }

        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "output.json"), "w") as file:
            json.dump(save_json, file)

    return content_json


def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")


def get_structured_response_offline(
        user_prompt: str,
        system_prompt: str,
        schema: dict[str, Any],
        config: SanDRAConfiguration,
        save_dir: str = None,
        temperature: float = 0.6,
        timeout: float = 30.0,
) -> dict[str, Any]:
    # Set up the timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout))

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = chat(
            messages=messages,
            model=config.model_name,
            format=schema,
            # temperature=temperature,
        )
        # Calculate duration in seconds
        signal.alarm(0)  # Cancel the alarm

        content_json = json.loads(response.message.content)

        if save_dir:
            prompt_json = {
                "system": system_prompt,
                "user": user_prompt,
                "schema": json.dumps(schema),
            }
            save_path = os.path.join(PROJECT_ROOT, "outputs", save_dir)
            save_json = {
                "input": prompt_json,
                "output": content_json,
            }

            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "output.json"), "w") as file:
                json.dump(save_json, file)

        return content_json
    except TimeoutError:
        print(f"Chat call timed out after {timeout} seconds")
        raise
    finally:
        signal.alarm(0)  # Make sure alarm is cancelled


def get_structured_response(
    user_prompt: str,
    system_prompt: str,
    schema: dict[str, Any],
    config: SanDRAConfiguration,
    save_dir: str = None,
    temperature: float = 0.6,
    retry_limit: int = 1,
) -> dict[str, Any]:
    """
    Query the API with a message consisting of:
    1. System Prompt
    2. User Prompt
    (3. Image)
    and save both request and response in text and json formats.
    """
    retries = retry_limit
    while retries > 0:
        try:
            if config.use_ollama:
                return get_structured_response_offline(
                    user_prompt,
                    system_prompt,
                    schema,
                    config,
                    save_dir=save_dir,
                    temperature=temperature,
                )
            else:
                return get_structured_response_online(
                    user_prompt,
                    system_prompt,
                    schema,
                    config,
                    save_dir=save_dir,
                    temperature=temperature,
                )
        except JSONDecodeError:
            print(f"JSON Decode Error, trying again in {retries} retries")
        except openai.APIConnectionError:
            print(f"API Connection Error, trying again in {retries} retries")
    raise ValueError("No more retries left")
