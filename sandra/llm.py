import os
from json import JSONDecodeError
from typing import Any

from openai import OpenAI
from ollama import chat
import json

from pydantic import BaseModel

from sandra.common.config import SanDRAConfiguration, PROJECT_ROOT


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


def get_structured_response_offline(
    user_prompt: str,
    system_prompt: str,
    schema: dict[str, Any],
    config: SanDRAConfiguration,
    save_dir: str = None,
    temperature: float = 0.6,
) -> dict[str, Any]:
    response = chat(
        messages=[
            # {
            #     'role': 'system',
            #     'content': system_prompt,
            # },
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        # options={
        #     'temperature': temperature,
        # },
        model=config.model_name,
        format=schema,
    )
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
    raise JSONDecodeError("No more retries left")


if __name__ == "__main__":

    class Country(BaseModel):
        name: str
        capital: str
        languages: list[str]

    client = ollama_client()
    user_prompt = "Tell me about Canada."
    system_prompt = "Tell me about Canada."
    config = SanDRAConfiguration()
    config.model_name = "qwen3:14b"
    # completion = client.beta.chat.completions.parse(
    #     model="qwen3:14b",
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt},
    #     ],
    #     response_format=Country,
    # )
    #
    # event = completion.choices[0].message.parsed
    # print(event)
    response = get_structured_response(
        user_prompt, system_prompt, Country.model_json_schema(), config
    )
    print(response)
