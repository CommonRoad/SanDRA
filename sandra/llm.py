import os
from typing import Any

from openai import OpenAI
import json

from sandra.common.config import SanDRAConfiguration, PROJECT_ROOT


def get_structured_response(
    user_prompt: str,
    system_prompt: str,
    schema: dict[str, Any],
    config: SanDRAConfiguration,
    save_dir: str = None,
    temperature: float = 0.6,
) -> dict[str, Any]:
    """
    Query the API with a message consisting of:
    1. System Prompt
    2. User Prompt
    (3. Image)
    and save both request and response in text and json formats.
    """
    client = OpenAI(
        api_key=config.api_key
    )
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
