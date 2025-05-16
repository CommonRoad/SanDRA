import os
from openai import BaseModel, OpenAI
import json
from config import SaLaRAConfiguration, PROJECT_ROOT


def get_structured_response(
    user_prompt: str,
    system_prompt: str,
    schema: BaseModel | dict,
    config: SaLaRAConfiguration,
    save_dir: str = None,
    temperature: float = 0.6,
) -> dict:
    """
    Query the API with a message consisting of:
    1. System Prompt
    2. User Prompt
    (3. Image)
    and save both request and response in text and json formats.
    """
    client = OpenAI(api_key=config.api_key)
    if isinstance(schema, dict):
        name = schema["title"]
    else:
        name = schema.__name__
        schema = schema.model_json_schema()

    response_text_format = {
        "format": {
            "type": "json_schema",
            "name": name,
            "schema": schema,
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
