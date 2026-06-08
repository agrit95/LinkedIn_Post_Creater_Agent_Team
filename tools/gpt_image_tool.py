import base64
import json
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from crewai.tools import BaseTool, EnvVar
from openai import OpenAI
from pydantic import BaseModel, Field


class ImagePromptSchema(BaseModel):
    """Input for GPT Image Tool."""

    image_description: str = Field(
        description="Description of the image to be generated."
    )


class GPTImageTool(BaseTool):
    name: str = "GPT Image Tool"
    description: str = (
        "Generates images using OpenAI GPT Image models (gpt-image-1). "
        "Provide a detailed image_description prompt."
    )
    args_schema: type[BaseModel] = ImagePromptSchema

    model: str = "gpt-image-1"
    size: str = "1024x1024"
    quality: str = "high"
    output_path: str = "linkedin_post_image.png"

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="OPENAI_API_KEY",
                description="API key for OpenAI services",
                required=True,
            ),
        ]
    )

    def _run(self, **kwargs: Any) -> str:
        image_description = kwargs.get("image_description")
        if not image_description:
            return "Image description is required."

        client = OpenAI()
        response = client.images.generate(
            model=self.model,
            prompt=image_description,
            size=self.size,
            quality=self.quality,
            n=1,
        )

        if not response or not response.data:
            return "Failed to generate image."

        image_data = response.data[0]
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if image_data.b64_json:
            output_path.write_bytes(base64.b64decode(image_data.b64_json))
        elif image_data.url:
            urlretrieve(image_data.url, output_path)
        else:
            return "Failed to generate image: no image data in response."

        return json.dumps(
            {
                "image_path": str(output_path.resolve()),
                "image_description": image_data.revised_prompt or image_description,
            }
        )
