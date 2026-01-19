"""Custom LLM wrapper for DeepEval with OpenAI or BackendAI support.

Uses LangChain's .with_structured_output() for Pydantic schema parsing.
"""

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM

from konte.config.settings import settings


class BackendAIModel(DeepEvalBaseLLM):
    """Custom LLM for DeepEval using OpenAI or BackendAI endpoint."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        # Use BackendAI if configured, otherwise use OpenAI
        if settings.use_backendai:
            self.model_name = model_name or settings.BACKENDAI_MODEL_NAME
            self.base_url = base_url or settings.BACKENDAI_ENDPOINT
            self.api_key = api_key or settings.BACKENDAI_API_KEY or "placeholder"
            self._llm = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.0,
                max_tokens=8000,
            )
        else:
            self.model_name = model_name or settings.CONTEXT_MODEL
            self.base_url = None  # Use OpenAI default
            self.api_key = api_key or settings.OPENAI_API_KEY
            self._llm = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                temperature=0.0,
                max_tokens=8000,
            )

    def load_model(self):
        """Return the model identifier."""
        return self.model_name

    def generate(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        """Synchronous generation with enforced structured output.

        DeepEval requires this signature to enforce schema output.
        ALWAYS returns a valid schema instance - never raises exception.
        """
        import json

        raw_response = ""

        try:
            # Try with json_mode (more compatible than json_schema with vLLM)
            structured_llm = self._llm.with_structured_output(schema, method="json_mode")
            result = structured_llm.invoke(prompt)
            if result is not None:
                return result
        except Exception:
            pass

        # Fallback: raw generation with manual JSON parsing
        try:
            response = self._llm.invoke(prompt)
            raw_response = response.content if hasattr(response, 'content') else str(response)
            data = json.loads(raw_response)
            return schema.model_validate(data)
        except Exception:
            pass

        # Ultimate fallback: create default schema
        return self._create_default_schema(schema, raw_response)

    async def a_generate(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        """Asynchronous generation with enforced structured output.

        DeepEval requires this signature to enforce schema output.
        ALWAYS returns a valid schema instance - never raises exception.
        """
        import json

        raw_response = ""

        try:
            # Try with json_mode (more compatible than json_schema with vLLM)
            structured_llm = self._llm.with_structured_output(schema, method="json_mode")
            result = await structured_llm.ainvoke(prompt)
            if result is not None:
                return result
        except Exception:
            pass

        # Fallback: raw generation with manual JSON parsing
        try:
            response = await self._llm.ainvoke(prompt)
            raw_response = response.content if hasattr(response, 'content') else str(response)
            data = json.loads(raw_response)
            return schema.model_validate(data)
        except Exception:
            pass

        # Ultimate fallback: create default schema
        return self._create_default_schema(schema, raw_response)

    def _create_default_schema(self, schema: type[BaseModel], raw_response: str) -> BaseModel:
        """Create a default schema instance with fallback values."""
        import json

        # Try to extract any JSON from the response
        try:
            # Find JSON in the response
            start = raw_response.find("{")
            end = raw_response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw_response[start:end])
            else:
                data = {}
        except Exception:
            data = {}

        # Fill in missing required fields with defaults
        for field_name, field_info in schema.model_fields.items():
            if field_name not in data:
                # Provide sensible defaults based on field type
                annotation = field_info.annotation
                if annotation == str or (hasattr(annotation, "__origin__") and annotation.__origin__ == str):
                    data[field_name] = "unknown"
                elif annotation == bool:
                    data[field_name] = False
                elif annotation == int:
                    data[field_name] = 0
                elif annotation == float:
                    data[field_name] = 0.0
                elif hasattr(annotation, "__origin__") and annotation.__origin__ == list:
                    data[field_name] = []
                else:
                    data[field_name] = None

        # Handle nested 'verdict' fields in lists (common DeepEval pattern)
        if "verdicts" in data and isinstance(data["verdicts"], list):
            for verdict_item in data["verdicts"]:
                if isinstance(verdict_item, dict) and "verdict" not in verdict_item:
                    verdict_item["verdict"] = "no"  # Default to "no" for safety

        return schema.model_validate(data)

    def get_model_name(self) -> str:
        """Return model name for logging."""
        return self.model_name
