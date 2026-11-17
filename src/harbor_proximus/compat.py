"""Agent-local adapters for Harbor 0.20 and Harbor 0.23.

These adapters do not change Harbor classes or install process-wide shims.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.agents.base import BaseAgent as HarborBaseAgent
from harbor.models.task.config import MCPServerConfig
from pydantic import BaseModel

try:
    from harbor.agents.capabilities import AgentCapabilities
    from harbor.agents.options import AgentOptions
except ModuleNotFoundError as error:
    if error.name not in {"harbor.agents.capabilities", "harbor.agents.options"}:
        raise

    AgentOptions = BaseModel

    @dataclass(frozen=True)
    class AgentCapabilities:
        """The capability used by Proximus on older Harbor releases."""

        atif: bool = False


class BaseAgent(HarborBaseAgent):
    """Initialize Proximus options without changing other Harbor agents."""

    options_model: type[BaseModel]
    SUPPORTS_ATIF = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        logger: logging.Logger | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        skills_dir: str | None = None,
        extra_env: dict[str, str] | None = None,
        load_trajectory: str | Path | None = None,
        environment_logs_dir: PurePosixPath | None = None,
        override_setup_timeout_sec: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Validate options before Harbor constructs an environment.

        Args:
            logs_dir: Host directory for this agent's artifacts.
            model_name: Provider-qualified model identifier.
            logger: Harbor logger, or its default when omitted.
            mcp_servers: Unsupported MCP configuration, rejected when present.
            skills_dir: Unsupported skills directory, rejected when present.
            extra_env: Model credentials selected by the caller.
            load_trajectory: Unsupported session import, rejected when present.
            environment_logs_dir: Sandbox log path provided by newer Harbor.
            override_setup_timeout_sec: Harbor's setup deadline override.
            **kwargs: Proximus options; Pydantic validates this dynamic boundary.

        Raises:
            ValueError: Unsupported task features or invalid options were supplied.
        """
        if mcp_servers or skills_dir or load_trajectory:
            raise ValueError("Proximus does not support MCP, skills or session loading")
        super().__init__(
            logs_dir,
            model_name,
            logger=logger,
            mcp_servers=mcp_servers,
            skills_dir=skills_dir,
            extra_env=extra_env,
            load_trajectory=load_trajectory,
            environment_logs_dir=environment_logs_dir,
            override_setup_timeout_sec=override_setup_timeout_sec,
            **kwargs,
        )
        # 0.23 validates these in BaseAgent; 0.20 leaves kwargs to the agent.
        if not hasattr(self, "options"):
            self.options = self.options_model.model_validate(kwargs)


@dataclass(frozen=True)
class ModelConnection:
    """Only the nonserialized connection fields consumed by Proximus."""

    provider: str
    api_key: str | None = None
    configured_base_url: str | None = None


def resolve_agent_connection(
    model: str,
    lookup: Callable[..., tuple[str, str] | None],
) -> ModelConnection:
    """Resolve explicit agent credentials, leaving cloud IAM to LiteLLM.

    Args:
        model: Provider-qualified LiteLLM model identifier.
        lookup: Read the first configured variable and record its consumption.

    Returns:
        The selected key and base URL, or None for ambient provider lookup.

    Raises:
        ValueError: An older Harbor runtime cannot resolve the explicit provider.
    """
    try:
        from harbor.agents.model_connection import (
            ModelConnectionSpec,
            resolve_model_connection,
        )
    except ModuleNotFoundError as error:
        if error.name != "harbor.agents.model_connection":
            raise
    else:
        connection = resolve_model_connection(model, ModelConnectionSpec(), lookup)
        return ModelConnection(
            connection.provider,
            connection.api_key,
            connection.configured_base_url,
        )

    provider = model.partition("/")[0]
    if provider in {"bedrock", "sagemaker", "vertex_ai"}:
        return ModelConnection("amazon-bedrock" if provider == "bedrock" else provider)
    variables = {
        "anthropic": (
            ("ANTHROPIC_API_KEY",),
            ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_BASE"),
        ),
        "openai": (("OPENAI_API_KEY",), ("OPENAI_BASE_URL", "OPENAI_API_BASE")),
        "azure": (("AZURE_API_KEY",), ("AZURE_API_BASE", "AZURE_OPENAI_ENDPOINT")),
        "gemini": (("GEMINI_API_KEY", "GOOGLE_API_KEY"), ("GEMINI_API_BASE",)),
        "openrouter": (("OPENROUTER_API_KEY",), ("OPENROUTER_API_BASE",)),
        "xai": (("XAI_API_KEY",), ("XAI_API_BASE",)),
    }
    if provider not in variables:
        raise ValueError(
            f"Proximus on Harbor 0.20 does not support provider {provider!r}"
        )
    key_names, base_names = variables[provider]
    key, base = lookup(*key_names), lookup(*base_names)
    return ModelConnection(provider, key[1] if key else None, base[1] if base else None)
