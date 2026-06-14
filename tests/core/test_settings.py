import json
import logging
import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from core.settings import LogLevel, Settings, check_str_is_http
from schema.models import (
    AnthropicModelName,
    AzureOpenAIModelName,
    OpenAIModelName,
    VertexAIModelName,
)


def test_check_str_is_http():
    # Test valid HTTP URLs
    assert check_str_is_http("http://example.com/") == "http://example.com/"
    assert check_str_is_http("https://api.test.com/") == "https://api.test.com/"

    # Test invalid URLs
    with pytest.raises(ValidationError):
        check_str_is_http("not_a_url")
    with pytest.raises(ValidationError):
        check_str_is_http("ftp://invalid.com")


def test_settings_default_values():
    settings = Settings(_env_file=None)
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8080
    assert settings.USE_AWS_BEDROCK is False
    assert settings.USE_FAKE_MODEL is False


def test_settings_no_api_keys():
    # Test that settings raises error when no API keys are provided
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="At least one LLM API key must be provided"):
            _ = Settings(_env_file=None)


def test_settings_with_openai_key():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.OPENAI_API_KEY == SecretStr("test_key")
        assert settings.DEFAULT_MODEL == OpenAIModelName.GPT_5_NANO
        assert settings.AVAILABLE_MODELS == set(OpenAIModelName)


def test_settings_with_anthropic_key():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.ANTHROPIC_API_KEY == SecretStr("test_key")
        assert settings.DEFAULT_MODEL == AnthropicModelName.HAIKU_45
        assert settings.AVAILABLE_MODELS == set(AnthropicModelName)


def test_settings_with_vertexai_credentials_file():
    with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "test_key"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.GOOGLE_APPLICATION_CREDENTIALS == SecretStr("test_key")
        assert settings.DEFAULT_MODEL == VertexAIModelName.GEMINI_20_FLASH
        assert settings.AVAILABLE_MODELS == set(VertexAIModelName)


def test_settings_with_multiple_api_keys():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test_openai_key",
            "ANTHROPIC_API_KEY": "test_anthropic_key",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.OPENAI_API_KEY == SecretStr("test_openai_key")
        assert settings.ANTHROPIC_API_KEY == SecretStr("test_anthropic_key")
        # When multiple providers are available, OpenAI should be the default
        assert settings.DEFAULT_MODEL == OpenAIModelName.GPT_5_NANO
        # Available models should include exactly all OpenAI and Anthropic models
        expected_models = set(OpenAIModelName)
        expected_models.update(set(AnthropicModelName))
        assert settings.AVAILABLE_MODELS == expected_models


def test_settings_base_url():
    settings = Settings(HOST="0.0.0.0", PORT=8000, _env_file=None)
    assert settings.BASE_URL == "http://0.0.0.0:8000"


def test_settings_is_dev():
    settings = Settings(MODE="dev", _env_file=None)
    assert settings.is_dev() is True

    settings = Settings(MODE="prod", _env_file=None)
    assert settings.is_dev() is False


def test_settings_with_azure_openai_key():
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT_MAP": '{"gpt-4o": "deployment-1", "gpt-4o-mini": "deployment-2"}',
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.AZURE_OPENAI_API_KEY.get_secret_value() == "test_key"
        assert settings.DEFAULT_MODEL == AzureOpenAIModelName.AZURE_GPT_4O_MINI
        assert settings.AVAILABLE_MODELS == set(AzureOpenAIModelName)


def test_settings_with_both_openai_and_azure():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test_openai_key",
            "AZURE_OPENAI_API_KEY": "test_azure_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT_MAP": '{"gpt-4o": "deployment-1", "gpt-4o-mini": "deployment-2"}',
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.OPENAI_API_KEY == SecretStr("test_openai_key")
        assert settings.AZURE_OPENAI_API_KEY == SecretStr("test_azure_key")
        # When multiple providers are available, OpenAI should be the default
        assert settings.DEFAULT_MODEL == OpenAIModelName.GPT_5_NANO
        # Available models should include both OpenAI and Azure OpenAI models
        expected_models = set(OpenAIModelName)
        expected_models.update(set(AzureOpenAIModelName))
        assert settings.AVAILABLE_MODELS == expected_models


def test_settings_azure_deployment_names():
    # Delete this test
    pass


def test_settings_azure_missing_deployment_names():
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        },
        clear=True,
    ):
        with pytest.raises(ValidationError, match="AZURE_OPENAI_DEPLOYMENT_MAP must be set"):
            Settings(_env_file=None)


def test_settings_azure_deployment_map():
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT_MAP": '{"gpt-4o": "deploy1", "gpt-4o-mini": "deploy2"}',
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.AZURE_OPENAI_DEPLOYMENT_MAP == {
            "gpt-4o": "deploy1",
            "gpt-4o-mini": "deploy2",
        }


def test_settings_azure_invalid_deployment_map():
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT_MAP": '{"gpt-4o": "deploy1"}',  # Missing required model
        },
        clear=True,
    ):
        with pytest.raises(ValueError, match="Missing required Azure deployments"):
            Settings(_env_file=None)


def test_settings_azure_openai():
    """Test Azure OpenAI settings."""
    deployment_map = {"gpt-4o": "deployment1", "gpt-4o-mini": "deployment2"}
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT_MAP": json.dumps(deployment_map),
        },
    ):
        settings = Settings(_env_file=None)
        assert settings.AZURE_OPENAI_API_KEY.get_secret_value() == "test-key"
        assert settings.AZURE_OPENAI_ENDPOINT == "https://test.openai.azure.com"
        assert settings.AZURE_OPENAI_DEPLOYMENT_MAP == deployment_map


def test_log_level_enum():
    """Test LogLevel enum and its conversion to logging levels."""
    assert LogLevel.DEBUG.to_logging_level() == logging.DEBUG
    assert LogLevel.INFO.to_logging_level() == logging.INFO
    assert LogLevel.WARNING.to_logging_level() == logging.WARNING
    assert LogLevel.ERROR.to_logging_level() == logging.ERROR
    assert LogLevel.CRITICAL.to_logging_level() == logging.CRITICAL


def test_settings_log_level_default():
    """Test that LOG_LEVEL defaults to WARNING."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.LOG_LEVEL == LogLevel.WARNING
        assert settings.LOG_LEVEL.to_logging_level() == logging.WARNING


def test_settings_log_level_from_env():
    """Test that LOG_LEVEL can be set from environment variable."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key", "LOG_LEVEL": "DEBUG"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.LOG_LEVEL == LogLevel.DEBUG
        assert settings.LOG_LEVEL.to_logging_level() == logging.DEBUG


def test_settings_log_level_invalid():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key", "LOG_LEVEL": "INVALID"}, clear=True):
        with pytest.raises(ValueError, match="validation error for Settings\nLOG_LEVEL\n"):
            Settings(_env_file=None)


# === PRD-C-017 B0 · F1 死键防护 + M9 温度覆盖 ===


def test_variant_model_mother_solve_label_hits_opus():
    """🔴 F1·G3 反性自检：variant_model('mother_solve_label') 必须命中 opus，绝不返 None。

    死键会让母题解题悄悄退 gpt-5.4(5/9)；母题侧零机器验证（06-15 去 sympy），这是唯一安全网。"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        s = Settings(_env_file=None)
        assert s.variant_model("mother_solve_label") == "claude-opus-4-8"


def test_variant_model_mother_solve_label_env_override():
    """.env 可覆盖母题档真名（换中转后核对真名用），但仍必须是显式值、不退 None。"""
    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test_key", "VARIANT_MODEL_MOTHER_SOLVE_LABEL": "claude-opus-4-8-x"},
        clear=True,
    ):
        s = Settings(_env_file=None)
        assert s.variant_model("mother_solve_label") == "claude-opus-4-8-x"


def test_variant_model_unknown_key_returns_none():
    """护栏对照：未注册键仍返 None（证明 mother_solve_label 是被显式加进两张表、非偶然命中）。"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        s = Settings(_env_file=None)
        assert s.variant_model("not_a_real_label") is None


def test_mother_solve_fail_fast_passes_for_default():
    """启动期 fail-fast 断言：默认配置下母题档命中 opus，不抛（import settings 已证一次）。"""
    from core.settings import assert_mother_solve_hits_opus

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        s = Settings(_env_file=None)
        assert assert_mother_solve_hits_opus(s) == "claude-opus-4-8"


def test_mother_solve_fail_fast_raises_on_death_path():
    """反性自检：构造死键情形（variant_model 退回 None）→ fail-fast 必须 raise，不静默退 gpt-5.4。"""
    from core.settings import MOTHER_SOLVE_MODEL_EXPECTED, assert_mother_solve_hits_opus

    class _DeadKey:
        def variant_model(self, env):  # 模拟死键：母题档返 None（→ relay 退 gpt-5.4）
            return None

    with pytest.raises(ValueError, match="F1 fail-fast"):
        assert_mother_solve_hits_opus(_DeadKey())  # type: ignore[arg-type]
    assert MOTHER_SOLVE_MODEL_EXPECTED == "claude-opus-4-8"
