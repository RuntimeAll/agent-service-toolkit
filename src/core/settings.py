from enum import StrEnum
from json import loads
from typing import Annotated, Any

from dotenv import find_dotenv
from pydantic import (
    BeforeValidator,
    Field,
    HttpUrl,
    SecretStr,
    TypeAdapter,
    computed_field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from schema.models import (
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    AzureOpenAIModelName,
    DeepseekModelName,
    FakeModelName,
    GoogleModelName,
    GroqModelName,
    OllamaModelName,
    OpenAICompatibleName,
    OpenAIModelName,
    OpenRouterModelName,
    Provider,
    VertexAIModelName,
)


class DatabaseType(StrEnum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MONGO = "mongo"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def to_logging_level(self) -> int:
        """Convert to Python logging level constant."""
        import logging

        mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }
        return mapping[self]


def check_str_is_http(x: str) -> str:
    http_url_adapter = TypeAdapter(HttpUrl)
    return str(http_url_adapter.validate_python(x))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    MODE: str | None = None

    HOST: str = "0.0.0.0"
    PORT: int = 8080
    GRACEFUL_SHUTDOWN_TIMEOUT: int = 30
    LOG_LEVEL: LogLevel = LogLevel.WARNING

    AUTH_SECRET: SecretStr | None = None

    OPENAI_API_KEY: SecretStr | None = None
    DEEPSEEK_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    GOOGLE_API_KEY: SecretStr | None = None
    GOOGLE_APPLICATION_CREDENTIALS: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None
    USE_AWS_BEDROCK: bool = False
    OLLAMA_MODEL: str | None = None
    OLLAMA_BASE_URL: str | None = None
    USE_FAKE_MODEL: bool = False
    OPENROUTER_API_KEY: str | None = None

    # If DEFAULT_MODEL is None, it will be set in model_post_init
    DEFAULT_MODEL: AllModelEnum | None = None  # type: ignore[assignment]
    AVAILABLE_MODELS: set[AllModelEnum] = set()  # type: ignore[assignment]

    # Set openai compatible api, mainly used for proof of concept
    COMPATIBLE_MODEL: str | None = None
    COMPATIBLE_API_KEY: SecretStr | None = None
    COMPATIBLE_BASE_URL: str | None = None

    OPENWEATHERMAP_API_KEY: SecretStr | None = None

    # MCP Configuration
    GITHUB_PAT: SecretStr | None = None
    MCP_GITHUB_SERVER_URL: str = "https://api.githubcopilot.com/mcp/"

    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "default"
    LANGCHAIN_ENDPOINT: Annotated[str, BeforeValidator(check_str_is_http)] = (
        "https://api.smith.langchain.com"
    )
    LANGCHAIN_API_KEY: SecretStr | None = None

    LANGFUSE_TRACING: bool = False
    LANGFUSE_HOST: Annotated[str, BeforeValidator(check_str_is_http)] = "https://cloud.langfuse.com"
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    LANGFUSE_SECRET_KEY: SecretStr | None = None

    # Database Configuration
    DATABASE_TYPE: DatabaseType = (
        DatabaseType.SQLITE
    )  # Options: DatabaseType.SQLITE or DatabaseType.POSTGRES
    SQLITE_DB_PATH: str = "checkpoints.db"

    # PostgreSQL Configuration
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: SecretStr | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: int | None = None
    POSTGRES_DB: str | None = None
    POSTGRES_APPLICATION_NAME: str = "agent-service-toolkit"
    POSTGRES_MIN_CONNECTIONS_PER_POOL: int = 1
    POSTGRES_MAX_CONNECTIONS_PER_POOL: int = 1

    # MongoDB Configuration
    MONGO_HOST: str | None = None
    MONGO_PORT: int | None = None
    MONGO_DB: str | None = None
    MONGO_USER: str | None = None
    MONGO_PASSWORD: SecretStr | None = None
    MONGO_AUTH_SOURCE: str | None = None

    # === PRD-C-009 举一反三 agent 专用配置 ===
    # 只读图谱 SQL：dev 库 miskt_data2 @ :3307（口令从 .env 读，别明文散落）
    VARIANT_DB_HOST: str = "127.0.0.1"
    VARIANT_DB_PORT: int = 3307
    VARIANT_DB_USER: str = "root"
    VARIANT_DB_PASSWORD: SecretStr | None = None
    VARIANT_DB_NAME: str = "miskt_data2"
    # RuoYi 底座（C 线 book-server :8090）入库用，双头鉴权
    RUOYI_BASE_URL: str = "http://localhost:8090"
    RUOYI_USERNAME: str = "teacher001"
    RUOYI_PASSWORD: SecretStr | None = None
    RUOYI_CLIENT_ID: str = "e5cd7e4891bf95d1d19206ce24a7b32e"
    RUOYI_TENANT_ID: str = "000000"
    RUOYI_TOKEN: str | None = None
    # 思考型模型读图建议 max_tokens（≥4096）
    VARIANT_MAX_TOKENS: int = 4096
    # 🔴 PRD-C-100 B0·H5 母题 opus 一把节点宽护栏（go/no-go 实测定）：6 张硬带图样本 completion
    #   分布 min=1324/p50=3707/max=5158、16000 上限零截断 → 旧 4096 会截断（5158>4096）。
    #   母题节点 = 宽护栏防失控不截断（D6，非省钱杠杆）；定 12288 ≈ 2.4× 实测峰值，留思考型偶发头。
    #   .env MOTHER_OPUS_MAX_TOKENS 可覆盖（生产期按真实分布调）。
    MOTHER_OPUS_MAX_TOKENS: int = 12288
    # 🔴 PRD-C-100 B3·造图翻命令模型（§10 契约「opus 翻 GeoGebra 命令」）：默认 opus（命令质量稳，
    #   且当前促销价 opus 0.007/0.035 比 gpt-5.4 0.0108/0.0648 还便宜）。成本敏感期可经 .env 切
    #   VARIANT_MODEL_FIGURE=gpt-5.4-mini。造图翻命令走 relay_pool 落 conv_trace(label=figure_geogebra)。
    VARIANT_MODEL_FIGURE: str = "claude-opus-4-8"
    # 🔴 PRD-C-100 B5·单一全局日预算护栏（D6/§10）：当日 conv_trace 累计花费 ≥ 此阈值（¥）→
    #   母题 opus 一把**拦截**（不调，提示老师稍后/明日再试）、造图**降级**（needs_figure，不调翻命令）。
    #   None/≤0 = 关（不限，默认）。三级预算（会话/老师/日）推多用户期，本轮只做单一全局日。
    GLOBAL_DAILY_BUDGET_YUAN: float | None = None
    # 闸B 回炉（REGEN）瘦身 max_tokens 上限（整改4·2026-06-12）：回炉只带单题题面+错因+确定
    # 上下文块，比首稿出题（一次出 N 道）小得多 → 单独压一个上限防输出失控（ct 9k-11k）。
    # 出题主调用（generate/add）仍走 VARIANT_MAX_TOKENS，不动。≤0 视为回退 VARIANT_MAX_TOKENS。
    VARIANT_REGEN_MAX_TOKENS: int = 2048
    # 教材版本（整改1·2026-06-12）：生题确定上下文块的「教材版本」行。RuoYi 知识点树/库当前无
    # 教材版本字段 → 不编造，默认空；填了（如「人教版」「浙教版」）才注入该行。
    TEXTBOOK_VERSION: str = ""
    # === PRD-C-013 P13 预算闸：state 级 LLM 调用计数上限（per-round 重置）===
    # 超限后**增强类**调用（闸A rework / 闸B heal / replenish 补题 / extract 兜底）跳过
    # 走既有 G5 降级路径（标 ⚠ / 保留原题，不卡死）；核心链（parse/generate/grade）不跳。
    # 出题轮预算更宽（首稿 + 三题 eager 双闸 + 难度总评）；编辑轮窄（单题重验）。
    VARIANT_BUDGET_GENERATE: int = 18
    VARIANT_BUDGET_EDIT: int = 6

    # === PRD-C-011 Block B：LLM 出口多中转站主备 + 熔断转移 ===
    # RELAY_POOL = JSON 数组（主→备有序）：[{"name","base_url","api_key","model"}]
    # 留空 → 从 COMPATIBLE_* 派生单中转站（名字取 RELAY_NAME）。
    RELAY_POOL: str | None = None
    # 单中转站（RELAY_POOL 为空）时的展示名（也是 conv_trace.relay 列默认值）。
    # 中性默认 "compatible"，跟着 COMPATIBLE_* 派生站走。
    RELAY_NAME: str = "compatible"
    # 轻活模型（S1.1）：难度总评等无识图、可降本的调用点经 per-call model 覆盖走它；
    # 留空 → 各调用点退回默认（relay 配置 model），行为不变。
    LLM_MODEL_LIGHT: str = "gpt-5.4-nano"

    # === 按环节分档模型路由（PRD-C-009 变式·2026-06-12）===
    # 举一反三管线按「环节」分档配模型：前置抽取环节降本（nano），深度思考档（gpt-5.4）只留
    # 给真正出题。四项 .env 可覆盖；缺省（None）→ 走 variant_model() 的回退链 = 现行为。
    #   ANALYZE = 读图+配方（多模态）；DNA = 锚定/标签池 refine；SOLVE = 闸B 独立重解+载荷抽取；
    #   GENERATE = 出题/回炉/补题/solution_only 重写（红线：这次不降档，默认仍 COMPATIBLE_MODEL）。
    # 换 deepseek 等：只改这四个 .env 项，代码不动。
    VARIANT_MODEL_ANALYZE: str | None = None
    VARIANT_MODEL_DNA: str | None = None
    VARIANT_MODEL_SOLVE: str | None = None
    VARIANT_MODEL_GENERATE: str | None = None
    # 🔴 PRD-C-015 批2·W1' 模型确认档（H2 预飞行推翻 nano）：候选池内确认解题模型用 gpt-5.4-mini
    #   +「先解题再选」prompt（nano recall 仅 24% 欠选成性；gpt-5.4 主模型 31% 发散；mini recall 69%/
    #   头牌 9/9）。缺省（None）→ 回退 LLM_MODEL_LIGHT（nano）；.env 显式切 gpt-5.4-mini 生效（推荐配）。
    VARIANT_MODEL_MODEL_CONFIRM: str | None = None
    # 🔴 PRD-C-017 F1 死键防护：母题解题+打标合并调用专用模型档（opus 4.8 经中转）。
    #   variant_model("mother_solve_label") 必须命中此档（默认即 claude-opus-4-8），否则
    #   死键返 None → _ainvoke_text(model=None) → relay 退站配 gpt-5.4，opus 静默不被调用，
    #   整卡核心价值蒸发。本档**默认值就是 opus**（不靠 .env 才生效），.env 可覆盖核中转真名。
    VARIANT_MODEL_MOTHER_SOLVE_LABEL: str = "claude-opus-4-8"
    RELAY_FAIL_THRESHOLD: int = 3  # 连续失败 N 次 → 熔断器 trip
    RELAY_COOLDOWN_S: float = 30.0  # 熔断冷却秒数（之后 half-open 探活）
    # RELAY_PRICES = JSON：{"<model>": {"in": ¥/1k_prompt_tokens, "out": ¥/1k_completion_tokens}}
    # 用于算 conv_trace.cost_yuan（实际消费）。留空 → cost 记 NULL（不瞎猜价）。
    RELAY_PRICES: str | None = None

    # Azure OpenAI Settings
    AZURE_OPENAI_API_KEY: SecretStr | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    AZURE_OPENAI_DEPLOYMENT_MAP: dict[str, str] = Field(
        default_factory=dict, description="Map of model names to Azure deployment IDs"
    )

    def model_post_init(self, __context: Any) -> None:
        api_keys = {
            Provider.OPENAI: self.OPENAI_API_KEY,
            Provider.OPENAI_COMPATIBLE: self.COMPATIBLE_BASE_URL and self.COMPATIBLE_MODEL,
            Provider.DEEPSEEK: self.DEEPSEEK_API_KEY,
            Provider.ANTHROPIC: self.ANTHROPIC_API_KEY,
            Provider.GOOGLE: self.GOOGLE_API_KEY,
            Provider.VERTEXAI: self.GOOGLE_APPLICATION_CREDENTIALS,
            Provider.GROQ: self.GROQ_API_KEY,
            Provider.AWS: self.USE_AWS_BEDROCK,
            Provider.OLLAMA: self.OLLAMA_MODEL,
            Provider.FAKE: self.USE_FAKE_MODEL,
            Provider.AZURE_OPENAI: self.AZURE_OPENAI_API_KEY,
            Provider.OPENROUTER: self.OPENROUTER_API_KEY,
        }
        active_keys = [k for k, v in api_keys.items() if v]
        if not active_keys:
            raise ValueError("At least one LLM API key must be provided.")

        for provider in active_keys:
            match provider:
                case Provider.OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAIModelName.GPT_5_NANO
                    self.AVAILABLE_MODELS.update(set(OpenAIModelName))
                case Provider.OPENAI_COMPATIBLE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAICompatibleName.OPENAI_COMPATIBLE
                    self.AVAILABLE_MODELS.update(set(OpenAICompatibleName))
                case Provider.DEEPSEEK:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = DeepseekModelName.DEEPSEEK_CHAT
                    self.AVAILABLE_MODELS.update(set(DeepseekModelName))
                case Provider.ANTHROPIC:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AnthropicModelName.HAIKU_45
                    self.AVAILABLE_MODELS.update(set(AnthropicModelName))
                case Provider.GOOGLE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GoogleModelName.GEMINI_20_FLASH
                    self.AVAILABLE_MODELS.update(set(GoogleModelName))
                case Provider.VERTEXAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = VertexAIModelName.GEMINI_20_FLASH
                    self.AVAILABLE_MODELS.update(set(VertexAIModelName))
                case Provider.GROQ:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GroqModelName.LLAMA_31_8B
                    self.AVAILABLE_MODELS.update(set(GroqModelName))
                case Provider.AWS:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AWSModelName.BEDROCK_HAIKU
                    self.AVAILABLE_MODELS.update(set(AWSModelName))
                case Provider.OLLAMA:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OllamaModelName.OLLAMA_GENERIC
                    self.AVAILABLE_MODELS.update(set(OllamaModelName))
                case Provider.OPENROUTER:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenRouterModelName.GEMINI_25_FLASH
                    self.AVAILABLE_MODELS.update(set(OpenRouterModelName))
                case Provider.FAKE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = FakeModelName.FAKE
                    self.AVAILABLE_MODELS.update(set(FakeModelName))
                case Provider.AZURE_OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AzureOpenAIModelName.AZURE_GPT_4O_MINI
                    self.AVAILABLE_MODELS.update(set(AzureOpenAIModelName))
                    # Validate Azure OpenAI settings if Azure provider is available
                    if not self.AZURE_OPENAI_API_KEY:
                        raise ValueError("AZURE_OPENAI_API_KEY must be set")
                    if not self.AZURE_OPENAI_ENDPOINT:
                        raise ValueError("AZURE_OPENAI_ENDPOINT must be set")
                    if not self.AZURE_OPENAI_DEPLOYMENT_MAP:
                        raise ValueError("AZURE_OPENAI_DEPLOYMENT_MAP must be set")

                    # Parse deployment map if it's a string
                    if isinstance(self.AZURE_OPENAI_DEPLOYMENT_MAP, str):
                        try:
                            self.AZURE_OPENAI_DEPLOYMENT_MAP = loads(
                                self.AZURE_OPENAI_DEPLOYMENT_MAP
                            )
                        except Exception as e:
                            raise ValueError(f"Invalid AZURE_OPENAI_DEPLOYMENT_MAP JSON: {e}")

                    # Validate required deployments exist
                    required_models = {"gpt-4o", "gpt-4o-mini"}
                    missing_models = required_models - set(self.AZURE_OPENAI_DEPLOYMENT_MAP.keys())
                    if missing_models:
                        raise ValueError(f"Missing required Azure deployments: {missing_models}")
                case _:
                    raise ValueError(f"Unknown provider: {provider}")

    def variant_model(self, env: str) -> str | None:
        """按环节解析变式管线该用哪个模型（per-call model 覆盖值）。

        env ∈ {"analyze","dna","solve","generate","model_confirm","mother_solve_label"}。
        返回值传给 _ainvoke_text(model=...)：
        - 返回 str → 该次请求换这个 model（站点不变）；
        - 返回 None → 沿用 relay 配置 model（= COMPATIBLE_MODEL，深度思考档），旧行为。

        缺省（对应 VARIANT_MODEL_* 为 None）即「回退现行为」：
        - analyze   → None（多模态读图回退深度档；🔴 2026-06-13 A/B 实测：真名 gpt-5.4-nano
                      读图 3/3 准、parsed_ok=true（25-37s），可降——经 .env VARIANT_MODEL_ANALYZE
                      显式切 nano；deepseek-v4-flash 读图会幻觉年级（confidence 0.9 却错档），禁用于读图）
        - dna       → LLM_MODEL_LIGHT（锚定/标签是池内选 id 的分类活，本就走 nano）
        - solve     → None（闸B 独立重解+载荷抽取回退深度档=gpt-5.4。🔴 2026-06-13 维护者裁定
                      **留空回退 5.4，不切 nano**：独立重解是阅卷角色，nano 误判 FAIL 会触发假回炉
                      （2026-06-13 nano 冒烟疑点：3/3 全回炉），且 5.4 与 nano 速度本就相当(9.4s vs 8s)。
                      先前「A/B 判决 3/3=100% 一致可降」结论被该冒烟疑点推翻；ANALYZE/DNA 仍 nano）
        - generate  → None（出题/回炉/补题/重写，红线不降档，保 COMPATIBLE_MODEL=gpt-5.4）

        🔴 昨晚（2026-06-12）误评根因：旧默认名 gpt-5-nano 属 gpt-5 老系、中转站 not found/空返，
           非 nano 能力问题。真名 gpt-5.4-nano 两站点皆挂。证据=tools/model_ab_nano_vs_deepseek 产物
           （model_ab_vision_nano.out / model_ab_solve_nano.out / model_ab_models_inventory.out）。
        """
        override = {
            "analyze": self.VARIANT_MODEL_ANALYZE,
            "dna": self.VARIANT_MODEL_DNA,
            "solve": self.VARIANT_MODEL_SOLVE,
            "generate": self.VARIANT_MODEL_GENERATE,
            # 模型确认档（PRD-C-015 批2·H2）：候选池内确认解题模型。
            "model_confirm": self.VARIANT_MODEL_MODEL_CONFIRM,
            # 🔴 PRD-C-017 F1：母题解题+打标合并调用档（opus 4.8）。override 这里读 .env 覆盖值；
            #   注意此键的 setting 本身**有默认值 claude-opus-4-8**（非 None），故 override 永不为空，
            #   下方 defaults 的同名项只作冗余护栏（理论上不会走到），双保险防死键退 gpt-5.4。
            "mother_solve_label": self.VARIANT_MODEL_MOTHER_SOLVE_LABEL,
        }.get(env)
        if override:
            return override
        # 缺省回退链（= 现行为；改默认值在此处一处定，调用点不重复）
        defaults: dict[str, str | None] = {
            "analyze": None,
            "dna": self.LLM_MODEL_LIGHT,
            "solve": None,
            "generate": None,
            # 缺省回退 nano（与 dna 同档）；.env 显式切 gpt-5.4-mini（H2 甜点档）覆盖。
            "model_confirm": self.LLM_MODEL_LIGHT,
            # 🔴 PRD-C-017 F1 冗余护栏：母题档即使 override 被人误清空也绝不回退到 None/gpt-5.4。
            #   母题侧零机器验证（06-15 去 sympy）→ opus 必须真被调用是唯一安全网，宁可硬钉死。
            "mother_solve_label": "claude-opus-4-8",
        }
        return defaults.get(env)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def BASE_URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"

    def is_dev(self) -> bool:
        return self.MODE == "dev"


settings = Settings()

# 🔴 PRD-C-017 F1·G3 反性自检（启动期 fail-fast）：母题解题+打标必须真命中 opus，
#   否则 variant_model 死键会让它静默退 gpt-5.4(5/9)——母题侧零机器验证（06-15 去 sympy），
#   这是本卡核心价值的唯一安全网。进程起不来好过母题悄悄用错模型解出歪基准、下游全错查不到根因。
MOTHER_SOLVE_MODEL_EXPECTED = "claude-opus-4-8"


def assert_mother_solve_hits_opus(s: "Settings") -> str:
    """🔴 PRD-C-017 F1·G3 反性自检：母题解题+打标必须真命中 opus，否则 raise。

    死键会让 variant_model 静默退 gpt-5.4(5/9)——母题侧零机器验证（06-15 去 sympy），
    这是本卡核心价值的唯一安全网。进程起不来好过母题悄悄用错模型解出歪基准、下游全错查不到根因。
    返回实际解析到的模型名（== MOTHER_SOLVE_MODEL_EXPECTED）。"""
    resolved = s.variant_model("mother_solve_label")
    if resolved != MOTHER_SOLVE_MODEL_EXPECTED:
        raise ValueError(
            f"PRD-C-017 F1 fail-fast: variant_model('mother_solve_label') = "
            f"{resolved!r}，期望 {MOTHER_SOLVE_MODEL_EXPECTED!r}。"
            "母题解题+打标必须命中 opus，死键退 gpt-5.4 会让放大器基准悄悄歪掉。"
            "请检查 settings.variant_model override/defaults 表与 VARIANT_MODEL_MOTHER_SOLVE_LABEL。"
        )
    return resolved


# 启动期触发（import settings 即跑）：母题档不命中 opus → 整进程起不来（设计如此）。
assert_mother_solve_hits_opus(settings)
