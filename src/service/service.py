import inspect
import json
import logging
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.langchain import (
    CallbackHandler,  # type: ignore[import-untyped]
)
from langgraph.types import Command, Interrupt
from langsmith import Client as LangsmithClient
from langsmith import uuid7

from pydantic import BaseModel, field_validator

from agents import DEFAULT_AGENT, AgentGraph, get_agent, get_all_agent_info, load_agent
from agents import conv_trace
from core import settings
from memory import initialize_database, initialize_store
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
)
from service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 🔴 PRD-A-018 F12·进程内 per-thread 读改写互斥锁
# ---------------------------------------------------------------------------
# 变式题组的结构化端点都是「aget_state → compute → aupdate_state」无原子保护：双击「全部入库」
# / 连点 edit-dna 等并发请求会读到同一旧快照、各自算 update、后写覆盖前写 → 重复落库 / 丢更新。
# 解法 = 同一 thread_id 的读改写整段串行（不同 thread 互不阻塞，并发不退化）。
# 进程内锁（dict[thread_id]→asyncio.Lock）：单实例够用；多实例分布式锁列 A-020（按需 Redis）。
# 配合库端 sourceHash 幂等键（variant_support.build_create_bo）= 双保险。
import asyncio as _asyncio  # noqa: E402

_THREAD_LOCKS: dict[str, _asyncio.Lock] = {}
_THREAD_LOCKS_GUARD = _asyncio.Lock()


async def _get_thread_lock(thread_id: str) -> _asyncio.Lock:
    """取/建某 thread_id 的进程内读改写锁（同 thread 串行；建锁本身用 guard 锁防竞态建两把）。"""
    lock = _THREAD_LOCKS.get(thread_id)
    if lock is None:
        async with _THREAD_LOCKS_GUARD:
            lock = _THREAD_LOCKS.get(thread_id)
            if lock is None:
                lock = _asyncio.Lock()
                _THREAD_LOCKS[thread_id] = lock
    return lock


def custom_generate_unique_id(route: APIRoute) -> str:
    """Generate idiomatic operation IDs for OpenAPI client generation."""
    return route.name


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Configurable lifespan that initializes the appropriate database checkpointer, store,
    and agents with async loading - for example for starting up MCP clients.
    """
    try:
        # Initialize both checkpointer (for short-term memory) and store (for long-term memory)
        async with initialize_database() as saver, initialize_store() as store:
            # Set up both components
            if hasattr(saver, "setup"):  # ignore: union-attr
                await saver.setup()
            # Only setup store for Postgres as InMemoryStore doesn't need setup
            if hasattr(store, "setup"):  # ignore: union-attr
                await store.setup()

            # Configure agents with both memory components and async loading
            agents = get_all_agent_info()
            for a in agents:
                try:
                    await load_agent(a.key)
                    logger.info(f"Agent loaded: {a.key}")
                except Exception as e:
                    logger.error(f"Failed to load agent {a.key}: {e}")
                    # Continue with other agents rather than failing startup

                agent = get_agent(a.key)
                # Set checkpointer for thread-scoped memory (conversation history)
                agent.checkpointer = saver
                # Set store for long-term memory (cross-conversation knowledge)
                agent.store = store
            yield
    except Exception as e:
        logger.error(f"Error during database/store/agents initialization: {e}")
        raise


app = FastAPI(lifespan=lifespan, generate_unique_id_function=custom_generate_unique_id)
router = APIRouter(dependencies=[Depends(verify_bearer)])


@router.get("/info")
async def info() -> ServiceMetadata:
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )


async def _handle_input(user_input: UserInput, agent: AgentGraph) -> tuple[dict[str, Any], UUID]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.
    """
    run_id = uuid7()
    thread_id = user_input.thread_id or str(uuid4())
    user_id = user_input.user_id or str(uuid4())

    configurable = {"thread_id": thread_id, "user_id": user_id}
    if user_input.model is not None:
        configurable["model"] = user_input.model

    callbacks: list[Any] = []
    if settings.LANGFUSE_TRACING:
        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        langfuse_handler = CallbackHandler()

        callbacks.append(langfuse_handler)

    if user_input.agent_config:
        # Check for reserved keys (including 'model' even if not in configurable)
        reserved_keys = {"thread_id", "user_id", "model"}
        if overlap := reserved_keys & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config)

    # 🔴 BUG-09（2026-06-19）·变式手动验算产品默认：真实请求若未显式指定 auto_verify，则注入
    #   False = 手动（生成秒到就绪、每题 pending，老师按需点 /variant/verify-one）。FE 想恢复自动
    #   则在 agent_config 里传 auto_verify=true。其它 agent 不读该键、无副作用。
    #   （直调节点的单测不走本入口 → 节点级 _auto_verify_on 缺省 True，既有验算测试不受影响。）
    configurable.setdefault("auto_verify", False)

    config = RunnableConfig(
        configurable=configurable,
        run_id=run_id,
        callbacks=callbacks,
    )

    # Check for interrupts that need to be resumed
    state = await agent.aget_state(config=config)
    interrupted_tasks = [
        task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
    ]

    input: Command | dict[str, Any]
    if interrupted_tasks:
        # assume user input is response to resume agent execution from interrupt
        input = Command(resume=user_input.message)
    else:
        input = {"messages": [HumanMessage(content=user_input.message)]}

    kwargs = {
        "input": input,
        "config": config,
    }

    return kwargs, run_id


@router.post("/{agent_id}/invoke", operation_id="invoke_with_agent_id")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.
    """
    # NOTE: Currently this only returns the last message or interrupt.
    # In the case of an agent outputting multiple AIMessages (such as the background step
    # in interrupt-agent, or a tool step in research-assistant), it's omitted. Arguably,
    # you'd want to include it. You could update the API to return a list of ChatMessages
    # in that case.
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent)

    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
        response_type, response = response_events[-1]
        if response_type == "values":
            # Normal response, the agent completed successfully
            output = langchain_to_chat_message(response["messages"][-1])
        elif response_type == "updates" and "__interrupt__" in response:
            # The last thing to occur was an interrupt
            # Return the value of the first interrupt as an AIMessage
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent)

    # 🔴 PRD-A-021 S0·F13 双通道并发互斥：通道A（/stream astream，graph 内）与通道B
    #   （/variant/* aget_state→aupdate_state，graph 外）此前对同一 thread 无任何互斥——并发写
    #   同一 checkpointer 时后写覆盖前写（丢 figure_url/手改/重复落库）。这里把 astream 这一**单轮**
    #   工作也纳入 per-thread 锁（复用 A-018 F12 的 _get_thread_lock，不另起一把），与所有通道B
    #   写端点串行化。锁只覆盖「本轮 astream」这段有界工作（generate 等节点自带 VARIANT_TIMEOUT
    #   墙钟闸），**绝不跨人类等待**（多轮交互每轮一个独立 /stream 请求；本请求结束即放锁）。
    thread_id = (
        ((kwargs.get("config") or {}).get("configurable") or {}).get("thread_id")
    )
    lock = await _get_thread_lock(thread_id) if thread_id else None
    # 🔴 「忙」即明确返回处理中信号，不静默排队几十秒：同 thread 已有在跑的轮次（通道A 另一请求
    #   或通道B 写端点持锁）→ 立即吐 processing 帧 + [DONE]，FE 据此禁用「发送/编辑」按钮、提示稍候，
    #   而非让本请求在锁上空等（体验=点了没反应）。不同 thread 各持各锁、互不阻塞。
    if lock is not None and lock.locked():
        yield (
            "data: "
            + json.dumps(
                {
                    "type": "processing",
                    "content": "这组题正在处理中（生成/编辑/入库未完成），请稍候再发指令。",
                }
            )
            + "\n\n"
        )
        yield "data: [DONE]\n\n"
        return

    acquired = False
    try:
        if lock is not None:
            await lock.acquire()
            acquired = True
        # Process streamed events from the graph and yield messages over the SSE stream.
        async for stream_event in agent.astream(
            **kwargs, stream_mode=["updates", "messages", "custom"], subgraphs=True
        ):
            if not isinstance(stream_event, tuple):
                continue
            # Handle different stream event structures based on subgraphs
            if len(stream_event) == 3:
                # With subgraphs=True: (node_path, stream_mode, event)
                _, stream_mode, event = stream_event
            else:
                # Without subgraphs: (stream_mode, event)
                stream_mode, event = stream_event
            new_messages = []
            if stream_mode == "updates":
                for node, updates in event.items():
                    # A simple approach to handle agent interrupts.
                    # In a more sophisticated implementation, we could add
                    # some structured ChatMessage type to return the interrupt value.
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue
                    updates = updates or {}
                    update_messages = updates.get("messages", [])
                    # special cases for using langgraph-supervisor library
                    if "supervisor" in node or "sub-agent" in node:
                        # the only tools that come from the actual agent are the handoff and handback tools
                        if isinstance(update_messages[-1], ToolMessage):
                            if "sub-agent" in node and len(update_messages) > 1:
                                # If this is a sub-agent, we want to keep the last 2 messages - the handback tool, and it's result
                                update_messages = update_messages[-2:]
                            else:
                                # If this is a supervisor, we want to keep the last message only - the handoff result. The tool comes from the 'agent' node.
                                update_messages = [update_messages[-1]]
                        else:
                            update_messages = []
                    new_messages.extend(update_messages)

            if stream_mode == "custom":
                new_messages = [event]

            # LangGraph streaming may emit tuples: (field_name, field_value)
            # e.g. ('content', <str>), ('tool_calls', [ToolCall,...]), ('additional_kwargs', {...}), etc.
            # We accumulate only supported fields into `parts` and skip unsupported metadata.
            # More info at: https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/
            processed_messages = []
            current_message: dict[str, Any] = {}
            for message in new_messages:
                if isinstance(message, tuple):
                    key, value = message
                    # Store parts in temporary dict
                    current_message[key] = value
                else:
                    # Add complete message if we have one in progress
                    if current_message:
                        processed_messages.append(_create_ai_message(current_message))
                        current_message = {}
                    processed_messages.append(message)

            # Add any remaining message parts
            if current_message:
                processed_messages.append(_create_ai_message(current_message))

            for message in processed_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                    continue
                # LangGraph re-sends the input message, which feels weird, so drop it
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            if stream_mode == "messages":
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                if "skip_stream" in metadata.get("tags", []):
                    continue
                # For some reason, astream("messages") causes non-LLM nodes to send extra messages.
                # Drop them.
                if not isinstance(msg, AIMessageChunk):
                    continue
                content = remove_tool_calls(msg.content)
                if content:
                    # Empty content in the context of OpenAI usually means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content.
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
    except Exception as e:
        # 🔴 PRD-C-100 B3·异常兜住（绝不裸 500）：变式图（generate opus 对压轴几何母题偶发慢吐
        #   reasoning 不收尾 → relay 池逐站墙钟超时耗尽 → RuntimeError 冒出 astream）等任何节点
        #   异常，旧实现一律吐笼统 'Internal server error'（FE 阶段灯只能显示「已中断·Internal
        #   server error」，老师不知发生了什么）。这里把**真实原因短摘**透出（墙钟超时/全站失败
        #   等），让 FE 渲染可读的有界 warn；HTTP 早已 200（SSE 已开流），此处只补一条 error 帧 +
        #   收尾 [DONE]，已先期 eager 上屏的变式卡保留不丢。绝不再发生「无限卡死」或「裸 500」。
        logger.error(f"Error in message generator: {e}")
        reason = _stream_error_reason(e)
        yield f"data: {json.dumps({'type': 'error', 'content': reason})}\n\n"
    finally:
        # 🔴 S0：本轮 astream 结束（正常/异常/客户端断流）即放锁——锁只随单个 /stream 请求生命周期，
        #   绝不跨请求/跨人类等待。只在本协程真正 acquire 到时才释放（busy 早退路径不进此处）。
        if lock is not None and acquired:
            lock.release()
        yield "data: [DONE]\n\n"


def _stream_error_reason(e: Exception) -> str:
    """SSE error 帧文案：把底层异常翻成老师能看懂的短句（B3·异常兜住）。
    relay 池全站墙钟超时/失败 → 「生成超时，请重试或换更简单的题」；其余 → 通用重试提示。
    绝不外泄堆栈/站名/密钥；保持有界（FE 阶段灯直接显示这句）。"""
    msg = str(e or "")
    low = msg.lower()
    if "wallclock" in low or "timeout" in low or "timed out" in low:
        return "生成超时了（模型对这道题响应过慢）。请重试，或换一道更简单/更清晰的题。"
    if "no relay available" in low or "blank relay" in low or "truncated" in low:
        return "AI 出口暂时不稳定，本轮没能生成完整变式。请稍后重试。"
    return "本轮生成中断了，请重试（如多次失败可换种说法或换题）。"


def _create_ai_message(parts: dict) -> AIMessage:
    sig = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered = {k: v for k, v in parts.items() if k in valid_keys}
    return AIMessage(**filtered)


def _sse_response_example() -> dict[int | str, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
    operation_id="stream_with_agent_id",
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()


@router.post("/history")
async def history(input: ChatHistoryInput) -> ChatHistory:
    """
    Get chat history.
    """
    # TODO: Hard-coding DEFAULT_AGENT here is wonky
    agent: AgentGraph = get_agent(DEFAULT_AGENT)
    try:
        state_snapshot = await agent.aget_state(
            config=RunnableConfig(configurable={"thread_id": input.thread_id})
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


class VariantPersistInput(BaseModel):
    """全部入库直连请求（2026-06-11 用户拍板：确定性动作不过 LLM 分类器）。"""

    thread_id: str
    ruoyi_token: str  # 必填：入库 owner = 登录老师本人（与图入口身份硬闸同源）


@router.post("/variant/persist")
async def variant_persist(input: VariantPersistInput) -> dict[str, Any]:
    """C 线扩展：「全部入库」直连——绕过 LLM 意图分类器，按 thread_id 从 checkpointer
    取当前题组 → 直调 persist_to_bank 节点函数（同一段代码：簿记/防重/血缘逻辑零分叉）
    → 回写 state（persisted 标记 + 回执消息进对话历史）→ 返回回执文本 + 最新 artifact。

    与 chat 通道说「入库」语义完全等价，只是省一次分类器 LLM 调用 + 零误判。
    """
    from agents.variant import _artifact_payload, persist_to_bank

    if conv_trace.teacher_id_from_token(input.ruoyi_token) is None:
        raise HTTPException(status_code=401, detail="登录态缺失或已过期，请重新登录")
    agent: AgentGraph = get_agent("variant")
    cfg = RunnableConfig(
        configurable={"thread_id": input.thread_id, "ruoyi_token": input.ruoyi_token}
    )
    # 🔴 F12·最常踩：双击「全部入库」并发 → per-thread 锁串行化读改写，第二次进来读到已写回的
    #   persisted 标记 → pending 为空 → 不重复落库。配合库端 sourceHash 幂等键双保险。
    lock = await _get_thread_lock(input.thread_id)
    try:
        async with lock:
            snapshot = await agent.aget_state(config=cfg)
            values: dict[str, Any] = snapshot.values or {}
            update = await persist_to_bank(values, cfg)  # type: ignore[arg-type]
            # 回写 checkpointer（as_node=persist_to_bank：簿记/回执与 chat 通道入库完全一致，
            # 后续编辑轮 assemble 快照「已收录」不回退、二次入库不重复落行）
            await agent.aupdate_state(cfg, update, as_node="persist_to_bank")
            merged = {**values, **{k: v for k, v in update.items() if k != "messages"}}
            msgs = update.get("messages") or []
            reply = str(msgs[-1].content) if msgs else ""
            # 🔴 B4 入库 → 写偏好记忆（确定性 D10）：常教年级册 + 常出题型。best-effort。
            _gb = ((merged.get("analysis") or {}).get("grade") or {}).get("value")
            _qt = ((merged.get("mother_dna") or {}).get("dna") or {}).get("qtype")
            await _mem_write_preference(input.ruoyi_token, grade_book=_gb, qtype=_qt)
            return {"ok": True, "reply": reply, "artifact": _artifact_payload(merged)}  # type: ignore[arg-type]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"variant_persist error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


class VariantFigureInput(BaseModel):
    """PRD-C-100 B3 带图管线 · service 层后处理（D14：不进 StateGraph）。
    mode=crop_mother（母题切图，需 image_url）/ compose_variant（变式造图，需 stem；带
    correction_prompt = 图片重生）。"""

    mode: str
    thread_id: str
    ruoyi_token: str
    image_url: str | None = None
    stem: str | None = None
    answer: str | None = None
    correction_prompt: str | None = None
    # 🔴 PRD-C-100 C·图片重生带上下文：上一版 GeoGebra commands（FE 存住每图上一版透传）。
    #   correction_prompt + prev_commands 都在 → compose 走增量修改分支（在上一版基础上改，不从零重画）。
    prev_commands: list[str] | None = None
    # 🔴 PRD-C-100 B3-配图：兼容 int/str（FE 传 item.seq/item.index 为 number）。
    #   此前死锁 str → pydantic v2 在请求校验层拒 int = 422（走不到 handler 内的降级兜底，
    #   配图全挡）。coerce 成 str 落地（compose 仅用作 stem 后缀 + 回显，str 安全）。
    item_id: int | str | None = None

    @field_validator("item_id", mode="before")
    @classmethod
    def _coerce_item_id(cls, v: Any) -> str | None:
        return None if v is None else str(v)


async def _exp_write(token: str, **kw: Any) -> None:
    """PRD-C-100 B4 经验层留痕 best-effort（图修正/改DNA 每次都写，只累计不消费 D13）。
    走 RuoYi HTTP（不直连 MySQL）；端点未上线/故障静默吞，绝不影响主流程。"""
    try:
        from agents.variant_support import RuoyiClient
        c = RuoyiClient(token=token)
        try:
            await c.write_dna_edit_log(**kw)
        finally:
            await c.aclose()
    except Exception as e:  # noqa: BLE001
        logger.debug(f"_exp_write best-effort skip: {e}")


async def _mem_write_preference(token: str, *, grade_book: str | None, qtype: str | None) -> None:
    """PRD-C-100 B4 入库 → 写偏好记忆 best-effort（确定性直存 D10）。"""
    try:
        from agents import teacher_memory as TM
        from agents.variant_support import RuoyiClient
        c = RuoyiClient(token=token)
        try:
            await TM.write_preference_on_persist(c, grade_book=grade_book, qtype=qtype)
        finally:
            await c.aclose()
    except Exception as e:  # noqa: BLE001
        logger.debug(f"_mem_write_preference best-effort skip: {e}")


def _spec_nonempty(spec: Any) -> Any:
    """figure_spec 取值归一（round4 半结构化）：spec 可为 str（旧·自然语言）或 dict
       （新·{"layout":..,"angle_labels":[..]}）。非空则原样返回（dict 不 str 化，结构无损下传
       compose），空串/空对象/其它 → None。"""
    if isinstance(spec, str):
        return spec.strip() or None
    if isinstance(spec, dict):
        return spec if (spec.get("layout") or spec.get("angle_labels")) else None
    return None


async def _lookup_figure_spec(thread_id: str, item_id: str | None) -> Any:
    """PRD-A-018 治本A：按 thread_id + item_id 从 checkpointer state 取该题的 figure_spec
       （出题节点产的「画什么」配图决策）。round4 起返回值可为 str（旧）或 dict（新半结构化）。

    🔴 item_id = FE 传的 item.seq || item.index（见 book-ui onComposeVariantFigure）。state 里每道题
       的有效 seq = it["_seq"]（generate eager 落的稳定生成序）或 index+1（缺省回退，与 _artifact_payload
       同口径）。先按 seq 匹配，命中即取 figure_spec；未命中再按 1-based index 兜底。
    🔴 best-effort：thread 无 state / 无 items / 取不到一律返回 None（compose 退回从 stem 现推，
       绝不因取 spec 失败而挡造图）。
    """
    if not item_id:
        return None
    try:
        agent: AgentGraph = get_agent("variant")
        snapshot = await agent.aget_state(
            config=RunnableConfig(configurable={"thread_id": thread_id})
        )
        items = (snapshot.values or {}).get("items") or []
        if not items:
            return None
        target = str(item_id).strip()
        # 按有效 seq 匹配（it["_seq"] 或 index+1）
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            eff_seq = it.get("_seq")
            eff_seq = str(eff_seq) if eff_seq not in (None, "") else str(i + 1)
            if eff_seq == target:
                return _spec_nonempty(it.get("figure_spec"))
        # seq 未命中 → 按 1-based index 兜底
        try:
            idx = int(target) - 1
        except (TypeError, ValueError):
            return None
        if 0 <= idx < len(items) and isinstance(items[idx], dict):
            return _spec_nonempty(items[idx].get("figure_spec"))
        return None
    except Exception as e:  # noqa: BLE001 — 取 spec 是增强不是关卡，失败降级 None
        logger.debug(f"_lookup_figure_spec best-effort skip: {e}")
        return None


async def _lookup_chapter_kp(thread_id: str) -> tuple[str | None, str | None]:
    """PRD-A-021 R3b·定型闸：从 checkpointer state 取母题章节名 + 主考点名（喂 compose 章节×图型定型闸）。

    源（复用 variant 既有取值口径，零伪造）：
      - 章节名 = variant._mother_chapter_name(state)（confirmed_chapter_name → analysis.chapter.value）。
      - 考点名 = analysis.kp.value（与 _mother_facts.kp_name 同源）。
    🔴 best-effort：thread 无 state / 取不到一律返回 (None, None) → compose 走逃生（不约束、自由翻命令）。
    """
    try:
        from agents.variant import _mother_chapter_name
        agent: AgentGraph = get_agent("variant")
        snapshot = await agent.aget_state(
            config=RunnableConfig(configurable={"thread_id": thread_id})
        )
        values = snapshot.values or {}
        chapter = _mother_chapter_name(values)
        kp = None
        analysis = values.get("analysis") or {}
        kp_obj = analysis.get("kp") or {}
        if isinstance(kp_obj, dict):
            kp = str(kp_obj.get("value") or "").strip() or None
        return chapter, kp
    except Exception as e:  # noqa: BLE001 — 定型闸取值是增强非关卡，失败降级逃生
        logger.debug(f"_lookup_chapter_kp best-effort skip: {e}")
        return None, None


@router.post("/variant/compose-figure")
async def variant_compose_figure(input: VariantFigureInput) -> dict[str, Any]:
    """PRD-C-100 B3：带图管线后处理端点（不进变式 StateGraph，四节点字节不动 D14）。
      - mode=crop_mother：figure-crop 检测+裁母题图 → PNG base64（母题图直贴）。
      - mode=compose_variant：opus 翻 GeoGebra 命令一轮直出 → mathfig 渲染 → PNG base64
        （correction_prompt 非空 = 图片重生，人在回路 D12）。
    🔴 OSS 不在此（人在回路省 spam）：入库时 FE 传 OSS（uploadMotherImage 既有）→ A-015 image 块。
    🔴 失败 → ok=False + needs_figure=True（降级，200 不 500，前端按 needs_figure ⚠ 外显 G11）。
    """
    from langchain_core.runnables.config import var_child_runnable_config

    from agents.figure import compose
    from agents.variant import _ainvoke_text, _parse_json

    if conv_trace.teacher_id_from_token(input.ruoyi_token) is None:
        raise HTTPException(status_code=401, detail="登录态缺失或已过期，请重新登录")
    # 设 config contextvar，让 compose 内 _ainvoke_text 的 ensure_config() 拿到 thread_id/teacher_id
    # （落 conv_trace label=figure_geogebra，造图翻命令 opus 调用不漏计 G6）。
    cfg: dict[str, Any] = {
        "configurable": {"thread_id": input.thread_id, "ruoyi_token": input.ruoyi_token}
    }
    ctok = var_child_runnable_config.set(cfg)  # type: ignore[arg-type]
    try:
        if input.mode == "crop_mother":
            if not input.image_url:
                raise HTTPException(status_code=400, detail="crop_mother 需 image_url")
            return await compose.crop_mother_figure(input.image_url)
        if input.mode == "compose_variant":
            if not input.stem:
                raise HTTPException(status_code=400, detail="compose_variant 需 stem")
            # 🔴 PRD-A-018 治本A：从 state 自取该题 figure_spec（出题节点产的「画什么」自然语言描述）
            #   传给 compose 作为权威画图依据（compose 照 spec 翻命令，不再逆推构型）。BE 自取最稳——
            #   FE 不必传；老线程/无 spec → 取到 None → compose 退回从 stem 现推（向后兼容）。
            #   任何取 state 失败一律降级 None（绝不因为取不到 spec 而挡住造图）。
            figure_spec = await _lookup_figure_spec(input.thread_id, input.item_id)
            # 🔴 PRD-A-021 R3b·章节×图型定型闸：BE 自取母题章节名 + 考点名传 compose（FE 不必传）。
            #   compose 据此查 biz_chapter_figure_map 取允许图型集约束 opus；取不到/无映射 → 逃生（不约束）。
            _chapter, _kp = await _lookup_chapter_kp(input.thread_id)
            result = await compose.compose_variant_figure(
                stem=input.stem, answer=input.answer, invoke=_ainvoke_text,
                parse_json=_parse_json, correction_prompt=input.correction_prompt,
                prev_commands=input.prev_commands,  # 🔴 PRD-C-100 C：图片重生带上一版命令 → 增量改图
                item_id=input.item_id, model=settings.VARIANT_MODEL_FIGURE,
                figure_spec=figure_spec,  # 🔴 PRD-A-018 治本A：出题产的配图自然语言描述（权威画什么）
                chapter=_chapter, kp=_kp,  # 🔴 PRD-A-021 R3b：章节×图型定型闸入参（add/regen/库内母题主路径）
            )
            # 🔴 B4 经验层留痕（图修正）：老师发修正提示词 → 每次都写（只累计 D13）。best-effort。
            if input.correction_prompt:
                await _exp_write(input.ruoyi_token, edit_kind="图修正",
                                 target_id=input.item_id, correction_prompt=input.correction_prompt)
            return result
        raise HTTPException(status_code=400, detail=f"未知 mode: {input.mode}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"variant_compose_figure error: {e}")
        # 降级：不 500，返回 needs_figure（G11 不卡流程）
        return {"ok": False, "needs_figure": True, "reason": f"造图异常: {str(e)[:80]}"}
    finally:
        var_child_runnable_config.reset(ctok)


class VariantPersistOneInput(BaseModel):
    """单题入库直连请求（PRD-C-014 B2·T5·B3 前置）：index=1-based。"""

    thread_id: str
    index: int
    ruoyi_token: str  # 必填：入库 owner = 登录老师本人（与「全部入库」同源身份硬闸）


@router.post("/variant/persist-one")
async def variant_persist_one(input: VariantPersistOneInput) -> dict[str, Any]:
    """C 线扩展：「单题入库」直连——按 thread_id 取题组 → 调 persist_one_to_bank（复用
    persist_items 单 item 路径，item 级 persisted 防重：已收录的重复调直接回已有 id，不二次落行）
    → 回写 state（persisted 标记 + 母题血缘 id）→ 返回 {ok, id, artifact}。

    与 /variant/persist 同范式（确定性动作不过 LLM 分类器）；index 越界 → 400；
    单题落库失败 → 200 带 ok=False（不当 500，前端按 ok 提示）。
    """
    from agents.variant import _artifact_payload, persist_one_to_bank

    if conv_trace.teacher_id_from_token(input.ruoyi_token) is None:
        raise HTTPException(status_code=401, detail="登录态缺失或已过期，请重新登录")
    agent: AgentGraph = get_agent("variant")
    cfg = RunnableConfig(
        configurable={"thread_id": input.thread_id, "ruoyi_token": input.ruoyi_token}
    )
    # 🔴 F12：per-thread 锁（连点「单题入库」并发去重；与全部入库共用同一把 thread 锁互斥）。
    lock = await _get_thread_lock(input.thread_id)
    try:
        async with lock:
            snapshot = await agent.aget_state(config=cfg)
            values: dict[str, Any] = snapshot.values or {}
            update, result, error = await persist_one_to_bank(
                values, input.index, token=input.ruoyi_token
            )
            if error:
                raise HTTPException(status_code=400, detail=error)
            # 有 state 变更（成功落库/回填血缘）才回写 checkpointer（防重簿记不丢）。
            if update:
                await agent.aupdate_state(cfg, update, as_node="persist_to_bank")
            merged = {**values, **update}
            return {
                "ok": bool(result.get("ok")),
                "id": result.get("id"),
                "skipped": bool(result.get("skipped")),
                "error": result.get("error"),
                "artifact": _artifact_payload(merged),  # type: ignore[arg-type]
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"variant_persist_one error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


@router.post("/variant/artifact")
async def variant_artifact(input: ChatHistoryInput) -> dict[str, Any]:
    """C 线扩展（2026-06-11 会话持久化）：按 thread_id 从 checkpointer 重建右栏
    artifact 快照（与流内 custom_data.artifact 同契约）。会话恢复时前端先回放
    /history 的气泡，再用本端点重建卡片栅。无题组返回空 items（前端显空态）。
    """
    agent: AgentGraph = get_agent("variant")
    try:
        state_snapshot = await agent.aget_state(
            config=RunnableConfig(configurable={"thread_id": input.thread_id})
        )
        values: dict[str, Any] = state_snapshot.values or {}
        from agents.variant import _artifact_payload

        # 🔴 2026-06-21（用户终审「刷新就没了」）：原来 items 空就早返回空 header → 有母题卡但**还没
        #   生成变式**（用户没点「开始举一反三」）的会话，刷新后右栏母题卡丢失。改为始终走
        #   _artifact_payload —— 它 header.mother_card = _build_mother_card(state)（sticky，items 空时
        #   out_items=[] 也正常；无 mother_dna 时返 None，FE 兼容显空态）。让母题卡随会话恢复。
        return _artifact_payload(values)  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"variant_artifact error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


# ---------------------------------------------------------------------------
# 题组编辑器三端点（PRD-C-009 二期）：reorder / edit-item / reverify。
# 同 /variant/persist 直连范式：aget_state → 调 variant.py 纯逻辑 → aupdate_state(as_node)
# → 返回 {ok, artifact:<_artifact_payload>}。题组是 toolkit 会话状态、编辑不落库 →
# 无需 ruoyi_token（"全部入库"仍走 /variant/persist 带 token 那条）。FE 刷新走 /variant/artifact 重建。
# ---------------------------------------------------------------------------
class VariantReorderInput(BaseModel):
    """题组重排请求：order = 1-based 全排列（长度=当前题数，每号恰一次）。"""

    thread_id: str
    order: list[int]


class VariantEditItemInput(BaseModel):
    """单题手动编辑请求：index=1-based；只 patch 传入字段（None=不动）。"""

    thread_id: str
    index: int
    stem: str | None = None
    answer: str | None = None
    solution: str | None = None


class VariantReverifyInput(BaseModel):
    """单题重跑闸B 请求：index=1-based。"""

    thread_id: str
    index: int


async def _variant_apply(thread_id: str, fn) -> dict[str, Any]:
    """题组编辑器三端点共用：取 state → fn(state) 算 update + 错误 → 回写 → 组帧返回。

    fn(values) 同步/异步均可，返回 (update, error)：error 非空 → 400；否则
    aupdate_state(as_node='__editor__') 回写 checkpointer → 返回最新 _artifact_payload。
    """
    from agents.variant import _artifact_payload

    agent: AgentGraph = get_agent("variant")
    cfg = RunnableConfig(configurable={"thread_id": thread_id})
    # 🔴 F12：per-thread 锁包住「读改写」整段（同 thread 串行，防连点 edit-dna 等丢更新）。
    lock = await _get_thread_lock(thread_id)
    try:
        async with lock:
            snapshot = await agent.aget_state(config=cfg)
            values: dict[str, Any] = snapshot.values or {}
            result = fn(values)
            if inspect.isawaitable(result):
                result = await result
            update, error = result
            if error:
                raise HTTPException(status_code=400, detail=error)
            # 回写 checkpointer（as_node 任取一个图内节点名：编辑只覆盖 items/manual_order，
            # 后续轮 assemble/persist 读到的就是编辑后的题组；不触发图继续跑）
            await agent.aupdate_state(cfg, update, as_node="exec_reorder")
            # 🔴 PRD-A-021 R4·F10：与 /variant/persist 的本地 merged 口径统一——messages 在图里
            #   是 add-reducer（append）通道，本地 {**values, **update} 若 update 带 messages 会把
            #   整段历史**整体覆盖**成本次的几条（_artifact_payload 据此算就读到被截断的快照）。
            #   故和 persist:538 一样剔除 messages 再 merge（aupdate_state 那侧由 LangGraph 走
            #   reducer append，已正确，不受影响）。当前编辑 fn 不返 messages = latent，统一防回归。
            merged = {**values, **{k: v for k, v in update.items() if k != "messages"}}
            return {"ok": True, "artifact": _artifact_payload(merged)}  # type: ignore[arg-type]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"_variant_apply error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


@router.post("/variant/reorder")
async def variant_reorder(input: VariantReorderInput) -> dict[str, Any]:
    """题组重排（零 LLM）：按 1-based 全排列 order 纯代码挪槽位 + seq 重编。

    非法 order（给不全/重复/越界）→ 400。簿记字段（check/gene/persisted/_seq）随题搬位、
    不错位（与 graph 节点 exec_reorder 共用 _reorder_items 纯函数）。
    """
    from agents.variant import reorder_items_state

    return await _variant_apply(
        input.thread_id, lambda values: reorder_items_state(values, input.order)
    )


@router.post("/variant/edit-item")
async def variant_edit_item(input: VariantEditItemInput) -> dict[str, Any]:
    """单题手动编辑（零 LLM）：只 patch 传入字段 → 净化富文本 → 标手动编辑 + check 置 manual。

    index 越界 → 400。manual_edited/from_edit 内部键不入库（白名单挡）；artifact 透传 tier='manual'。
    """
    from agents.variant import edit_item_state

    def _fn(values):
        update, _item, error = edit_item_state(
            values, input.index, stem=input.stem, answer=input.answer, solution=input.solution
        )
        return update, error

    return await _variant_apply(input.thread_id, _fn)


@router.post("/variant/reverify")
async def variant_reverify(input: VariantReverifyInput) -> dict[str, Any]:
    """单题重跑闸B（on-demand，单题 LLM+sympy）：复用 _check_one_item 判决路径，更新该题徽章。

    index 越界 → 400。判决仍只读 sympy verdict（铁律不破）；跑完 tier 变真实验算结果，
    洗掉 manual 的"待验算"语义。
    """
    from agents.variant import reverify_item_state

    async def _fn(values):
        update, _item, error = await reverify_item_state(values, input.index)
        return update, error

    return await _variant_apply(input.thread_id, _fn)


class VariantVerifyOneInput(BaseModel):
    """🔴 BUG-09（2026-06-19）：无状态单题程序验算请求（手动验算后端核心）。

    stem=题面、answer=题面标答（必填）；qtype/options 可选（结构分流用，缺省按解答类）。
    无 thread_id/index（无状态，不读会话）；service 层 bearer 已鉴权（router dependency）。
    """

    stem: str
    answer: str
    qtype: str | None = None
    options: Any = None


@router.post("/variant/verify-one")
async def variant_verify_one(input: VariantVerifyOneInput) -> dict[str, Any]:
    """🔴 BUG-09：无状态单题程序验算（同步返回，非 SSE）。

    复用闸B 判决路径（_solve_one + _machine_verify，纯 sympy；判决只读 verdict，铁律不破）。
    返回 {"verdict": "pass"|"fail"|"degrade", "detail": str, "computed": str|None}：
      pass=标答自洽 / fail=标答错（computed=真算值）/ degrade=sympy 吃不下→转人工（非判错）。
    永不 500（异常按 degrade 收口）。前端「待验算」徽章点验算 → 调本端点 → 据 verdict 刷状态。
    """
    from langchain_core.runnables.config import var_child_runnable_config

    from agents.variant import verify_one_stem

    # 设 config contextvar（让验算链内 _ainvoke_text 的 ensure_config() 拿到归属；无身份不报错）。
    cfg: dict[str, Any] = {"configurable": {"thread_id": "verify-one"}}
    ctok = var_child_runnable_config.set(cfg)  # type: ignore[arg-type]
    try:
        return await verify_one_stem(
            input.stem, input.answer, qtype=input.qtype, options=input.options
        )
    except Exception as e:  # noqa: BLE001 — verify_one_stem 本应自兜，这里纯保险（不 500）
        logger.error(f"variant_verify_one error: {e}")
        return {"verdict": "degrade", "detail": f"验算异常: {str(e)[:80]}", "computed": None}
    finally:
        var_child_runnable_config.reset(ctok)


class VariantSetFigureUrlInput(BaseModel):
    """PRD-C-100 BC2 + PRD-A-021 R2b·U1：变式配图回写请求（零 LLM）。index=1-based。

    两类配图态（可单传或一并传）：
      · figure_url：FE 入库时 uploadMotherImage 传 OSS 拿 https url → 入库进 A-015 image 块。
      · figure_base64：🔴 R2b·U1 —— compose 产的生成态 PNG base64，老师认账后**立即**回写 state
        （不等入库）→ 落 checkpoint，刷新/切 tab 从 state 取回（治生成态配图只在 FE 内存即丢）。
    撤图 = 两者都传 None/空。figure_base64 形参缺省（旧 FE 只传 url）→ 不碰 base64（向后兼容）。
    """

    thread_id: str
    index: int
    figure_url: str | None = None
    figure_base64: str | None = None


@router.post("/variant/set-figure-url")
async def variant_set_figure_url(input: VariantSetFigureUrlInput) -> dict[str, Any]:
    """变式配图回写（零 LLM）：把 figure_url（https OSS）/ figure_base64（生成态 PNG）存进
    state.items[i]，落 checkpoint（走 items merge reducer，刷新不丢）。

    index 越界 → 400；figure_url 非 https → 400。回写后入库时 build_create_bo 据 figure_url 产
    image 块（仅 base64 无 url 时 persist 侧 server-side 上 OSS 兜底）。与编辑器三端点同直连范式。
    """
    from agents.variant import set_item_figure_state

    def _fn(values):
        update, _item, error = set_item_figure_state(
            values, input.index, input.figure_url, figure_base64=input.figure_base64
        )
        return update, error

    return await _variant_apply(input.thread_id, _fn)


class VariantMarkManualBlockInput(BaseModel):
    """PRD-C-100 BC3：标/清「老师手动排版过」印记（零 LLM）。index=1-based。

    老师对已入库变式点「手动排版」→ FE 跳 A-015 网格编辑器存 blockJson → 回会话调本端点标印记
    （edited=True）。确认重生时先调 edited=False 清印记，再走既有 regen。
    """

    thread_id: str
    index: int
    edited: bool = True


@router.post("/variant/mark-manual-block")
async def variant_mark_manual_block(input: VariantMarkManualBlockInput) -> dict[str, Any]:
    """标/清「老师手动排版过」印记（零 LLM）：标 manual_block + manual_edited + from_edit（edited=True）
    或清 manual_block（edited=False）。这俩内部键不入库（白名单挡）；artifact 透传 manual_block + question_id 给 FE。

    index 越界 → 400。与编辑器端点同直连范式（aget_state → 纯逻辑 → aupdate_state → _artifact_payload）。
    """
    from agents.variant import mark_item_manual_block_state

    def _fn(values):
        update, _item, error = mark_item_manual_block_state(values, input.index, input.edited)
        return update, error

    return await _variant_apply(input.thread_id, _fn)


# ---------------------------------------------------------------------------
# DNA 双模态编辑两端点（PRD-C-014 B4·T1/T2，契约 PRD §10.5 / §3.5）。
# 同上直连范式（aget_state → variant.py 纯逻辑 → aupdate_state(as_node) → _artifact_payload）。
# edit-dna = 零 LLM 结构化回写（G11）；revise = 有界 LLM 锚定重做（G12 diff 锁 / whole 走闸B sympy）。
# ---------------------------------------------------------------------------
class VariantEditDnaInput(BaseModel):
    """单题 DNA 维度结构化回写请求（零 LLM）：index=1-based；field=维度键；value=新值。

    field ∈ {main_kp, secondary_kps, qtype, exam_type, difficulty, tags, scene, grade}。
    value 形态随 field：main_kp/secondary_kps = code 串 或 {code,name}（secondary 为数组）；
    difficulty = 1..4 整数；tags = 字符串数组；其余 = 字符串。
    """

    thread_id: str
    index: int
    field: str
    value: Any = None
    # 🔴 PRD-C-100 B4：经验层留痕需 teacher_id（从 token 解）。可选（不传则跳过留痕，编辑本身不需登录态）。
    ruoyi_token: str | None = None


class VariantReviseInput(BaseModel):
    """单题有界 LLM 锚定重做请求：index=1-based；target ∈ skeleton/scene/whole；instruction=老师指令。

    whole 走 REGEN + 闸B sympy 重验 → 入库 owner 用 ruoyi_token（与 persist 同身份硬闸，可选）。
    """

    thread_id: str
    index: int
    target: str
    instruction: str = ""
    ruoyi_token: str | None = None


@router.post("/variant/edit-dna")
async def variant_edit_dna(input: VariantEditDnaInput) -> dict[str, Any]:
    """单题 DNA 维度结构化回写（🔴 零 LLM·G11）：校验合法值（题型/考察类型闭集、难度 1-4、
    副 kp≤3）→ 非法 400 拒收 → 回写 state 对应键（main_kp/grade 同步 header+BO，其余 DNA 维改
    mother_dna.dna，qtype/difficulty 改 item）→ 标该题 manual_edited → 返回最新 artifact。
    """
    from agents.variant import edit_dna_state

    def _fn(values):
        update, _item, error = edit_dna_state(
            values, input.index, input.field, input.value
        )
        return update, error

    out = await _variant_apply(input.thread_id, _fn)
    # 🔴 B4 经验层留痕（改DNA维）：每次改维都写（只累计 D13）。best-effort（成功才写）。
    if out.get("ok") and getattr(input, "ruoyi_token", None):
        await _exp_write(input.ruoyi_token, edit_kind="dna维", dim=str(input.field),
                         after=str(input.value), target_id=input.thread_id)
    return out


@router.post("/variant/revise")
async def variant_revise(input: VariantReviseInput) -> dict[str, Any]:
    """单题有界 LLM 锚定重做：target=skeleton/scene → 只重写该文本维（diff 锁 target·G12，
    不动其余维）；target=whole → REGEN 整题重出 + 闸B sympy 重验（判决只读 verdict）。
    标 manual；LLM 失败/解析不出 → 降级回 {ok:false,error}（200），不崩。index/target 非法 → 400。
    """
    from agents.variant import _artifact_payload, revise_item

    agent: AgentGraph = get_agent("variant")
    cfg = RunnableConfig(
        configurable={"thread_id": input.thread_id, "ruoyi_token": input.ruoyi_token}
    )
    # 🔴 F12：per-thread 锁（revise 改 items → 与并发编辑/入库串行，防丢更新）。
    lock = await _get_thread_lock(input.thread_id)
    try:
        async with lock:
            snapshot = await agent.aget_state(config=cfg)
            values: dict[str, Any] = snapshot.values or {}
            update, result, error = await revise_item(
                values, input.index, input.target, input.instruction, cfg
            )
            if error:
                raise HTTPException(status_code=400, detail=error)
            if update:
                await agent.aupdate_state(cfg, update, as_node="exec_regenerate")
            merged = {**values, **update}
            return {
                "ok": bool(result.get("ok")),
                "error": result.get("error"),
                "artifact": _artifact_payload(merged),  # type: ignore[arg-type]
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"variant_revise error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


# ---------------------------------------------------------------------------
# PRD-C-015 批4·DNA 改→重生两端点：regen（手动重生待重生集合）/ undo-regen（撤销重生）。
# 同直连范式（aget_state → variant.py 纯逻辑 → aupdate_state(as_node) → _artifact_payload）。
# 🔴 重生走有界 LLM（_regen_once/重写解析）+ 闸B sympy 重验（判决只读 verdict，铁律不破）；
#    撤销重生零 LLM（回快照）。题组是会话态、重生不落库 → 无需 ruoyi_token。
# ---------------------------------------------------------------------------
class VariantRegenInput(BaseModel):
    """手动「重生」请求：indexes=可选 1-based 题号子集（None/空=全待重生集合）。"""

    thread_id: str
    indexes: list[int] | None = None


class VariantUndoRegenInput(BaseModel):
    """「撤销重生」请求：index=1-based，回该题上一版重生前快照。"""

    thread_id: str
    index: int


@router.post("/variant/regen")
async def variant_regen(input: VariantRegenInput) -> dict[str, Any]:
    """手动「重生」（D-merge6/8 + 缺口12）：对待重生集合（dna_dirty 题）一次性重出/重写解析。

    软重生维脏→_regen_once 整题重出+闸B；仅重写解析维脏→只重写 solution+闸B 重验；保留手改
    （manual 维不被母题基准覆盖）；重生前存快照（撤销用）；重生后清 dirty。无 dirty → 空操作。
    返回 {ok, regenerated:[idx], failed:[{idx,error}], artifact}。
    """
    from agents.variant import _artifact_payload, regen_dirty_items

    agent: AgentGraph = get_agent("variant")
    cfg = RunnableConfig(configurable={"thread_id": input.thread_id})
    # 🔴 F12：per-thread 锁（重生改 items → 与并发编辑/入库串行，防丢更新/重复落库）。
    lock = await _get_thread_lock(input.thread_id)
    try:
        async with lock:
            snapshot = await agent.aget_state(config=cfg)
            values: dict[str, Any] = snapshot.values or {}
            update, result, error = await regen_dirty_items(values, input.indexes)
            if error:
                raise HTTPException(status_code=400, detail=error)
            if update:
                await agent.aupdate_state(cfg, update, as_node="exec_regenerate")
            merged = {**values, **update}
            return {
                "ok": True,
                "regenerated": result.get("regenerated") or [],
                "failed": result.get("failed") or [],
                "artifact": _artifact_payload(merged),  # type: ignore[arg-type]
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"variant_regen error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


@router.post("/variant/undo-regen")
async def variant_undo_regen(input: VariantUndoRegenInput) -> dict[str, Any]:
    """「撤销重生」（缺口12，零 LLM）：第 index 道回上一版重生前快照。无快照 → 400。"""
    from agents.variant import undo_regen_item

    def _fn(values):
        update, _item, error = undo_regen_item(values, input.index)
        return update, error

    return await _variant_apply(input.thread_id, _fn)


@app.get("/health")
async def health_check():
    """Health check endpoint."""

    health_status = {"status": "ok"}

    if settings.LANGFUSE_TRACING:
        try:
            langfuse = Langfuse()
            health_status["langfuse"] = "connected" if langfuse.auth_check() else "disconnected"
        except Exception as e:
            logger.error(f"Langfuse connection error: {e}")
            health_status["langfuse"] = "disconnected"

    return health_status


app.include_router(router)
