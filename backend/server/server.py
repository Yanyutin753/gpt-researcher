import json
import os
from typing import Dict, List, Optional, Any, Union
import time
import asyncio
from datetime import datetime

from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
    File,
    UploadFile,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.server.websocket_manager import WebSocketManager
from backend.server.server_utils import (
    get_config_dict,
    sanitize_filename,
    update_environment_variables,
    handle_file_upload,
    handle_file_deletion,
    execute_multi_agents,
    handle_websocket_communication,
)

from backend.server.websocket_manager import run_agent
from backend.utils import write_md_to_word, write_md_to_pdf
from gpt_researcher.utils.logging_config import setup_research_logging
from gpt_researcher.utils.enum import Tone, ReportType
from backend.chat.chat import ChatAgentWithMemory

import logging

# Get logger instance
logger = logging.getLogger(__name__)

# Don't override parent logger settings
logger.propagate = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Only log to console
)

# Models


class ResearchRequest(BaseModel):
    task: str
    report_type: str
    report_source: str
    tone: str
    headers: dict | None = None
    repo_name: str
    branch_name: str
    generate_in_background: bool = True


class ConfigRequest(BaseModel):
    ANTHROPIC_API_KEY: str
    TAVILY_API_KEY: str  # Supports comma-separated multiple keys
    LANGCHAIN_TRACING_V2: str
    LANGCHAIN_API_KEY: str
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str = ""
    DOC_PATH: str
    RETRIEVER: str
    GOOGLE_API_KEY: str = ""
    GOOGLE_CX_KEY: str = ""
    BING_API_KEY: str = ""
    SEARCHAPI_API_KEY: str = ""
    SERPAPI_API_KEY: str = ""
    SERPER_API_KEY: str = ""
    SEARX_URL: str = ""
    XAI_API_KEY: str
    DEEPSEEK_API_KEY: str


# Chat Completions API Models
class ChatMessage(BaseModel):
    role: str = Field(
        ...,
        description="The role of the message author: 'system', 'user', or 'assistant'",
    )
    content: str = Field(..., description="The content of the message")
    name: Optional[str] = Field(None, description="Optional name of the author")


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(
        ..., description="List of messages in the conversation"
    )
    model: Optional[str] = Field(
        "gpt-researcher", description="Model to use for completion"
    )
    temperature: Optional[float] = Field(
        0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        None, ge=1, description="Maximum tokens to generate"
    )
    stream: Optional[bool] = Field(
        False, description="Whether to stream partial responses"
    )

    # GPT-Researcher specific parameters
    report_type: Optional[str] = Field(
        "research_report", description="Type of research report"
    )
    report_source: Optional[str] = Field("web", description="Source for research data")
    tone: Optional[str] = Field("Objective", description="Tone of the report")
    query_domains: Optional[List[str]] = Field(
        [], description="Specific domains to search"
    )
    source_urls: Optional[List[str]] = Field([], description="Additional source URLs")
    document_urls: Optional[List[str]] = Field(
        [], description="Document URLs to include"
    )
    mcp_enabled: Optional[bool] = Field(False, description="Enable MCP functionality")
    mcp_strategy: Optional[str] = Field("fast", description="MCP strategy to use")
    mcp_configs: Optional[List[Any]] = Field(
        [], description="MCP configuration options"
    )


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Dict[str, int]] = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


# App initialization
app = FastAPI()

# Static files and templates
app.mount("/site", StaticFiles(directory="./frontend"), name="site")
app.mount("/static", StaticFiles(directory="./frontend/static"), name="static")
templates = Jinja2Templates(directory="./frontend")

# WebSocket manager
manager = WebSocketManager()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DOC_PATH = os.getenv("DOC_PATH", "./my-docs")

# Startup event


@app.on_event("startup")
def startup_event():
    os.makedirs("outputs", exist_ok=True)
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
    # os.makedirs(DOC_PATH, exist_ok=True)  # Commented out to avoid creating the folder if not needed


# Routes


def map_model_to_report_type(model: Optional[str], provided_report_type: Optional[str]) -> str:
    """Map model name to a report type. Model hints take precedence.

    Examples of model naming that influence report type:
    - "*-multi*" -> "multi_agents"
    - "*-detailed*" -> detailed_report
    - "*-outline*" -> outline_report
    - "*-resource*" -> resource_report
    - "*-custom*" -> custom_report
    - "*-subtopic*" -> subtopic_report
    - "*-deep*" -> deep
    Otherwise falls back to provided_report_type or research_report.
    """
    try:
        model_normalized = (model or "").lower()

        if "multi" in model_normalized:
            return "multi_agents"
        if "detailed" in model_normalized:
            return ReportType.DetailedReport.value
        if "outline" in model_normalized:
            return ReportType.OutlineReport.value
        if "resource" in model_normalized:
            return ReportType.ResourceReport.value
        if "custom" in model_normalized:
            return ReportType.CustomReport.value
        if "subtopic" in model_normalized:
            return ReportType.SubtopicReport.value
        if "deep" in model_normalized:
            return ReportType.DeepResearch.value

        # Fallback to provided
        return provided_report_type or ReportType.ResearchReport.value
    except Exception:
        # Defensive fallback
        return provided_report_type or ReportType.ResearchReport.value


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "report": None}
    )


@app.get("/report/{research_id}")
async def read_report(request: Request, research_id: str):
    docx_path = os.path.join("outputs", f"{research_id}.docx")
    if not os.path.exists(docx_path):
        return {"message": "Report not found."}
    return FileResponse(docx_path)


async def write_report(research_request: ResearchRequest, research_id: str = None):
    report_information = await run_agent(
        task=research_request.task,
        report_type=research_request.report_type,
        report_source=research_request.report_source,
        source_urls=[],
        document_urls=[],
        tone=Tone[research_request.tone],
        websocket=None,
        stream_output=None,
        headers=research_request.headers,
        query_domains=[],
        config_path="",
        return_researcher=True,
    )

    docx_path = await write_md_to_word(report_information[0], research_id)
    pdf_path = await write_md_to_pdf(report_information[0], research_id)
    if research_request.report_type != "multi_agents":
        report, researcher = report_information
        response = {
            "research_id": research_id,
            "research_information": {
                "source_urls": researcher.get_source_urls(),
                "research_costs": researcher.get_costs(),
                "visited_urls": list(researcher.visited_urls),
                "research_images": researcher.get_research_images(),
                # "research_sources": researcher.get_research_sources(),  # Raw content of sources may be very large
            },
            "report": report,
            "docx_path": docx_path,
            "pdf_path": pdf_path,
        }
    else:
        response = {
            "research_id": research_id,
            "report": "",
            "docx_path": docx_path,
            "pdf_path": pdf_path,
        }

    return response


@app.post("/report/")
async def generate_report(
    research_request: ResearchRequest, background_tasks: BackgroundTasks
):
    research_id = sanitize_filename(f"task_{int(time.time())}_{research_request.task}")

    if research_request.generate_in_background:
        background_tasks.add_task(
            write_report, research_request=research_request, research_id=research_id
        )
        return {
            "message": "Your report is being generated in the background. Please check back later.",
            "research_id": research_id,
        }
    else:
        response = await write_report(research_request, research_id)
        return response


@app.get("/files/")
async def list_files():
    if not os.path.exists(DOC_PATH):
        os.makedirs(DOC_PATH, exist_ok=True)
    files = os.listdir(DOC_PATH)
    print(f"Files in {DOC_PATH}: {files}")
    return {"files": files}


@app.post("/api/multi_agents")
async def run_multi_agents():
    return await execute_multi_agents(manager)


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    return await handle_file_upload(file, DOC_PATH)


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    return await handle_file_deletion(filename, DOC_PATH)


# Chat Completions API streaming implementation


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible Chat Completions API endpoint.
    Supports both streaming and non-streaming responses.
    """
    try:
        completion_id = f"chatcmpl-{int(time.time())}{hash(str(request.messages))}"

        # Extract research task from the last user message
        task = ""
        for message in reversed(request.messages):
            if message.role == "user":
                task = message.content
                break

        if not task:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "No user message found",
                        "type": "invalid_request",
                    }
                },
            )

        # Handle streaming response
        if request.stream:
            # Simple streaming generator that handles everything
            async def generate_stream():
                completion_id_local = completion_id
                model_local = request.model
                created = int(time.time())

                try:
                    # Send initial chunk to establish connection
                    initial_chunk = ChatCompletionStreamResponse(
                        id=completion_id_local,
                        created=created,
                        model=model_local,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0,
                                delta={"reasoning_content": "🔍 开始研究任务...\n", "content": ""},
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {initial_chunk.model_dump_json()}\n\n"

                    # Create a shared message queue for real-time streaming
                    stream_queue = asyncio.Queue()
                    research_completed = False

                    class StreamCollector:
                        def __init__(self, queue):
                            self.queue = queue

                        async def send_json(self, data):
                            message_type = data.get("type", "")
                            content = data.get("output", "") or data.get("content", "")
                            report_content = ""
                            reasoning_content = ""

                            logger.info(
                                f"📨 收到消息: type={message_type}, content_len={len(content) if content else 0}"
                            )

                            if content:
                                match message_type:
                                    case "report":
                                        report_content = f"{content}"
                                    case "logs":
                                        reasoning_content = f"{content}"
                                    case "images":
                                        for i, image in enumerate(json.loads(content)):
                                            print(f"![{i+1}]({image})\n")
                                    case _:
                                        report_content = f"{content}"

                                chunk = ChatCompletionStreamResponse(
                                    id=completion_id_local,
                                    created=created,
                                    model=model_local,
                                    choices=[
                                        ChatCompletionStreamChoice(
                                            index=0,
                                            delta={"content": report_content, "reasoning_content": reasoning_content},
                                            finish_reason=None,
                                        )
                                    ],
                                )
                                chunk_data = f"data: {chunk.model_dump_json()}\n\n"

                                # Use put_nowait to avoid blocking
                                try:
                                    self.queue.put_nowait(chunk_data)
                                except asyncio.QueueFull:
                                    logger.warning("⚠️ 队列已满，跳过消息")
                                except Exception as e:
                                    logger.error(f"❌ 队列操作错误: {e}")

                    collector = StreamCollector(stream_queue)

                    # Start research task
                    async def run_research():
                        try:
                            logger.info(f"🚀 开始研究任务: {task}")
                            chosen_report_type = map_model_to_report_type(
                                request.model, request.report_type
                            )
                            final_report = await run_agent(
                                task=task,
                                report_type=chosen_report_type,
                                report_source=request.report_source,
                                source_urls=request.source_urls or [],
                                document_urls=request.document_urls or [],
                                tone=Tone[request.tone],
                                websocket=collector,
                                headers=None,
                                query_domains=request.query_domains or [],
                                config_path="",
                                mcp_enabled=request.mcp_enabled,
                                mcp_strategy=request.mcp_strategy,
                                mcp_configs=request.mcp_configs or [],
                            )

                            logger.info(
                                f"✅ 研究任务完成: {len(str(final_report))} 字符"
                            )

                            # Send completion signal
                            completion_chunk = ChatCompletionStreamResponse(
                                id=completion_id_local,
                                created=created,
                                model=model_local,
                                choices=[
                                    ChatCompletionStreamChoice(
                                        index=0, delta={}, finish_reason="stop"
                                    )
                                ],
                            )
                            stream_queue.put_nowait(
                                f"data: {completion_chunk.model_dump_json()}\n\n"
                            )
                            stream_queue.put_nowait("data: [DONE]\n\n")

                        except Exception as e:
                            logger.error(f"❌ 研究任务错误: {e}")
                            error_chunk = ChatCompletionStreamResponse(
                                id=completion_id_local,
                                created=created,
                                model=model_local,
                                choices=[
                                    ChatCompletionStreamChoice(
                                        index=0,
                                        delta={"content": f"\n\n❌ 错误: {str(e)}"},
                                        finish_reason="error",
                                    )
                                ],
                            )
                            stream_queue.put_nowait(
                                f"data: {error_chunk.model_dump_json()}\n\n"
                            )
                            stream_queue.put_nowait("data: [DONE]\n\n")

                        finally:
                            # Signal completion to stream
                            stream_queue.put_nowait(None)

                    # Start the research task
                    research_task = asyncio.create_task(run_research())

                    # Stream messages as they come
                    timeout_count = 0
                    max_timeouts = 20  # Allow up to 20 timeouts (100 seconds)

                    while True:
                        try:
                            # Wait for message with shorter timeout for more responsive streaming
                            chunk = await asyncio.wait_for(
                                stream_queue.get(), timeout=2.0
                            )
                            if chunk is None:  # End signal
                                logger.info("🏁 收到结束信号，流式传输完成")
                                break
                            yield chunk
                            timeout_count = (
                                0  # Reset timeout counter on successful data
                            )

                        except asyncio.TimeoutError:
                            timeout_count += 1
                            logger.debug(f"⏰ 流式超时 #{timeout_count}")

                            # Send keep-alive only occasionally to reduce noise
                            if timeout_count % 5 == 0:
                                yield ": keep-alive\n\n"

                            # If too many timeouts, check if research task is still alive
                            if timeout_count >= max_timeouts:
                                if research_task.done():
                                    logger.warning("🔚 研究任务已完成但未收到结束信号")
                                    break
                                else:
                                    logger.warning("⚠️ 研究任务可能卡住，继续等待...")
                                    timeout_count = 0  # Reset and continue waiting

                        except Exception as e:
                            logger.error(f"❌ 流处理错误: {e}")
                            break

                except Exception as e:
                    logger.error(f"❌ 流式处理错误: {e}")
                    error_chunk = ChatCompletionStreamResponse(
                        id=completion_id_local,
                        created=created,
                        model=model_local,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0,
                                delta={"content": f"\n\n❌ 错误: {str(e)}"},
                                finish_reason="error",
                            )
                        ],
                    )
                    yield f"data: {error_chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"

            # Return streaming response with correct media type
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering
                },
            )

        # Handle non-streaming response
        else:
            try:
                # Run research task
                chosen_report_type = map_model_to_report_type(
                    request.model, request.report_type
                )
                report = await run_agent(
                    task=task,
                    report_type=chosen_report_type,
                    report_source=request.report_source,
                    source_urls=request.source_urls or [],
                    document_urls=request.document_urls or [],
                    tone=Tone[request.tone],
                    websocket=None,
                    headers=None,
                    query_domains=request.query_domains or [],
                    config_path="",
                    mcp_enabled=request.mcp_enabled,
                    mcp_strategy=request.mcp_strategy,
                    mcp_configs=request.mcp_configs or [],
                )

                # Create response
                response = ChatCompletionResponse(
                    id=completion_id,
                    created=int(time.time()),
                    model=request.model,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=str(report)),
                            finish_reason="stop",
                        )
                    ],
                    usage={
                        "prompt_tokens": sum(
                            len(m.content.split()) for m in request.messages
                        ),
                        "completion_tokens": len(str(report).split()),
                        "total_tokens": sum(
                            len(m.content.split()) for m in request.messages
                        )
                        + len(str(report).split()),
                    },
                )

                return response

            except Exception as e:
                logger.error(f"Error in non-streaming chat completion: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": str(e), "type": "server_error"}},
                )

    except Exception as e:
        logger.error(f"Error in chat completions endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error"}},
        )


@app.get("/v1/models")
async def models():
    """
    OpenAI-compatible Models API endpoint.
    Supports both streaming and non-streaming responses.
    """
    models_list = [
        {"id": "gpt-researcher", "object": "model"},
        {"id": "gpt-researcher-detailed", "object": "model"},
        {"id": "gpt-researcher-outline", "object": "model"},
        {"id": "gpt-researcher-resource", "object": "model"},
        {"id": "gpt-researcher-custom", "object": "model"},
        {"id": "gpt-researcher-subtopic", "object": "model"},
        {"id": "gpt-researcher-deep", "object": "model"},
        {"id": "gpt-researcher-multi", "object": "model"},
    ]
    return JSONResponse(status_code=200, content={"data": models_list})
    
    

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await handle_websocket_communication(websocket, manager)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
