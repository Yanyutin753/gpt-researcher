import json
import os
from typing import Dict, List, Optional, AsyncGenerator
import time
import uuid
import asyncio

from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
    File,
    UploadFile,
    BackgroundTasks,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

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
from gpt_researcher.agent import GPTResearcher
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

# Model to Report Type mapping
MODEL_REPORT_MAPPING = {
    # 研究报告模型
    "gpt-researcher": ReportType.ResearchReport.value,
    "gpt-researcher-3.5": ReportType.ResearchReport.value,
    "gpt-researcher-4": ReportType.ResearchReport.value,
    # 资源报告模型
    "gpt-resource": ReportType.ResourceReport.value,
    "gpt-resource-finder": ReportType.ResourceReport.value,
    # 大纲报告模型
    "gpt-outline": ReportType.OutlineReport.value,
    "gpt-outliner": ReportType.OutlineReport.value,
    # 详细报告模型
    "gpt-detailed": ReportType.DetailedReport.value,
    "gpt-comprehensive": ReportType.DetailedReport.value,
    # 子主题报告模型
    "gpt-subtopic": ReportType.SubtopicReport.value,
    "gpt-topic-explorer": ReportType.SubtopicReport.value,
    # 深度研究模型
    "gpt-deep": ReportType.DeepResearch.value,
    "gpt-deep-research": ReportType.DeepResearch.value,
    "gpt-deep-dive": ReportType.DeepResearch.value,
    # 自定义报告模型
    "gpt-custom": ReportType.CustomReport.value,
}

# 支持的模型列表
SUPPORTED_MODELS = list(MODEL_REPORT_MAPPING.keys())


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-researcher"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 16384
    stream: Optional[bool] = True
    report_source: Optional[str] = "web"
    source_urls: Optional[List[str]] = None
    query_domains: Optional[List[str]] = None
    concise: Optional[bool] = True  # 控制报告简洁性


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Dict[str, int]] = None


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
    TAVILY_API_KEY: str
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


class StreamHandler:
    """Handler for capturing research logs and progress"""

    def __init__(self):
        self.messages = asyncio.Queue()

    async def send(self, data: dict):
        """Capture messages from researcher"""
        await self.messages.put(data)

    async def receive_json(self):
        """Not used but required for compatibility"""
        return {}

    async def send_json(self, data: dict):
        """Capture JSON messages"""
        await self.messages.put(data)


async def stream_research_response(
    query: str,
    model: str,
    report_type: str,
    report_source: str,
    source_urls: Optional[List[str]],
    query_domains: Optional[List[str]],
    concise: bool = True,
) -> AsyncGenerator[str, None]:
    """Generate streaming response for research with WebSocket-compatible format"""

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    # Create a custom handler to capture logs
    stream_handler = StreamHandler()

    try:
        # Send initial OpenAI format message
        initial_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"

        # Send initial research message
        ws_msg = {
            "type": "logs",
            "content": "research_started",
            "output": "🧙‍♂️ Gathering information and analyzing your research topic...\n",
        }

        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": ws_msg["output"]}, "finish_reason": None}
            ],
            "websocket_message": ws_msg,
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        # Initialize researcher with stream handler
        researcher = GPTResearcher(
            query=query,
            report_type=report_type,
            report_source=report_source,
            source_urls=source_urls,
            query_domains=query_domains,
            websocket=stream_handler,
            verbose=True,
        )

        # Start research in background
        research_task = asyncio.create_task(researcher.conduct_research())

        # Track last message to avoid duplicates
        last_content = ""

        # Stream progress messages
        while not research_task.done():
            try:
                # Get message with timeout
                message = await asyncio.wait_for(stream_handler.messages.get(), timeout=0.5)

                if isinstance(message, dict):
                    msg_type = message.get("type", "logs")
                    content = message.get("content", "")
                    output = message.get("output", "")

                    # Skip duplicate messages - check both content and output
                    current_msg = f"{content}:{output}"
                    if current_msg == last_content:
                        continue
                    last_content = current_msg

                    # Use the output field if available, otherwise use content
                    # The output field contains the actual detailed message
                    message_to_format = output if output else content
                    
                    # Format output text based on the actual message
                    output_text = format_progress_message(message_to_format)

                    if output_text:
                        # Create WebSocket-compatible message
                        ws_msg = {"type": msg_type, "content": content, "output": output_text}

                        # Send as OpenAI chunk
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": output_text},
                                    "finish_reason": None,
                                }
                            ],
                            "websocket_message": ws_msg,
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Warning: Error processing message: {str(e)}")
                continue

        # Wait for research to complete
        await research_task

        # Generate final report
        report = await researcher.write_report()
        
        # Post-process report for conciseness if requested
        if concise:
            # Remove redundant paragraphs
            paragraphs = report.split('\n\n')
            unique_paragraphs = []
            seen_content = set()
            
            for para in paragraphs:
                # Create a simplified version for comparison (ignore minor differences)
                simplified = para.lower().strip()
                # Skip if too similar to already seen content
                if len(simplified) > 50:  # Only check longer paragraphs
                    is_duplicate = False
                    for seen in seen_content:
                        # Check similarity (simple approach)
                        if len(set(simplified.split()) & set(seen.split())) / len(set(simplified.split())) > 0.7:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_paragraphs.append(para)
                        seen_content.add(simplified)
                else:
                    unique_paragraphs.append(para)
            
            report = '\n\n'.join(unique_paragraphs)
        
        # Get images and embed them in the report
        try:
            images = researcher.get_research_images()
            print(f"Debug: Found {len(images) if images else 0} images")
            
            if images and len(images) > 0:
                # Add an images section at the end of the report
                image_section = "\n\n## 相关图片\n\n"
                images_added = 0
                
                for idx, img in enumerate(images[:6], 1):  # Increase limit to 6 images
                    if isinstance(img, dict) and 'url' in img:
                        img_url = img['url']
                        caption = img.get('description', '') or img.get('alt', '') or f'图片 {idx}'
                        
                        # Clean up caption
                        caption = caption.strip()
                        if not caption:
                            caption = f'相关图片 {idx}'
                        
                        image_section += f"![{caption}]({img_url})\n\n"
                        images_added += 1
                        print(f"Debug: Added image {idx}: {img_url[:50]}...")
                
                if images_added > 0:
                    # Append images to the report
                    report += image_section
                    print(f"Debug: Successfully added {images_added} images to report")
                else:
                    print("Debug: No valid images found to add")
            else:
                print("Debug: No images available from researcher")
        except Exception as e:
            print(f"Error adding images to report: {e}")
            import traceback
            traceback.print_exc()

        # Send report as WebSocket-style message
        ws_msg = {"type": "report", "content": "report", "output": report}

        # Stream report in chunks
        chunk_size = 2000  # Increased chunk size for better performance
        total_length = len(report)
        chunks_sent = 0
        
        for i in range(0, total_length, chunk_size):
            chunk_content = report[i : min(i + chunk_size, total_length)]
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"content": chunk_content}, "finish_reason": None}
                ],
                "websocket_message": {"type": "report", "content": "report_chunk", "output": chunk_content},
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            chunks_sent += 1
            # Remove sleep for faster streaming, only add minimal delay if needed
            if chunks_sent % 10 == 0:  # Only pause every 10 chunks
                await asyncio.sleep(0.001)
        
        print(f"Debug: Sent {chunks_sent} chunks, total {total_length} characters")
        
        # Ensure all data is sent before finishing
        await asyncio.sleep(0.05)  # Slightly longer delay to ensure flush

        # Send final chunk with stop signal
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"\n\n❌ Error: {str(e)}"},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


# 消息内容翻译映射
MESSAGE_TRANSLATIONS = {
    # 查询和源处理
    "fetching_query_content": "正在获取查询内容...",
    "added_source_url": "添加了新的数据源",
    "add_source": "添加数据源",
    "selecting_sources": "正在选择相关数据源...",
    
    # 研究阶段
    "researching": "正在进行研究...",
    "research_complete": "研究完成",
    "conducting_research": "正在执行研究任务...",
    
    # 搜索相关
    "searching": "正在搜索相关信息...",
    "search_complete": "搜索完成",
    "web_search": "正在进行网络搜索...",
    "searching_with_query": "正在使用查询进行搜索...",
    
    # 数据抓取
    "scraping_urls": "正在抓取网页内容...",
    "scraping_content": "正在提取页面内容...",
    "scraping_images": "正在获取相关图片...",
    "scraping_complete": "数据抓取完成",
    "scraping_web_sources": "正在抓取网络资源...",
    
    # 内容分析
    "analyzing": "正在分析内容...",
    "analyzing_content": "正在深度分析内容...",
    "processing_content": "正在处理内容数据...",
    "summarizing": "正在生成摘要...",
    
    # 报告生成
    "writing_report": "正在撰写研究报告...",
    "generating_report": "正在生成报告...",
    "formatting_report": "正在格式化报告...",
    "finalizing_report": "正在完成报告...",
    
    # 其他
    "thinking": "正在思考...",
    "processing": "正在处理...",
    "loading": "正在加载...",
    "completed": "已完成",
}

def translate_message(content: str) -> str:
    """翻译消息内容为用户友好的中文描述"""
    # 先尝试直接匹配
    if content in MESSAGE_TRANSLATIONS:
        return MESSAGE_TRANSLATIONS[content]
    
    # 尝试模糊匹配（忽略大小写）
    content_lower = content.lower()
    for key, value in MESSAGE_TRANSLATIONS.items():
        if key.lower() in content_lower or content_lower in key.lower():
            return value
    
    # 如果是URL或特殊格式，进行特殊处理
    if content.startswith("http"):
        return f"正在处理链接: {content[:50]}..."
    
    # 如果包含特定关键词，返回相应描述
    if "search" in content_lower:
        return "正在搜索..."
    elif "scrap" in content_lower:
        return "正在抓取数据..."
    elif "analyz" in content_lower or "analy" in content_lower:
        return "正在分析..."
    elif "writ" in content_lower or "generat" in content_lower:
        return "正在生成内容..."
    elif "add" in content_lower:
        return "正在添加资源..."
    elif "fetch" in content_lower:
        return "正在获取数据..."
    
    # 默认返回原内容（但限制长度）
    if len(content) > 50:
        return content[:50] + "..."
    return content


def format_progress_message(content: str) -> str:
    """Format progress messages for display with more detailed information"""
    if not content:
        return ""

    # Strip any leading/trailing whitespace
    content = content.strip()
    content_lower = content.lower()

    # Handle messages that already have emojis and good formatting
    if content.startswith("✅") or content.startswith("🔍") or content.startswith("📚") or content.startswith("🤔"):
        return content if content.endswith("\n") else f"{content}\n"
    
    # First check if it's a URL being added
    if content.startswith("http"):
        return f"📚 Added source: {content}\n"
    
    # Check for specific patterns that indicate actual content
    if "added source url" in content_lower:
        # The URL should already be in the message
        return content if content.endswith("\n") else f"{content}\n"
    
    # Check for query-related messages
    if "generated sub-queries" in content_lower or "sub-queries:" in content_lower:
        return content if content.endswith("\n") else f"{content}\n"
    
    if "running research for" in content_lower:
        # The message already contains the full information
        return content if content.endswith("\n") else f"{content}\n"
    
    # Check for scraping messages
    if "scraping" in content_lower:
        # Check if it's already formatted nicely
        if "🌐" in content or "📄" in content:
            return content if content.endswith("\n") else f"{content}\n"
        return f"📥 {content}\n"
    
    # Check for research context messages
    if "relevant content" in content_lower or "combined research" in content_lower:
        return content if content.endswith("\n") else f"{content}\n"
    
    # Check for research outline messages
    if "research outline" in content_lower or "planned:" in content_lower:
        return content if content.endswith("\n") else f"{content}\n"
    
    # Check for cost messages
    if "total research costs" in content_lower or "💸" in content:
        return content if content.endswith("\n") else f"{content}\n"
    
    # Map generic content types to display messages
    message_map = {
        "searching": "🔍 Searching for information...\n",
        "search": "🔍 Searching...\n",
        "scraping": "📥 Extracting content from sources...\n",
        "analyzing": "🧠 Analyzing data...\n",
        "writing": "✍️ Generating report...\n",
        "generating": "✍️ Writing content...\n",
        "fetching": "📋 Preparing research...\n",
        "added_source": "📚 Added new source\n",
        "context": "🔗 Building context...\n",
        "subquer": "📝 Creating sub-queries...\n",
        "processing": "⚙️ Processing...\n",
        "research_outline": "📋 Planning research outline...\n",
        "conduct": "🔬 Conducting research...\n",
        "crawl": "🕷️ Crawling web pages...\n",
        "scraped": "📄 Content extracted\n",
    }

    # Check for matches
    for key, message in message_map.items():
        if key in content_lower:
            # If we have more specific info, include it
            if ":" in content and not content.startswith("http"):
                parts = content.split(":", 1)
                if len(parts) > 1 and parts[1].strip():
                    return f"{message.strip()} - {parts[1].strip()}\n"
            return message

    # For any other message that contains useful information, show it
    if len(content) > 10 and not any(skip in content_lower for skip in ["agent_generated", "research_started", "logs"]):
        # Truncate very long messages
        if len(content) > 100:
            return f"• {content[:100]}...\n"
        return f"• {content}\n"
    
    # Default - only show if it seems meaningful
    if len(content) > 20:
        return f"• {content}\n"
    
    return ""


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


@app.get("/v1/models")
async def list_models():
    """
    List available models (OpenAI-compatible endpoint)
    """
    models_list = [
        {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "gpt-researcher",
            "description": f"Report type: {MODEL_REPORT_MAPPING[model_id]}",
        }
        for model_id in SUPPORTED_MODELS
    ]

    return {"object": "list", "data": models_list}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint with GPT Researcher integration

    Supported models:
    - gpt-researcher: Standard research report
    - gpt-resource: Resource finding report
    - gpt-outline: Outline report
    - gpt-detailed: Comprehensive detailed report
    - gpt-subtopic: Subtopic exploration report
    - gpt-deep: Deep research report
    - gpt-custom: Custom report format

    Supports streaming (stream=true) for real-time progress updates
    """
    try:
        # Validate model
        if request.model not in MODEL_REPORT_MAPPING:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is not supported. Available models: {', '.join(SUPPORTED_MODELS)}",
            )

        # Extract the last user message as the query
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        query = user_messages[-1].content

        # Get report type from model
        report_type = MODEL_REPORT_MAPPING[request.model]

        # Check if streaming is requested
        if request.stream:
            return StreamingResponse(
                stream_research_response(
                    query=query,
                    model=request.model,
                    report_type=report_type,
                    report_source=request.report_source,
                    source_urls=request.source_urls,
                    query_domains=request.query_domains,
                    concise=request.concise,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming response
        researcher = GPTResearcher(
            query=query,
            report_type=report_type,
            report_source=request.report_source,
            source_urls=request.source_urls,
            query_domains=request.query_domains,
            verbose=False,
        )

        # Conduct research
        await researcher.conduct_research()

        # Generate report
        report = await researcher.write_report()

        # Format response in OpenAI-compatible format
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=report),
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": len(query.split()),
                "completion_tokens": len(report.split()),
                "total_tokens": len(query.split()) + len(report.split()),
            },
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await handle_websocket_communication(websocket, manager)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)