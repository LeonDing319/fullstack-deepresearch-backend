# Copyright 2024
# Directory: yt-DeepResearch-Backend/main.py
"""
FastAPI Backend for Deep Research Agent
Provides streaming API endpoints for deep research with multiple AI model support
"""

import asyncio
from contextlib import suppress
import json
import logging
import math
import os
import time
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.deep_research_service import DeepResearchService
from services.model_service import ModelService
from models.research_models import (
    ResearchRequest,
    ResearchResponse,
    StreamingEvent,
    ModelComparison,
    ResearchHistory,
    ComparisonSession,
    ComparisonResult,
    StageTimings,
    ResearchStage
)
from utils.metrics import MetricsCollector


class MultiModelComparisonRequest(BaseModel):
    """Request model for multi-model comparison"""
    query: str = Field(..., description="Research question to test across models", min_length=1)
    models: List[str] = Field(..., description="List of model IDs to compare", min_items=1)
    api_keys: Dict[str, str] = Field(..., description="API keys for each model", min_items=1)

# Configure logging with Google Standards
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
DEBUG_LOG_PATH = "/Users/admin/Leon/VibeCoding/全栈Deep Reserch/.cursor/debug.log"


def _debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: Optional[Dict] = None,
    run_id: str = "pre-fix",
) -> None:
    """Write compact NDJSON debug logs for runtime evidence collection."""
    try:
        payload = {
            "id": f"log_{int(time.time() * 1000)}_{hypothesis_id}",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

# Initialize FastAPI app with comprehensive configuration
app = FastAPI(
    title="Deep Research Agent API",
    description="Streaming deep research API with multiple AI model support (Zhipu GLM-4.7, DeepSeek V3.2, Kimi K2)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
research_service = DeepResearchService()
model_service = ModelService()
metrics_collector = MetricsCollector()

# Registry of active research sessions for cancellation support
# Key: research_id, Value: asyncio.Event (set when cancellation is requested)
active_research_cancellations: Dict[str, asyncio.Event] = {}

@app.get("/")
@app.head("/")
async def root():
    """Health check endpoint - supports both GET and HEAD for Cloud Run health checks"""
    return {
        "message": "Deep Research Agent API is running",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
@app.head("/health")
async def health_check():
    """Detailed health check with service status - supports both GET and HEAD"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "research_service": "active",
            "model_service": "active",
            "metrics_collector": "active"
        },
        "supported_models": ["zhipu", "deepseek", "kimi"]
    }

@app.post("/research/stream")
async def stream_research(request: ResearchRequest):
    """
    Stream deep research process with real-time updates
    
    This endpoint provides Server-Sent Events streaming of the research process,
    showing each stage, thinking process, and tool usage in real-time.
    
    Args:
        request: Research request containing query, model, and API key
    
    Returns:
        StreamingResponse: Server-sent events stream
    """
    try:
        # Validate the request
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Research query cannot be empty")
        
        if not request.api_key.strip():
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Validate model selection
        available_models = await model_service.get_available_models()
        if request.model not in [model.id for model in available_models["models"]]:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported model: {request.model}. Available models: {[m.id for m in available_models['models']]}"
            )
        
        # Create streaming generator
        async def generate_research_stream() -> AsyncGenerator[str, None]:
            research_id = f"research_{int(time.time())}"
            start_time = time.time()
            last_emit_at = start_time
            heartbeat_interval_seconds = 20
            cancel_event = asyncio.Event()
            active_research_cancellations[research_id] = cancel_event

            try:
                # Initialize research session
                yield f"data: {json.dumps({'type': 'session_start', 'research_id': research_id, 'timestamp': datetime.utcnow().isoformat(), 'model': request.model, 'query': request.query})}\n\n"
                # region agent log
                _debug_log(
                    hypothesis_id="H4",
                    location="main.py:generate_research_stream:session_start",
                    message="backend_session_started",
                    data={"researchId": research_id, "model": request.model},
                )
                # endregion

                # Stream research via queue so heartbeat can be emitted during long model/tool calls
                event_queue: asyncio.Queue[Optional[StreamingEvent]] = asyncio.Queue()
                producer_error: Optional[Exception] = None
                collected_report_content: str = ""

                async def produce_research_events() -> None:
                    nonlocal producer_error
                    try:
                        async for produced_event in research_service.stream_research(
                            query=request.query,
                            model=request.model,
                            api_key=request.api_key,
                            research_id=research_id,
                            cancel_event=cancel_event,
                        ):
                            await event_queue.put(produced_event)
                    except Exception as producer_exception:
                        producer_error = producer_exception
                    finally:
                        await event_queue.put(None)

                producer_task = asyncio.create_task(produce_research_events())

                while True:
                    # Check cancellation between outbound events
                    if cancel_event.is_set():
                        # region agent log
                        _debug_log(
                            hypothesis_id="H2",
                            location="main.py:generate_research_stream:cancel_branch",
                            message="backend_cancel_event_detected",
                            data={"researchId": research_id},
                        )
                        # endregion
                        if not producer_task.done():
                            producer_task.cancel()
                            with suppress(asyncio.CancelledError):
                                await producer_task
                        logger.info(f"Research {research_id} cancelled by user")
                        yield f"data: {json.dumps({'type': 'cancelled', 'research_id': research_id, 'timestamp': datetime.utcnow().isoformat()})}\n\n"
                        return

                    try:
                        event = await asyncio.wait_for(
                            event_queue.get(),
                            timeout=heartbeat_interval_seconds
                        )
                    except asyncio.TimeoutError:
                        gap_since_emit_ms = int((time.time() - last_emit_at) * 1000)
                        heartbeat_event = {
                            "type": "heartbeat",
                            "research_id": research_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "content": "Research still running",
                            "metadata": {"gapSincePreviousEmitMs": gap_since_emit_ms}
                        }
                        # region agent log
                        _debug_log(
                            hypothesis_id="H3",
                            location="main.py:generate_research_stream:heartbeat_emit",
                            message="backend_heartbeat_emitted",
                            data={"researchId": research_id, "gapSincePreviousEmitMs": gap_since_emit_ms},
                        )
                        # endregion
                        yield f"data: {json.dumps(heartbeat_event)}\n\n"
                        last_emit_at = time.time()
                        continue

                    if event is None:
                        break

                    # Collect final report content (same pattern as Compare endpoint)
                    if event.content and event.stage == ResearchStage.FINAL_REPORT:
                        collected_report_content += event.content + "\n"

                    gap_since_emit_ms = int((time.time() - last_emit_at) * 1000)
                    if gap_since_emit_ms > 15000 or event.type in {"error", "timeout_warning", "cancelled"}:
                        # region agent log
                        _debug_log(
                            hypothesis_id="H3",
                            location="main.py:generate_research_stream:event_emit",
                            message="backend_event_emitted",
                            data={
                                "researchId": research_id,
                                "eventType": event.type,
                                "stage": str(event.stage) if event.stage else "",
                                "gapSincePreviousEmitMs": gap_since_emit_ms,
                            },
                        )
                        # endregion
                    yield f"data: {json.dumps(event.dict())}\n\n"
                    last_emit_at = time.time()

                with suppress(asyncio.CancelledError):
                    await producer_task

                if producer_error is not None:
                    raise producer_error

                # Calculate final metrics
                end_time = time.time()
                duration = end_time - start_time

                # Send completion event with collected report content
                completion_event = {
                    'type': 'research_complete',
                    'research_id': research_id,
                    'duration': duration,
                    'model': request.model,
                    'timestamp': datetime.utcnow().isoformat(),
                    'report_content': collected_report_content.strip() if collected_report_content else '',
                }
                yield f"data: {json.dumps(completion_event)}\n\n"

                # Store metrics for comparison
                await metrics_collector.store_research_metrics(
                    research_id=research_id,
                    model=request.model,
                    duration=duration,
                    query=request.query
                )

            except asyncio.CancelledError:
                logger.info(f"Research {research_id} cancelled (CancelledError)")
                yield f"data: {json.dumps({'type': 'cancelled', 'research_id': research_id, 'timestamp': datetime.utcnow().isoformat()})}\n\n"
            except Exception as e:
                logger.error(f"Error in research stream: {str(e)}")
                error_event = {
                    'type': 'error',
                    'message': str(e),
                    'research_id': research_id,
                    'timestamp': datetime.utcnow().isoformat()
                }
                yield f"data: {json.dumps(error_event)}\n\n"
            finally:
                active_research_cancellations.pop(research_id, None)
        
        return StreamingResponse(
            generate_research_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Keep-Alive": "timeout=300, max=100",  # 5 minute timeout
                "X-Content-Type-Options": "nosniff",
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting research stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research/{research_id}/cancel")
async def cancel_research(research_id: str):
    """Cancel an active research session"""
    cancel_event = active_research_cancellations.get(research_id)
    if cancel_event is None:
        raise HTTPException(status_code=404, detail="Research session not found or already completed")

    cancel_event.set()
    logger.info(f"Cancellation requested for research: {research_id}")

    return {
        "message": "Research cancellation requested",
        "research_id": research_id,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/models")
async def get_available_models():
    """Get list of available AI models with their capabilities"""
    return await model_service.get_available_models()

@app.get("/research/history")
async def get_research_history(limit: int = 10):
    """Get research history with performance metrics"""
    return await metrics_collector.get_research_history(limit=limit)

@app.get("/research/comparison")
async def get_model_comparison():
    """Get performance comparison between different models"""
    return await metrics_collector.get_model_comparison()

@app.delete("/research/history/{research_id}")
async def delete_research(research_id: str):
    """Delete a specific research session"""
    success = await metrics_collector.delete_research(research_id)
    if not success:
        raise HTTPException(status_code=404, detail="Research not found")
    return {"message": "Research deleted successfully"}

@app.post("/research/compare")
async def run_multi_model_comparison(request: MultiModelComparisonRequest):
    """
    Run the same research query across multiple models in parallel
    with real-time SSE streaming of per-model progress percentages
    """
    # Validate models before starting the stream
    available_models = await model_service.get_available_models()
    available_model_ids = [model.id for model in available_models["models"]]
    
    for model in request.models:
        if model not in available_model_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {model}. Available: {available_model_ids}"
            )
        if model not in request.api_keys or not request.api_keys[model].strip():
            raise HTTPException(
                status_code=400,
                detail=f"Missing API key for model: {model}"
            )
    
    session_id = f"compare_{int(time.time())}"
    
    # Stage floor: minimum progress % when reaching a given stage.
    # Actual progress is primarily driven by elapsed time (unique per model).
    STAGE_FLOOR: Dict[str, int] = {
        "initialization": 2, "clarification": 5, "research_brief": 10,
        "research_execution": 20, "research_planning": 25, "research_query": 35,
        "research_finding": 45, "research_analysis": 55, "research_synthesis": 65,
        "tool_usage": 30, "thinking": 30, "final_report": 75, "completed": 100,
    }
    STAGE_NAMES: Dict[str, str] = {
        "initialization": "Initializing...", "clarification": "Analyzing requirements...",
        "research_brief": "Creating research plan...", "research_execution": "Conducting research...",
        "research_planning": "Planning strategy...", "research_query": "Searching sources...",
        "research_finding": "Processing findings...", "research_analysis": "Analyzing data...",
        "research_synthesis": "Synthesizing results...", "tool_usage": "Using tools...",
        "thinking": "AI reasoning...", "final_report": "Generating report...",
        "completed": "Completed",
    }
    
    async def generate_comparison_stream() -> AsyncGenerator[str, None]:
      try:
        # SSE padding to force TCP/proxy buffer flush (standard SSE technique)
        yield ": stream opened\n\n"
        yield f": padding{' ' * 4096}\n\n"
        logger.info(f"[Compare SSE] Padding sent for session {session_id}")

        # Send session_start so frontend knows the stream is connected
        yield f"data: {json.dumps({'type': 'session_start', 'session_id': session_id, 'timestamp': datetime.utcnow().isoformat(), 'query': request.query, 'models': request.models})}\n\n"
        logger.info(f"[Compare SSE] session_start sent for session {session_id}")

        progress_queue: asyncio.Queue[Optional[Dict]] = asyncio.Queue()

        async def run_model_with_progress(model_id: str, model_index: int = 0) -> ComparisonResult:
            """Run a single model's research, pushing progress events to the shared queue."""
            start_time = time.time()
            stage_start_time = start_time
            current_stage = ResearchStage.CLARIFICATION
            stage_timings = StageTimings()
            sources_found = 0
            report_content = ""
            supervisor_tools: List[str] = []
            error_msg: Optional[str] = None
            chunk_count = 0
            last_progress = 2  # Start at 2% to show immediate feedback

            def update_stage_timing(stage: ResearchStage, duration: float):
                if stage == ResearchStage.CLARIFICATION:
                    stage_timings.clarification = duration
                elif stage == ResearchStage.RESEARCH_BRIEF:
                    stage_timings.research_brief = duration
                elif stage == ResearchStage.RESEARCH_EXECUTION:
                    stage_timings.research_execution = duration
                elif stage == ResearchStage.FINAL_REPORT:
                    stage_timings.final_report = duration

            await progress_queue.put({
                "type": "model_progress", "model": model_id,
                "progress": 2, "stage": "Connecting to model API...", "elapsed": 0,
            })

            try:
                async for event in research_service.stream_research(
                    query=request.query,
                    model=model_id,
                    api_key=request.api_keys[model_id],
                    research_id=f"{session_id}_{model_id}",
                    comparison_mode=True
                ):
                    chunk_count += 1
                    elapsed = time.time() - start_time

                    # Detect error/timeout events from the research service
                    if event.type in ("error", "timeout_warning"):
                        error_msg = event.content or event.error or "Research error"
                        logger.warning(f"Model {model_id} received {event.type}: {error_msg[:100]}")

                    # Track stage transitions
                    if event.stage and event.stage != current_stage:
                        stage_duration = time.time() - stage_start_time
                        update_stage_timing(current_stage, stage_duration)
                        current_stage = event.stage
                        stage_start_time = time.time()

                    # Calculate progress using elapsed time + stage floor + event count.
                    # Each model gets a slightly different curve coefficient so
                    # parallel models always show distinct progress percentages.
                    stage_key = current_stage.value if current_stage else "initialization"
                    stage_floor = STAGE_FLOOR.get(stage_key, 20)
                    # Per-model curve: different decay constants (55s, 60s, 65s)
                    # so models diverge naturally over time
                    decay = 55 + model_index * 5
                    time_pct = int(85 * (1 - math.exp(-elapsed / decay)))
                    # Small bonus for events processed (each event ≈ 0.3%)
                    event_bonus = min(10, int(chunk_count * 0.3))
                    progress = max(stage_floor, time_pct + event_bonus, last_progress)
                    last_progress = progress
                    stage_name = STAGE_NAMES.get(stage_key, "Processing...")

                    # Push progress event (send every event to keep UI responsive)
                    await progress_queue.put({
                        "type": "model_progress", "model": model_id,
                        "progress": progress, "stage": stage_name, "elapsed": elapsed,
                    })

                    # Track sources
                    if event.type == "sources_found" and event.metadata:
                        sources = event.metadata.get("sources", [])
                        if isinstance(sources, list):
                            sources_found += len(sources)

                    # Track supervisor tools
                    if event.type == "tool_usage" and event.metadata:
                        tool_name = event.metadata.get("tool_name")
                        if tool_name and tool_name not in supervisor_tools:
                            supervisor_tools.append(tool_name)

                    # Collect only final report content (not debug info / stage descriptions)
                    if event.content and event.stage == ResearchStage.FINAL_REPORT:
                        report_content += event.content + "\n"

                # Complete final stage
                final_stage_duration = time.time() - stage_start_time
                update_stage_timing(current_stage, final_stage_duration)
                # Only mark as successful if no error events were received
                success = error_msg is None

            except Exception as e:
                logger.error(f"Error in model {model_id} research: {str(e)}")
                error_msg = str(e)
                success = False
                final_stage_duration = time.time() - stage_start_time
                update_stage_timing(current_stage, final_stage_duration)

            total_duration = time.time() - start_time
            word_count = len(report_content.split()) if report_content else 0

            result = ComparisonResult(
                model=model_id,
                duration=total_duration,
                stage_timings=stage_timings,
                sources_found=sources_found,
                word_count=word_count,
                success=success,
                error=error_msg,
                report_content=report_content,
                supervisor_tools_used=supervisor_tools
            )

            # Push completion/error event with full result for immediate display
            completion_evt: Dict = {
                "type": "model_complete" if success else "model_error",
                "model": model_id,
                "progress": 100 if success else last_progress,
                "stage": "Completed" if success else (error_msg[:100] if error_msg else "Failed"),
                "elapsed": total_duration,
            }
            if success:
                completion_evt["result_summary"] = {
                    "word_count": word_count,
                    "sources_found": sources_found,
                    "duration": round(total_duration, 1),
                }
                # Include full result so frontend can show report immediately
                completion_evt["result"] = result.dict()
            await progress_queue.put(completion_evt)

            return result
        
        # Per-model timeout to prevent indefinite hanging (safety net)
        # Thinking models (Kimi K2) can take 3-5min per step; allow enough total time
        MODEL_TIMEOUT = 600  # 10 minutes per model maximum
        
        async def run_model_with_timeout(model_id: str, model_index: int = 0) -> ComparisonResult:
            """Wrap model research with a timeout safety net, staggering starts"""
            # Stagger model starts by 1.5s each to reduce API contention
            if model_index > 0:
                await asyncio.sleep(model_index * 1.5)
            try:
                return await asyncio.wait_for(
                    run_model_with_progress(model_id, model_index),
                    timeout=MODEL_TIMEOUT
                )
            except asyncio.TimeoutError:
                timeout_msg = f"Research timed out after {MODEL_TIMEOUT}s - model API may be overloaded"
                logger.error(f"Model {model_id}: {timeout_msg}")
                await progress_queue.put({
                    "type": "model_error", "model": model_id,
                    "progress": 0, "stage": timeout_msg,
                    "elapsed": float(MODEL_TIMEOUT),
                })
                return ComparisonResult(
                    model=model_id, duration=float(MODEL_TIMEOUT),
                    stage_timings=StageTimings(),
                    sources_found=0, word_count=0, success=False,
                    error=timeout_msg, report_content="",
                    supervisor_tools_used=[]
                )
        
        # Start all model tasks in parallel with timeout protection (staggered starts)
        tasks = [asyncio.create_task(run_model_with_timeout(m, i)) for i, m in enumerate(request.models)]
        
        async def watch_completion():
            await asyncio.gather(*tasks, return_exceptions=True)
            await progress_queue.put(None)  # Sentinel to stop the stream
        
        asyncio.create_task(watch_completion())
        
        # Emit events from queue with heartbeat support (short timeout to keep stream active)
        event_count = 0
        while True:
            try:
                event_data = await asyncio.wait_for(progress_queue.get(), timeout=5)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
                continue
            if event_data is None:
                logger.info(f"[Compare SSE] Sentinel received, exiting main loop after {event_count} events")
                break
            event_count += 1
            if event_count <= 5 or event_count % 20 == 0:
                logger.info(f"[Compare SSE] Emitting event #{event_count}: type={event_data.get('type')} model={event_data.get('model')}")
            yield f"data: {json.dumps(event_data)}\n\n"
        
        # Collect final results
        valid_results: List[ComparisonResult] = []
        for i, task in enumerate(tasks):
            try:
                result = task.result()
                if isinstance(result, Exception):
                    valid_results.append(ComparisonResult(
                        model=request.models[i], duration=0.0, stage_timings=StageTimings(),
                        sources_found=0, word_count=0, success=False, error=str(result),
                        report_content="", supervisor_tools_used=[]
                    ))
                else:
                    valid_results.append(result)
            except Exception as e:
                valid_results.append(ComparisonResult(
                    model=request.models[i], duration=0.0, stage_timings=StageTimings(),
                    sources_found=0, word_count=0, success=False, error=str(e),
                    report_content="", supervisor_tools_used=[]
                ))
        
        # Create and store comparison session
        comparison_session = ComparisonSession(
            session_id=session_id,
            query=request.query,
            timestamp=datetime.utcnow().isoformat(),
            results=valid_results
        )
        try:
            await asyncio.wait_for(
                metrics_collector.store_comparison_session(comparison_session),
                timeout=10
            )
        except Exception as store_err:
            logger.warning(f"[Compare SSE] Failed to store session (non-fatal): {store_err}")

        # Emit final session data
        yield f"data: {json.dumps({'type': 'session_complete', 'session': comparison_session.dict()})}\n\n"
        logger.info(f"[Compare SSE] session_complete sent for session {session_id}")

      except asyncio.CancelledError:
        logger.info(f"[Compare SSE] Stream cancelled for session {session_id}")
        yield f"data: {json.dumps({'type': 'error', 'message': 'Stream cancelled', 'session_id': session_id})}\n\n"
      except Exception as e:
        logger.error(f"[Compare SSE] Unhandled error in stream for session {session_id}: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'session_id': session_id, 'timestamp': datetime.utcnow().isoformat()})}\n\n"

    return StreamingResponse(
        generate_comparison_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",
            "Keep-Alive": "timeout=300, max=100",
            "X-Content-Type-Options": "nosniff",
        }
    )

@app.post("/research/test")
async def test_research_endpoint(request: ResearchRequest):
    """Test endpoint for development and debugging"""
    return {
        "message": "Test endpoint - research parameters received",
        "query": request.query,
        "model": request.model,
        "api_key_length": len(request.api_key) if request.api_key else 0,
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    # Disable reload in production for faster startup
    reload = os.getenv("ENVIRONMENT", "production") == "development"
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level="info"
    )