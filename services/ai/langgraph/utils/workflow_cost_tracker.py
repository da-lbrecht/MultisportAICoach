import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .langsmith_cost_extractor import LangSmithCostExtractor, TokenUsageCallbackHandler, WorkflowCostSummary

logger = logging.getLogger(__name__)


@dataclass
class WorkflowExecution:
    trace_id: str
    root_run_id: str
    start_time: datetime
    end_time: datetime | None = None
    execution_time_seconds: float = 0.0
    cost_summary: WorkflowCostSummary | None = None


class WorkflowCostTracker:

    def __init__(self, project_name: str = "garmin_ai_coach_analysis"):
        self.project_name = project_name
        self.cost_extractor = LangSmithCostExtractor()

    async def run_workflow_with_cost_tracking(
        self,
        workflow_app,
        initial_state: dict[str, Any],
        thread_id: str,
        user_id: str | None = None,
        progress_callback: Callable[[str, WorkflowCostSummary], Awaitable[None]] | None = None,
    ) -> tuple[dict[str, Any], WorkflowExecution]:

        root_run_id = uuid.uuid4()

        execution = WorkflowExecution(
            trace_id=str(root_run_id),  # For root runs, trace_id equals run_id
            root_run_id=str(root_run_id),
            start_time=datetime.now(),
        )

        final_state = dict(initial_state) if initial_state else {}

        token_tracker = TokenUsageCallbackHandler()

        try:
            logger.info("Starting workflow execution with deterministic root_run_id: %s", root_run_id)

            config = {
                "run_id": root_run_id,
                "run_name": "garmin_ai_coach_workflow",
                "tags": [
                    f"user:{user_id}" if user_id else "user:unknown",
                    "app:garmin_ai_coach",
                    f"thread:{thread_id}" if thread_id else "thread:none",
                ],
                "metadata": {
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "project": self.project_name,
                },
                # Propagate to all llm.ainvoke() calls inside nodes via LangChain contextvars
                "callbacks": [token_tracker],
            }

            if thread_id:
                config["configurable"] = {"thread_id": thread_id}

            prev_lengths: dict[str, int | None] = {"analysis_html": None, "planning_html": None}

            async for chunk in workflow_app.astream(
                initial_state, config=config, stream_mode="values"
            ):
                logger.debug("Workflow step: %s", list(chunk.keys()) if chunk else "None")
                if chunk:
                    final_state = chunk  # Take the latest complete state snapshot

                    for _key in ("analysis_html", "planning_html"):
                        if chunk.get(_key):
                            curr_len = len(str(chunk[_key]))
                            if prev_lengths[_key] != curr_len:
                                logger.info("%s updated: %d chars", _key, curr_len)
                                prev_lengths[_key] = curr_len
                            else:
                                logger.debug("%s unchanged: %d chars", _key, curr_len)

            logger.info(
                "Workflow execution complete with deterministic trace_id: %s",
                execution.trace_id,
            )

            execution.end_time = datetime.now()
            execution.execution_time_seconds = (
                execution.end_time - execution.start_time
            ).total_seconds()

            try:
                # Primary: callback-based extraction (works without LangSmith)
                if token_tracker.total_tokens > 0:
                    logger.info(
                        "Callback captured %d tokens across %d LLM call(s)",
                        token_tracker.total_tokens,
                        token_tracker.call_count,
                    )
                    cost_summary = self.cost_extractor.extract_costs_from_callback(
                        token_tracker, execution.execution_time_seconds
                    )
                    cost_summary.trace_id = execution.trace_id
                    cost_summary.root_run_id = execution.root_run_id
                else:
                    # Fallback: try LangSmith (only useful when LANGSMITH_API_KEY is set)
                    logger.info("Callback captured 0 tokens — trying LangSmith trace: %s", execution.trace_id)
                    cost_summary = self.cost_extractor.extract_workflow_costs_by_trace(
                        execution.trace_id, execution.execution_time_seconds
                    )
                    cost_summary.root_run_id = execution.root_run_id
                    if cost_summary.total_tokens == 0:
                        logger.warning("Both callback and LangSmith returned 0 tokens")

                execution.cost_summary = cost_summary

                if progress_callback and cost_summary.total_cost_usd > 0:
                    await progress_callback("workflow_complete", cost_summary)

                logger.info(
                    "Workflow execution complete: $%.4f (%d tokens, %d LLM calls)",
                    cost_summary.total_cost_usd,
                    cost_summary.total_tokens,
                    token_tracker.call_count,
                )

            except Exception as e:
                logger.error("Error extracting costs for trace %s: %s", execution.trace_id, e)
                if token_tracker.total_tokens > 0:
                    try:
                        cost_summary = self.cost_extractor.extract_costs_from_callback(
                            token_tracker, execution.execution_time_seconds
                        )
                        cost_summary.trace_id = execution.trace_id
                        cost_summary.root_run_id = execution.root_run_id
                        execution.cost_summary = cost_summary
                    except Exception:
                        execution.cost_summary = self.cost_extractor._zero_workflow_summary(
                            execution.trace_id
                        )
                else:
                    execution.cost_summary = self.cost_extractor._zero_workflow_summary(
                        execution.trace_id
                    )

        except Exception as e:
            logger.error("Error in workflow execution: %s", e)
            execution.end_time = datetime.now()
            execution.execution_time_seconds = (
                execution.end_time - execution.start_time
            ).total_seconds()
            execution.cost_summary = self.cost_extractor._zero_workflow_summary(execution.trace_id)
            raise

        return final_state, execution

    def get_legacy_cost_summary(self, execution: WorkflowExecution) -> dict[str, Any]:
        if not execution.cost_summary:
            return {"total_cost_usd": 0.0, "total_tokens": 0, "agents": [], "model_breakdown": {}}

        cost_summary = execution.cost_summary

        agents = []
        model_breakdown = {}

        for node in cost_summary.node_costs:
            agents.append(
                {
                    "name": node.name,
                    "cost_usd": node.cost_usd,
                    "tokens": node.tokens,
                    "execution_time_seconds": 0.0,  # Per-node time not available yet
                    "models": [
                        {
                            "name": node.model or "unknown",
                            "cost_usd": node.cost_usd,
                            "input_tokens": node.input_tokens,
                            "output_tokens": node.output_tokens,
                            "total_tokens": node.tokens,
                            "web_search_requests": node.web_search_requests,
                        }
                    ],
                }
            )

            model_key = node.model or "unknown"
            if model_key not in model_breakdown:
                model_breakdown[model_key] = {
                    "cost_usd": 0.0,
                    "tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "web_search_requests": 0,
                }

            model_breakdown[model_key]["cost_usd"] += node.cost_usd
            model_breakdown[model_key]["tokens"] += node.tokens
            model_breakdown[model_key]["input_tokens"] += node.input_tokens
            model_breakdown[model_key]["output_tokens"] += node.output_tokens
            model_breakdown[model_key]["web_search_requests"] += node.web_search_requests

        return {
            "total_cost_usd": cost_summary.total_cost_usd,
            "total_tokens": cost_summary.total_tokens,
            "total_execution_time_seconds": cost_summary.execution_time_seconds,
            "agent_count": len(cost_summary.node_costs),
            "agents": agents,
            "model_breakdown": model_breakdown,
        }


class ProgressIntegratedCostTracker(WorkflowCostTracker):

    def __init__(self, project_name: str = "garmin_ai_coach_analysis", progress_manager=None):
        super().__init__(project_name)
        self.progress_manager = progress_manager

    async def run_workflow_with_progress(
        self, workflow_app, initial_state: dict[str, Any], thread_id: str, user_id: str | None = None
    ) -> tuple[dict[str, Any], WorkflowExecution]:

        async def progress_callback(_event: str, cost_summary: WorkflowCostSummary):
            if self.progress_manager and hasattr(self.progress_manager, "analysis_stats"):
                self.progress_manager.analysis_stats["total_cost_usd"] = cost_summary.total_cost_usd
                self.progress_manager.analysis_stats["total_tokens"] = cost_summary.total_tokens

                if cost_summary.node_costs:
                    self.progress_manager.analysis_stats["agents_completed"] = len(
                        cost_summary.node_costs
                    )

                logger.info(
                    "Updated progress manager with cost: $%.4f",
                    cost_summary.total_cost_usd,
                )

        return await self.run_workflow_with_cost_tracking(
            workflow_app, initial_state, thread_id, user_id, progress_callback
        )
