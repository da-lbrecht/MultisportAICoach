import logging
import os
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langsmith import Client
from requests import HTTPError

logger = logging.getLogger(__name__)

# Pricing per 1 million tokens (USD). Update as provider rates change.
_MODEL_PRICING_USD_PER_M: dict[str, dict[str, float]] = {
    # Anthropic — current models (June 2026)
    "claude-fable-5": {"input": 10.00, "output": 50.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    # Anthropic — legacy (kept for historical runs)
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-opus-4-1-20250805": {"input": 15.00, "output": 75.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "o3": {"input": 10.00, "output": 40.00},
    "o4-mini": {"input": 1.10, "output": 4.40},
}


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that accumulates token usage across all LLM calls.

    LangGraph propagates callbacks from the parent ``astream`` config into every
    node's execution context via ``contextvars``.  Any ``llm.ainvoke()`` call made
    inside a node will therefore trigger ``on_llm_end`` here — no node changes needed.
    """

    raise_error = False  # Prevent callback exceptions from crashing the workflow

    def __init__(self) -> None:
        super().__init__()
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.call_count: int = 0
        self._model_usage: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                usage = getattr(msg, "usage_metadata", None)
                if not usage:
                    continue
                inp = int(usage.get("input_tokens", 0) or 0)
                out = int(usage.get("output_tokens", 0) or 0)
                rm = getattr(msg, "response_metadata", {}) or {}
                model = rm.get("model") or rm.get("model_name", "")
                with self._lock:
                    self.input_tokens += inp
                    self.output_tokens += out
                    self.call_count += 1
                    if model:
                        entry = self._model_usage.setdefault(model, {"input": 0, "output": 0})
                        entry["input"] += inp
                        entry["output"] += out

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class NodeCostSummary:
    name: str
    run_id: str
    model: str | None
    cost_usd: float
    tokens: int
    input_tokens: int = 0
    output_tokens: int = 0
    web_search_requests: int = 0


@dataclass
class WorkflowCostSummary:
    trace_id: str
    root_run_id: str
    total_cost_usd: float
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    total_web_searches: int
    node_costs: list[NodeCostSummary]
    execution_time_seconds: float = 0.0


class LangSmithCostExtractor:

    def __init__(self):
        self.client = None
        if os.getenv("LANGSMITH_API_KEY"):
            try:
                self.client = Client()
                logger.info("LangSmith cost extractor initialized")
            except Exception as exc:
                logger.warning("Failed to initialize LangSmith client: %s", exc)

    def safe_read_run(
        self, run_id: str, retries: int = 3, backoff: float = 0.5, load_children: bool = True
    ):
        if not self.client:
            return None

        for i in range(retries):
            try:
                return self.client.read_run(run_id, load_child_runs=load_children)
            except HTTPError as e:
                logger.warning(
                    "HTTP error reading run %s, attempt %s/%s: %s",
                    run_id,
                    i + 1,
                    retries,
                    e,
                )
                if i == retries - 1:
                    raise
                time.sleep(backoff * (2**i))
            except Exception:
                logger.exception("Unexpected error reading run %s", run_id)
                if i == retries - 1:
                    raise
                time.sleep(backoff)
        return None

    def extract_workflow_costs_by_trace(
        self, trace_id: str, execution_time: float = 0.0
    ) -> WorkflowCostSummary:
        if not self.client:
            logger.warning("LangSmith client not available - returning zero costs")
            return self._zero_workflow_summary(trace_id)

        try:
            all_runs = list(
                self.client.list_runs(
                    trace=trace_id, select=["id", "name", "run_type", "total_cost", "total_tokens"]
                )
            )
            logger.info("Found %s total runs for trace %s", len(all_runs), trace_id)
            for run in all_runs[:5]:  # Log first 5 for debugging
                run_cost = float(run.total_cost or 0)
                logger.info(
                    "  Run: %s (type: %s) - Cost: $%.4f",
                    run.name,
                    run.run_type,
                    run_cost,
                )

            llm_runs = list(
                self.client.list_runs(
                    trace=trace_id,
                    filter='eq(run_type, "llm")',
                    select=[
                        "id",
                        "name",
                        "total_cost",
                        "total_tokens",
                        "prompt_tokens",
                        "completion_tokens",
                        "serialized",
                    ],
                )
            )
            logger.info("Found %s LLM runs for trace %s", len(llm_runs), trace_id)

            total_cost = Decimal("0")
            total_tokens = 0
            total_input_tokens = 0
            total_output_tokens = 0
            total_web_searches = 0
            node_costs = []

            for run in llm_runs:
                cost = Decimal(str(run.total_cost or 0))
                tokens = run.total_tokens or 0
                input_tokens = run.prompt_tokens or 0
                output_tokens = run.completion_tokens or 0
                model = (run.serialized or {}).get("model", "unknown")

                web_searches = 0
                if "search" in run.name.lower() or "web" in run.name.lower():
                    web_searches = 1  # Approximate

                total_cost += cost
                total_tokens += tokens
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_web_searches += web_searches

                node_costs.append(
                    NodeCostSummary(
                        name=run.name,
                        run_id=str(run.id),
                        model=model,
                        cost_usd=float(cost),
                        tokens=tokens,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        web_search_requests=web_searches,
                    )
                )

            logger.info(
                "Extracted costs for trace %s: $%.4f (%s tokens)",
                trace_id,
                float(total_cost),
                total_tokens,
            )

            return WorkflowCostSummary(
                trace_id=trace_id,
                root_run_id="",  # Will be filled by caller if needed
                total_cost_usd=float(total_cost),
                total_tokens=total_tokens,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_web_searches=total_web_searches,
                node_costs=node_costs,
                execution_time_seconds=execution_time,
            )

        except Exception:
            logger.exception("Failed to extract workflow costs for trace %s", trace_id)
            return self._zero_workflow_summary(trace_id)

    def extract_run_costs(self, run_id: str) -> dict[str, Any]:
        try:
            root_run = self.safe_read_run(run_id, load_children=True)
            if not root_run:
                return self._zero_cost_summary(run_id)

            workflow_summary = self.extract_workflow_costs_by_trace(str(root_run.trace_id))
            workflow_summary.root_run_id = run_id

            model_breakdown = {}
            for node in workflow_summary.node_costs:
                model_key = node.model or "unknown"
                if model_key not in model_breakdown:
                    model_breakdown[model_key] = {
                        "cost_usd": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "web_search_requests": 0,
                    }

                model_breakdown[model_key]["cost_usd"] += node.cost_usd
                model_breakdown[model_key]["input_tokens"] += node.input_tokens
                model_breakdown[model_key]["output_tokens"] += node.output_tokens
                model_breakdown[model_key]["total_tokens"] += node.tokens
                model_breakdown[model_key]["web_search_requests"] += node.web_search_requests

            return {
                "total_cost_usd": workflow_summary.total_cost_usd,
                "total_tokens": workflow_summary.total_tokens,
                "model_breakdown": model_breakdown,
                "run_id": run_id,
                "trace_id": workflow_summary.trace_id,
            }

        except Exception:
            logger.exception("Failed to extract costs from LangSmith run %s", run_id)
            return self._zero_cost_summary(run_id)

    def extract_costs_from_messages(
        self, messages: list, execution_time: float = 0.0
    ) -> "WorkflowCostSummary":
        """Extract token usage directly from LangChain AIMessage objects in the state.

        This is the primary fallback when LangSmith is not configured or returns
        no data. LangChain stores ``usage_metadata`` (input_tokens, output_tokens)
        on every AIMessage, so this works with any provider without any external
        service or API key.
        """
        total_input = 0
        total_output = 0
        total_cost = Decimal("0")

        for msg in messages:
            usage: dict | None = None
            model = ""

            # LangChain message object (live during streaming)
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                usage = msg.usage_metadata
                rm = getattr(msg, "response_metadata", {}) or {}
                model = rm.get("model") or rm.get("model_name", "")

            # Serialised dict (e.g. from MemorySaver checkpoint)
            elif isinstance(msg, dict) and msg.get("type") == "ai":
                usage = msg.get("usage_metadata") or {}
                rm = msg.get("response_metadata") or {}
                model = rm.get("model") or rm.get("model_name", "")

            if not usage:
                continue

            inp = int(usage.get("input_tokens", 0) or 0)
            out = int(usage.get("output_tokens", 0) or 0)
            total_input += inp
            total_output += out

            pricing = _MODEL_PRICING_USD_PER_M.get(model, {})
            if pricing:
                total_cost += Decimal(str(inp * pricing["input"] / 1_000_000))
                total_cost += Decimal(str(out * pricing["output"] / 1_000_000))
            elif model:
                logger.debug("No pricing data for model '%s' — tokens counted, cost unknown", model)

        logger.info(
            "Message-based token extraction: %d in + %d out = %d total tokens, est. $%.4f",
            total_input,
            total_output,
            total_input + total_output,
            float(total_cost),
        )

        return WorkflowCostSummary(
            trace_id="",
            root_run_id="",
            total_cost_usd=float(total_cost),
            total_tokens=total_input + total_output,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_web_searches=0,
            node_costs=[],
            execution_time_seconds=execution_time,
        )

    def _zero_cost_summary(self, run_id: str | None = None) -> dict[str, Any]:
        return {
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "model_breakdown": {},
            "run_id": run_id,
            "trace_id": None,
        }

    def _zero_workflow_summary(self, trace_id: str) -> WorkflowCostSummary:
        return WorkflowCostSummary(
            trace_id=trace_id,
            root_run_id="",
            total_cost_usd=0.0,
            total_tokens=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_web_searches=0,
            node_costs=[],
            execution_time_seconds=0.0,
        )

    def extract_costs_from_callback(
        self,
        handler: "TokenUsageCallbackHandler",
        execution_time: float,
    ) -> WorkflowCostSummary:
        """Build a WorkflowCostSummary from a TokenUsageCallbackHandler.

        This is the primary cost-extraction path when LangSmith is not configured.
        The handler is registered in the LangGraph ``astream`` config so every
        ``llm.ainvoke()`` inside a node automatically fires ``on_llm_end``.
        """
        total_cost = Decimal("0")
        node_costs: list[NodeCostSummary] = []

        for model, usage in handler._model_usage.items():
            inp = usage.get("input", 0)
            out = usage.get("output", 0)
            pricing = _MODEL_PRICING_USD_PER_M.get(model, {})
            cost = Decimal(str(inp)) * Decimal(str(pricing.get("input", 0))) / Decimal("1000000") + \
                   Decimal(str(out)) * Decimal(str(pricing.get("output", 0))) / Decimal("1000000")
            total_cost += cost
            node_costs.append(
                NodeCostSummary(
                    name=model,
                    run_id="",
                    model=model,
                    cost_usd=float(cost),
                    tokens=inp + out,
                    input_tokens=inp,
                    output_tokens=out,
                )
            )

        if not node_costs and handler.total_tokens > 0:
            # Model name unknown — record as a single unattributed entry
            node_costs.append(
                NodeCostSummary(
                    name="unknown",
                    run_id="",
                    model=None,
                    cost_usd=0.0,
                    tokens=handler.total_tokens,
                    input_tokens=handler.input_tokens,
                    output_tokens=handler.output_tokens,
                )
            )

        return WorkflowCostSummary(
            trace_id="",
            root_run_id="",
            total_cost_usd=float(total_cost),
            total_tokens=handler.total_tokens,
            total_input_tokens=handler.input_tokens,
            total_output_tokens=handler.output_tokens,
            total_web_searches=0,
            node_costs=node_costs,
            execution_time_seconds=execution_time,
        )
