"""
monitoring/telemetry.py
=======================
Centralised OpenTelemetry setup for the Mental Health Chatbot.

Instruments:
  - TracerProvider  → auto-traces every FastAPI request + outgoing httpx calls
  - LoggerProvider  → emits structured metric events (Axiom accepts OTLP logs,
                      not OTLP metrics, so metrics are sent as log events)

Metrics (emitted as structured log events)
------------------------------------------
1. chatbot.intent.count      Model/NLP metric
   Rationale: intent distribution drift (e.g. sudden out_of_scope spike)
   signals model degradation or unexpected user behaviour before users
   start complaining.

2. chatbot.message.length    Data metric
   Rationale: distribution shifts catch abuse (very long prompts),
   bot traffic (very short messages), or UI regressions.

3. chatbot.requests.total /  Server metric
   chatbot.errors.total
   Standard SRE signal; error_rate = errors/requests catches Gemini
   quota exhaustion (429), Qdrant failures, and pipeline crashes
   before the error rate becomes visible to end-users.

All telemetry is exported via OTLP/HTTP to the local OTel Collector,
which forwards traces and log events to Axiom.
"""

import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# OTel Logs SDK
from opentelemetry._logs import set_logger_provider, LogRecord, SeverityNumber
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

_SERVICE_NAME = "mental-health-chatbot"
_otel_logger = None


def setup_telemetry(app) -> None:
    """
    Call once inside the FastAPI lifespan startup block.
    Reads OTEL_COLLECTOR_ENDPOINT from the environment (default: http://localhost:4318).
    """
    global _otel_logger

    collector_endpoint = os.getenv(
        "OTEL_COLLECTOR_ENDPOINT", "http://localhost:4318"
    )

    resource = Resource.create({"service.name": _SERVICE_NAME})

    #  Traces 
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=f"{collector_endpoint}/v1/traces")
        )
    )
    trace.set_tracer_provider(tracer_provider)

    #  Logs (used to ship metric events to Axiom) 
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            OTLPLogExporter(endpoint=f"{collector_endpoint}/v1/logs")
        )
    )
    set_logger_provider(logger_provider)
    _otel_logger = logger_provider.get_logger(_SERVICE_NAME)

    #  Auto-instrumentation 
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()


def _emit(event_name: str, attributes: dict) -> None:
    """Emit a structured log event that shows up as a metric in Axiom."""
    if _otel_logger is None:
        return
    record = LogRecord(
        severity_number=SeverityNumber.INFO,
        severity_text="INFO",
        body=event_name,
        event_name=event_name,
        attributes={"event": event_name, **attributes},
    )
    _otel_logger.emit(record)


#  Public helpers called from main.py 

def record_request() -> None:
    """Metric 3 (server): count every /chat request."""
    _emit("chatbot.requests.total", {})


def record_error(error_code: str) -> None:
    """Metric 3 (server): count pipeline errors."""
    _emit("chatbot.errors.total", {"error_code": error_code})


def record_intent(intent: str) -> None:
    """Metric 1 (model/NLP): track intent distribution."""
    _emit("chatbot.intent.count", {"intent": intent})


def record_message_length(length: int) -> None:
    """Metric 2 (data): track incoming message length distribution."""
    _emit("chatbot.message.length", {"length": length})
