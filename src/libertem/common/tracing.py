import os
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional
import contextlib
import uuid

from opentelemetry import trace, context
from opentelemetry.trace import NonRecordingSpan, SpanContext

if TYPE_CHECKING:
    from libertem.io.dataset.base import Partition

logger = logging.getLogger()


def maybe_setup_tracing(
    service_name: str,
    service_id: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
):
    """
    Set up tracing if the OTEL_ENABLE environment variable is set.
    Only call this once per process.

    Please see :func:`setup_tracing` for a description of the parameters.
    """
    if os.environ.get('OTEL_ENABLE') is not None:
        return setup_tracing(service_name, service_id, otlp_endpoint)


def setup_tracing(
    service_name: str,
    service_id: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
):
    """
    Set up tracing and span exporting. Call once per process.

    Parameters
    ----------

    service_name
        The service name which will appear in the trace

    service_id
        The service id, if there are multiple instances of the same service
        (i.e. worker processes). Should stay the same over the runtime of
        the program, but can be different when "major lifecycle events" happen,
        for example program restarts

    otlp_endpoint
        Set up exporting of tracing spans via OTLP. If this is not set, this
        will default to the value of the OTEL_EXPORTER_OTLP_ENDPOINT
        environment variable, or fall back to http://localhost:4317
    """
    if otlp_endpoint is None:
        otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint is None:
        otlp_endpoint = "http://localhost:4317"
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        # from opentelemetry.sdk.trace.export import ConsoleSpanExporter
    except ImportError:
        logger.warn("could not import opentelemetry library, disabling tracing")
        return

    if service_id is None:
        service_id = str(uuid.uuid4())
    provider = TracerProvider(resource=Resource.create({
        "service.name": service_name,
        "service.instance.id": service_id,
    }))
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    # console_exporter = ConsoleSpanExporter()
    processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)


@contextlib.contextmanager
def attach_to_parent(span_context: SpanContext):
    """
    Attach to an existing span, such that new spans created will be parented
    to it. This is useful for serializing the `SpanContext` to a different
    process, possibly running on a different node, and keeping the tracing
    relation.

    See also: https://opentelemetry.io/docs/instrumentation/python/cookbook/
    """
    trace_context = trace.set_span_in_context(NonRecordingSpan(span_context))
    token = context.attach(trace_context)
    try:
        yield
    finally:
        context.detach(token)


def add_partition_to_span(partition: "Partition") -> None:
    """
    Add Partition metadata (start_idx, end_idx) to the current tracing span
    """
    span = trace.get_current_span()
    slice_ = partition.slice
    span.set_attributes({
        "libertem.partition.start_idx": slice_.origin[0],
        "libertem.partition.end_idx": slice_.origin[0] + slice_.shape[0],
    })


class TracedThreadPoolExecutor(ThreadPoolExecutor):
    """
    A :code:`~concurrent.futures.ThreadPoolExecutor` that propagates the submit
    time tracing context to the worker threads.

    >>> tracer = trace.get_tracer(__name__)
    >>> pool = TracedThreadPoolExecutor(tracer)
    >>> fut = pool.submit(...)
    """
    def __init__(self, tracer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracer = tracer

    def _in_parent_span(self, span_context: SpanContext, fn, *args, **kwargs):
        with attach_to_parent(span_context):
            return fn(*args, **kwargs)

    def submit(self, __fn, *args, **kwargs):
        if context.get_current():
            span = trace.get_current_span()
            span_context = span.get_span_context()
            return super().submit(
                lambda: self._in_parent_span(span_context, __fn, *args, **kwargs)
            )
        return super().submit(__fn, *args, **kwargs)
