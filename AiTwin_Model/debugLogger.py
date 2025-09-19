import logging
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

class DebugLogger:
    _instances = {}

    def __new__(cls, service_name="default-service"):
        if service_name not in cls._instances:
            instance = super(DebugLogger, cls).__new__(cls)
            instance._init(service_name)
            cls._instances[service_name] = instance
        return cls._instances[service_name]

    def _init(self, service_name):
        # --- Configure logging ---
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # --- Configure OpenTelemetry Tracer with service.name ---
        resource = Resource.create({"service.name": service_name})

        provider = TracerProvider(resource=resource)
        # Fix: Removed the 'insecure' argument
        otlp_exporter = OTLPSpanExporter(
            endpoint="http://localhost:4318/v1/traces"  # Jaeger OTLP HTTP endpoint
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(service_name)

    def start_span(self, name: str):
        """Context manager for creating spans"""
        return self.tracer.start_as_current_span(name)

    def log(self, message: str, level: str = "info", **attributes):
        """Log message + attach attributes to active span"""
        log_method = getattr(self.logger, level, self.logger.info)
        log_method(message)

        span = trace.get_current_span()
        if span.is_recording():
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)