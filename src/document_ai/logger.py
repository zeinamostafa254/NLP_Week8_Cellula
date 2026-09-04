import logging
import sys

def setup_logging(level=logging.INFO):
    """Configure the root logger for the entire application."""
    # Define a clear format that shows time, module name, and the message
    log_format = "%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%H:%M:%S")

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid adding multiple handlers if called multiple times
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)

    # Silence noisy third-party loggers so our logs stand out
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
