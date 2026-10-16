"""Background task processing with Celery.

Optional module — only activated when ``celery_enabled = true`` in config.
Provides asynchronous document processing via a Celery worker queue.
"""

#: Celery's limits on one attempt at processing a document (``celery_app.py``).
TASK_SOFT_TIME_LIMIT_S = 540
TASK_TIME_LIMIT_S = 600
#: Attempts ``process_document_task`` makes (``max_retries=2``), and the wait between them.
TASK_ATTEMPTS = 3
TASK_RETRY_DELAY_S = 30
