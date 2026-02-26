"""Background cleanup service for old uploaded documents."""
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

from src.config import get_settings
from src.models import Document

logger = logging.getLogger(__name__)


class DocumentCleanupService:
    """Periodically deletes documents older than configured retention period."""

    def __init__(self):
        self.settings = get_settings()
        self.upload_dir = Path(self.settings.UPLOAD_DIR)
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self):
        """Start the background cleanup loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "Started document cleanup service (retention_days=%s, interval_hours=%s)",
            self.settings.DOCUMENT_RETENTION_DAYS,
            self.settings.CLEANUP_INTERVAL_HOURS,
        )

    async def stop(self):
        """Stop the background cleanup loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Stopped document cleanup service")

    async def run_once(self) -> int:
        """Run one cleanup cycle and return count of deleted documents."""
        cutoff = datetime.now() - timedelta(days=self.settings.DOCUMENT_RETENTION_DAYS)
        stale_docs = await Document.filter(upload_date__lt=cutoff).all()

        deleted_count = 0
        for doc in stale_docs:
            file_path = self.upload_dir / doc.filename
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as file_error:
                logger.warning("Failed to delete file %s: %s", file_path, file_error)

            await doc.delete()
            deleted_count += 1

        if deleted_count:
            logger.info("Cleanup removed %s stale documents", deleted_count)

        return deleted_count

    async def _run_loop(self):
        """Run cleanup periodically until stopped."""
        try:
            while self._running:
                try:
                    await self.run_once()
                except Exception as error:
                    logger.error("Document cleanup cycle failed: %s", error, exc_info=True)

                await asyncio.sleep(max(1, self.settings.CLEANUP_INTERVAL_HOURS) * 3600)
        except asyncio.CancelledError:
            logger.info("Document cleanup loop cancelled")
