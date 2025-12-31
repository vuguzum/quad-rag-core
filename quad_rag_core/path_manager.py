# qdrant_rag_core/path_manager.py
import os
import threading
from typing import Set, List
from .path_watcher import RAGFileWatcher
from typing import Dict
import re
from .qdrant_manager import QdrantManager
from .embedder import LocalEmbedder
from qdrant_client.models import PointStruct
import uuid
from .config import TEXT_FILE_EXTENSIONS

# Unified, fixed ID for metadata in ANY collection
WATCHER_METADATA_ID = str(uuid.UUID("f0f0f0f0-0000-0000-0000-000000000001"))

class PathWatcherManager:
    def __init__(self, qdrant_manager: QdrantManager, embedder: LocalEmbedder, collection_prefix: str = "rag"):
        self.watchers: Dict[str, RAGFileWatcher] = {}
        self.collection_prefix = collection_prefix
        self.qdrant_manager = qdrant_manager
        self.embedder = embedder
        self._lock = threading.Lock()
         # 🔥 On initialization — load existing collections
        self._restore_from_qdrant()

    def _restore_from_qdrant(self):
        """Restores watchers from Qdrant metadata."""
        collections = self.qdrant_manager.client.get_collections().collections
        pattern = re.compile(rf"^{re.escape(self.collection_prefix)}_")

        for col in collections:
            if not pattern.match(col.name):
                continue

            try:
                # Get metadata
                meta = self.qdrant_manager.client.retrieve(
                    collection_name=col.name,
                    ids=[WATCHER_METADATA_ID],
                    with_payload=True,
                    with_vectors=False
                )
                if not meta or not meta[0].payload:
                    print(f"[WARN] No metadata in {col.name}")
                    continue

                config = meta[0].payload.get("watcher_config")
                if not config:
                    continue

                folder_path = config["folder_path"]
                content_types = config.get("content_types", ["text"])

                # Check if folder exists
                if not os.path.exists(folder_path):
                    print(f"[WARN] Path not found: {folder_path}, skipping {col.name}")
                    continue

                # Create real watcher
                watcher = RAGFileWatcher(
                    folder_path=folder_path,
                    collection_name=col.name,
                    qdrant_manager=self.qdrant_manager,
                    embedder=self.embedder,
                    content_types=content_types,
                    skip_initial_scan=True
                )
                watcher.is_restored = True
                watcher.file_count_mode = "chunks"
                # Count chunks (points without metadata)
                chunk_count = max(0, self._count_points(col.name) - 1)
                watcher.total_files = chunk_count
                watcher.processed_files = chunk_count
                watcher.progress_percent = 100
                watcher.status = "watching"
                self.watchers[folder_path] = watcher
                watcher.start()
                self.watchers[folder_path] = watcher
                print(f"[INFO] Restored watcher for {folder_path}")

            except Exception as e:
                print(f"[ERROR] Failed to restore {col.name}: {e}")

    def _count_points(self, collection_name: str) -> int:
            try:
                count = self.qdrant_manager.client.count(collection_name=collection_name).count
                return count
            except:
                return 0                    

    def _normalize_path(self, path: str) -> str:
        return os.path.normpath(os.path.abspath(path))

    def _check_conflict(self, new_path: str) -> List[str]:
        """Returns list of conflicting paths."""
        new_path = self._normalize_path(new_path)
        conflicts = []
        for watched in self.watchers:
            if (new_path.startswith(watched + os.sep) or
                watched.startswith(new_path + os.sep)):
                conflicts.append(watched)
        return conflicts

    @staticmethod
    def _sanitize_collection_name(base_name: str, max_length: int = 64) -> str:
        """Converts string to valid Qdrant collection name."""
        if not isinstance(base_name, str):
            raise ValueError(f"Expected string for collection name, got {type(base_name)}: {base_name}")
        if not base_name.strip():
            raise ValueError("Collection name cannot be empty")
        
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name)
        # Remove duplicate underscores and dots/dashes at ends
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_.-')
        
        if not sanitized:
            raise ValueError("Sanitized collection name is empty")
            
        return sanitized[:max_length]
            
    def watch_folder(self, folder_path: str, content_types: List[str] = list(TEXT_FILE_EXTENSIONS)):
        if not folder_path:
            raise ValueError("folder_path cannot be empty")
            
        folder_path = self._normalize_path(folder_path)
        print(f"[DEBUG] Normalized path: {folder_path}")  # for debugging
        
        with self._lock:
            conflicts = self._check_conflict(folder_path)
            if conflicts:
                raise ValueError(f"Path conflict with: {conflicts}")

            # Create readable name
            path_for_name = folder_path.replace(":", "").replace("\\", "_").replace("/", "_")
            base_name = f"{self.collection_prefix}_{path_for_name}"
            print(f"[DEBUG] Base name for collection: {base_name}")  # for debugging
            
            clean_name = self._sanitize_collection_name(base_name)
            print(f"[DEBUG] Final collection name: {clean_name}")

            self.qdrant_manager.ensure_collection(clean_name)
            # In qdrant_rag_core/path_manager.py, inside watch_folder
            meta_point = PointStruct(
                id=WATCHER_METADATA_ID,  # fixed ID
                vector=[0.0] * 768,     # dummy vector (or use Matryoshka 1D if supported)
                payload={
                    "watcher_config": {
                        "folder_path": folder_path,
                        "content_types": content_types or ["text"],
                        "collection_prefix": self.collection_prefix
                    }
                }
            )
            self.qdrant_manager.upsert_points(clean_name, [meta_point])

            watcher = RAGFileWatcher(
                folder_path=folder_path,
                collection_name=clean_name,
                qdrant_manager=self.qdrant_manager,
                embedder=self.embedder,
                content_types=content_types or ["text"]
            )
            watcher.start()
            self.watchers[folder_path] = watcher
            

    def unwatch_folder(self, path: str):
        if path in self.watchers:
            del self.watchers[path]
            print(f"[INFO] Folder {path} removed from watchers")
        else:
            print(f"[WARNING] Folder {path} not found in watchers")


    def get_watched_folders(self):
        #print(f"get_watched_folders", flush=True)
        # Add statuses
        folders = []
        for path, watcher in self.watchers.items():
            if watcher is None:
                print(f"[WARNING] watcher is None for path: {path}")
                continue
            #print(f"watcher: {watcher}", flush=True)
            status = watcher.get_status()
            #print(f"status = {status}")
            folders.append({
                    "path": path,
                    "status": status["status"],
                    "total_files": status["total_files"],
                    "processed_files": status["processed_files"],
                    "progress_percent": status["progress_percent"],
                    "collection_name": watcher.collection_name,
                    "count_type": status["count_type"]
                })

        return folders