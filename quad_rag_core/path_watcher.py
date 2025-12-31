# qdrant_rag_core/path_watcher.py
import os
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .file_processor import process_file
from qdrant_client.models import PointStruct, PointIdsList
from .utils import get_file_mime_type, is_text_file
from .embedder import LocalEmbedder
from .qdrant_manager import QdrantManager
from .file_processor import chunk_text
from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileMovedEvent, FileDeletedEvent
from qdrant_client.models import Filter, FieldCondition, MatchValue
from .config import CHUNK_CHARACTERS_PREVIEW

class RAGFileWatcher(FileSystemEventHandler):
    def __init__(self, folder_path, 
                collection_name, 
                qdrant_manager: QdrantManager, 
                embedder: LocalEmbedder, 
                content_types,
                skip_initial_scan: bool = False  
                ):
        self.is_restored = False  # True, if restored from Qdrant
        self.file_count_mode = "files"  # or "chunks"
        self.folder_path = folder_path
        self.collection_name = collection_name
        self.qdrant_manager = qdrant_manager
        self.embedder = embedder
        self.content_types = content_types
        self.observer = Observer()
        self._stop_event = threading.Event()
        self.skip_initial_scan = skip_initial_scan
        
        # Progress state
        self.total_files = 0
        self.processed_files = 0
        self.progress_percent = 0
        self.status = "idle"  # idle, scanning, watching
        self._progress_lock = threading.Lock()

    def start(self):
        self.observer.schedule(self, self.folder_path, recursive=True)
        self.observer.start()
        if self.skip_initial_scan:
            # Restore state from Qdrant
            self.status = "watching"
            self.file_count_mode = "chunks"
            self.is_restored = True
            self.total_files = self._count_existing_points()  # ← these are chunks!
            self.processed_files = self.total_files
            self.progress_percent = 100
        else:
            self.status = "scanning"
            self.file_count_mode = "files"
            self.is_restored = False
            self.processed_files = 0
            threading.Thread(target=self._initial_scan, daemon=True).start()

    def stop(self):
        self._stop_event.set()
        self.observer.stop()
        self.observer.join()
        self.status = "stopped"

    def get_status(self):
        with self._progress_lock:
            self.progress_percent = int((self.processed_files / max(self.total_files, 1)) * 100)
            return {
                "status": self.status,
                "total_files": self.total_files,
                "processed_files": self.processed_files,
                "progress_percent": self.progress_percent,
                "count_type": "chunks" if self.is_restored else "files"  
            }
        
    def _count_existing_points(self):
        # Count only chunks, excluding metadata
        try:
            count = self.qdrant_manager.client.count(
                collection_name=self.collection_name,
                exact=True
            ).count
            return max(0, count - 1)  # -1 for metadata
        except:
            return 0
            
    def _initial_scan(self):
        """Scans entire folder on first run."""
        try:
            # First count total number of files
            file_list = []
            for root, dirs, files in os.walk(self.folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if self._should_process(file_path):
                        file_list.append(file_path)
            
            with self._progress_lock:
                self.total_files = len(file_list)
                self.processed_files = 0

            # Process files
            for file_path in file_list:
                if self._stop_event.is_set():
                    break
                self._process_file(file_path)
                with self._progress_lock:
                    self.processed_files += 1

        except Exception as e:
            print(f"Error scanning {self.folder_path}: {e}")
        
        with self._progress_lock:
            self.status = "watching"  # now only watching for changes

    def _should_process(self, file_path: str) -> bool:
        mime = get_file_mime_type(file_path)
        if "text" in self.content_types and is_text_file(file_path):
            return True
        if "pdf" in self.content_types and mime == "application/pdf":
            return True
        return False

    def _process_file(self, file_path: str):
        if self._stop_event.is_set():
            return
        try:
            if not self._should_process(file_path):
                return

            # Delete old chunks of this file
            self._delete_file_from_qdrant(file_path)

            content = process_file(file_path, self.content_types)
            if not content or len(content.strip()) < 10:
                return

            chunks = chunk_text(content, chunk_size=100, overlap=0.15)
            for i, chunk in enumerate(chunks):
                if self._stop_event.is_set():
                    break
                if len(chunk.strip()) < 10:
                    continue

                vector = self.embedder.embed_document(chunk)
                point_id = abs(hash(file_path + str(i) + str(os.path.getmtime(file_path))))
                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "path": file_path,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "content_preview": chunk[:CHUNK_CHARACTERS_PREVIEW],
                        "mtime": os.path.getmtime(file_path)
                    }
                )
                self.qdrant_manager.upsert_points(self.collection_name, [point])

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    def on_created(self, event):
        if event.is_directory:
            return
        self._handle_file_event(str(event.src_path))

    def on_modified(self, event):
        if event.is_directory:
            return
        # avoid duplication with on_created on some OS
        self._handle_file_event(str(event.src_path))

    def on_moved(self, event):
        # event.dest_path — new name
        if not event.is_directory:
            self._handle_file_event(str(event.src_path))
        # old file can be deleted from Qdrant (optional)
        if hasattr(event, 'src_path') and not event.is_directory:
            self._delete_file_from_qdrant(str(event.src_path))

    def on_deleted(self, event):
        if not event.is_directory:
            self._delete_file_from_qdrant(str(event.src_path))

    def _handle_file_event(self, file_path: str):
        print(f"[DEBUG] File changed: {file_path}")
        if self._should_process(file_path):
            print(f"[DEBUG] → will be processed")
            threading.Thread(target=self._delayed_process, args=(file_path,), daemon=True).start()
        else:
            print(f"[DEBUG] → ignored")

    def _delayed_process(self, file_path: str, delay: float = 0.5):
        """Waits until file stops changing, then processes it."""
        time.sleep(delay)  # simple protection against partial write
        self._process_file(file_path)

    def _delete_file_from_qdrant(self, file_path: str):
        """Deletes all file chunks from collection."""
        try:

            # can use scroll + delete_by_id
            points = self.qdrant_manager.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="path", match=MatchValue(value=file_path))]
                ),
                limit=1000,
                with_payload=False
            )
            ids_to_delete = [point.id for point in points[0]]
            if ids_to_delete:
                self.qdrant_manager.client.delete(
                    collection_name=self.collection_name,
                    points_selector=PointIdsList(points=ids_to_delete)
                )
        except Exception as e:
            print(f"[WARN] Failed to delete {file_path} from Qdrant: {e}")