import cv2
import os
import time
import threading
import queue
import collections
import glob
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False
    logger.warning("PyAV não instalado. Formato MKV não estará disponível.")

class Recorder:
    def __init__(self, output_dir='../videos', fps=20, codec='mp4v',
                 pre_record_seconds=2, max_queue_size=60,
                 max_storage_bytes=None, storage_policy='stop',
                 equipment_id='0000', filename_prefix='clip',
                 use_mkv=False, mkv_codec='libx264',
                 protected=False):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.fps = fps
        self.use_mkv = use_mkv and PYAV_AVAILABLE
        if self.use_mkv and not PYAV_AVAILABLE:
            logger.error("PyAV não disponível, revertendo para MP4.")
            self.use_mkv = False

        self.codec = codec
        self.mkv_codec = mkv_codec
        self.pre_record_seconds = pre_record_seconds
        self.buffer_size = int(fps * pre_record_seconds)
        self.frame_buffer = collections.deque(maxlen=self.buffer_size)  # (frame, timestamp)
        self.frame_queue = queue.Queue(maxsize=max_queue_size)

        self.max_storage_bytes = max_storage_bytes
        self.storage_policy = storage_policy
        self.protected = protected

        self.equipment_id = equipment_id
        self.filename_prefix = filename_prefix

        self._shutdown_sentinel = object()
        self._shutdown_started = False

        self.uploader = None

        self.thread = threading.Thread(target=self._record_worker, daemon=False)
        self.thread.start()

        self.lock = threading.Lock()
        self.end_timestamp = None
        self.recording = False
        self.writer = None
        self.container = None
        self.stream = None
        self._first_ts = None

        ext = "mkv" if self.use_mkv else "mp4"
        logger.info(f"Recorder inicializado: {output_dir}, prefixo={filename_prefix}, formato={ext}, "
                    f"pré-gravação={pre_record_seconds}s, id={equipment_id}, protegido={protected}")

    def set_uploader(self, uploader_obj):
        self.uploader = uploader_obj

    def _get_total_storage_used(self):
        total = 0
        for f in glob.glob(os.path.join(self.output_dir, "*")):
            try:
                total += os.path.getsize(f)
            except OSError:
                pass
        return total

    def _is_debug_file(self, filepath):
        return os.path.basename(filepath).startswith('debug_')

    def _try_free_space(self):
        if self.max_storage_bytes is None:
            return True
        total = self._get_total_storage_used()
        if total <= self.max_storage_bytes:
            return True
        all_files = glob.glob(os.path.join(self.output_dir, "*"))
        commons = [f for f in all_files if not self._is_debug_file(f)]
        commons.sort(key=os.path.getmtime)
        while commons and self._get_total_storage_used() > self.max_storage_bytes:
            oldest = commons.pop(0)
            try:
                os.remove(oldest)
                logger.info(f"Removido: {oldest}")
            except OSError as e:
                logger.error(f"Erro ao remover {oldest}: {e}")
        return self._get_total_storage_used() <= self.max_storage_bytes

    def start_recording(self):
        with self.lock:
            if self.recording:
                return False
            if self.storage_policy == 'delete_oldest':
                self._try_free_space()
            self.recording = True
            start_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            ext = "mkv" if self.use_mkv else "mp4"
            self.filename = os.path.join(self.output_dir,
                f"{self.filename_prefix}_{start_ts}_{self.equipment_id}.{ext}")
            self.end_timestamp = None
            if self.use_mkv:
                self.container = av.open(self.filename, 'w')
                self.stream = self.container.add_stream(self.mkv_codec, rate=self.fps)
                self.stream.pix_fmt = 'yuv420p'
                self.stream.width = None
                self.stream.height = None
                self.stream.time_base = av.time_base(1_000_000)
                self._first_ts = None
            else:
                self.writer = None
            buffer_copy = list(self.frame_buffer)
            for item in buffer_copy:
                self.frame_queue.put(item)
            logger.debug(f"Iniciando gravação: {self.filename}")
            return True

    def add_frame(self, frame, timestamp=None):
        with self.lock:
            if timestamp is None:
                timestamp = time.time()
            self.frame_buffer.append((frame.copy(), timestamp))
            if self.recording:
                self.frame_queue.put((frame.copy(), timestamp))

    def stop_recording(self):
        with self.lock:
            if self.recording:
                self.recording = False
                self.end_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                self.frame_queue.put(None)

    def shutdown(self):
        if not self._shutdown_started:
            self._shutdown_started = True
            self.frame_queue.put(self._shutdown_sentinel)
            self.thread.join()

    def _record_worker(self):
        while True:
            item = self.frame_queue.get()
            if item is None:
                if self.use_mkv:
                    self._finalize_mkv()
                else:
                    self._finalize_mp4()
                continue
            elif item is self._shutdown_sentinel:
                if self.use_mkv:
                    self._finalize_mkv()
                else:
                    if self.writer:
                        self.writer.release()
                break
            else:
                frame, ts = item
                if self.use_mkv:
                    self._write_frame_mkv(frame, ts)
                else:
                    self._write_frame_mp4(frame)

    def _write_frame_mkv(self, frame, ts):
        if self.stream.width is None:
            h, w = frame.shape[:2]
            self.stream.width = w
            self.stream.height = h
        if self._first_ts is None:
            self._first_ts = ts
        pts = int((ts - self._first_ts) * 1_000_000)
        img = av.VideoFrame.from_ndarray(frame, format='bgr24')
        img.pts = pts
        for packet in self.stream.encode(img):
            self.container.mux(packet)

    def _write_frame_mp4(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (w, h))
        self.writer.write(frame)

    def _finalize_mkv(self):
        if self.stream is not None:
            for packet in self.stream.encode(None):
                self.container.mux(packet)
            self.container.close()
            self.stream = None
            self.container = None
        self._rename_and_handle()

    def _finalize_mp4(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        self._rename_and_handle()

    def _rename_and_handle(self):
        if self.end_timestamp and self.filename:
            base, ext = os.path.splitext(self.filename)
            parts = base.rsplit('_', 1)
            if len(parts) == 2:
                new_base = f"{parts[0]}_{self.end_timestamp}_{parts[1]}"
            else:
                new_base = f"{base}_{self.end_timestamp}"
            new_filename = new_base + ext
            try:
                os.rename(self.filename, new_filename)
                final_filename = new_filename
            except OSError as e:
                logger.error(f"Erro ao renomear: {e}")
                final_filename = self.filename
        else:
            final_filename = self.filename

        logger.debug(f"Vídeo finalizado: {final_filename}")
        self._handle_finished_file(final_filename)

    def _handle_finished_file(self, filepath):
        uploaded = False
        if self.uploader:
            try:
                success = self.uploader.upload_sync(filepath)
                if success:
                    uploaded = True
                    logger.info(f"Upload concluído: {filepath}")
                else:
                    logger.error(f"Upload falhou para {filepath}")
            except Exception as e:
                logger.exception(f"Exceção no upload: {e}")

        if self.max_storage_bytes is None:
            return

        current_usage = self._get_total_storage_used()
        if self.storage_policy == 'stop':
            if current_usage > self.max_storage_bytes and not self.protected:
                try:
                    os.remove(filepath)
                    logger.info(f"Política 'stop': clipe deletado após upload ({filepath})")
                except OSError as e:
                    logger.error(f"Erro ao deletar {filepath}: {e}")
        elif self.storage_policy == 'delete_oldest':
            self._try_free_space()
            if self._get_total_storage_used() > self.max_storage_bytes and not self.protected:
                try:
                    os.remove(filepath)
                    logger.info(f"Política 'delete_oldest': espaço insuficiente, clipe deletado após upload ({filepath})")
                except OSError as e:
                    logger.error(f"Erro ao deletar {filepath}: {e}")