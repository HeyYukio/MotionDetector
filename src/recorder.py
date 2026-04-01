import cv2
import os
import threading
import queue
import collections
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class Recorder:
    def __init__(self, output_dir='../videos', fps=20, codec='mp4v',
                 pre_record_seconds=2, max_queue_size=60):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.fps = fps
        self.codec = codec
        self.fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = None
        self.recording = False
        self.pre_record_seconds = pre_record_seconds
        self.buffer_size = int(fps * pre_record_seconds)
        self.frame_buffer = collections.deque(maxlen=self.buffer_size)
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.thread = threading.Thread(target=self._record_worker, daemon=True)
        self.thread.start()
        self.on_video_finished = None  # callback quando vídeo é finalizado
        self.lock = threading.Lock()   # para acesso ao buffer
        self.end_timestamp = None      # armazena timestamp do fim
        logger.info(f"Recorder inicializado: {output_dir}, pré-gravação={pre_record_seconds}s")

    def start_recording(self):
        with self.lock:
            if not self.recording:
                self.recording = True
                start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.filename = os.path.join(self.output_dir, f"clip_{start_timestamp}.mp4")
                self.end_timestamp = None  # reseta timestamp de fim
                # O writer será criado no primeiro frame após o buffer
                # Primeiro, coloca os frames do buffer na fila
                logger.debug(f"Iniciando gravação: {self.filename} (buffer: {len(self.frame_buffer)} frames)")
                buffer_copy = list(self.frame_buffer)
                for bframe in buffer_copy:
                    self.frame_queue.put(bframe)

    def add_frame(self, frame):
        """Adiciona frame ao buffer e, se gravando, à fila de gravação."""
        with self.lock:
            # Sempre adiciona ao buffer (para pré-gravação)
            self.frame_buffer.append(frame.copy())
            if self.recording:
                if self.writer is None:
                    # Cria writer com as dimensões do primeiro frame (do buffer)
                    h, w = frame.shape[:2]
                    self.writer = cv2.VideoWriter(self.filename, self.fourcc, self.fps, (w, h))
                # Coloca o frame atual na fila
                self.frame_queue.put(frame.copy())

    def stop_recording(self):
        with self.lock:
            if self.recording:
                self.recording = False
                # Captura o timestamp exato do fim da gravação
                self.end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                logger.debug("Parando gravação")
                self.frame_queue.put(None)  # sinaliza fim

    def _record_worker(self):
        while True:
            frame = self.frame_queue.get()
            if frame is None:
                if self.writer:
                    self.writer.release()
                    self.writer = None
                    # Renomeia o arquivo para incluir o timestamp de fim, se disponível
                    if self.end_timestamp:
                        base, ext = os.path.splitext(self.filename)
                        new_filename = f"{base}_{self.end_timestamp}{ext}"
                        try:
                            os.rename(self.filename, new_filename)
                            final_filename = new_filename
                            logger.debug(f"Vídeo renomeado para: {final_filename}")
                        except OSError as e:
                            logger.error(f"Erro ao renomear vídeo: {e}")
                            final_filename = self.filename
                    else:
                        final_filename = self.filename
                    if self.on_video_finished:
                        self.on_video_finished(final_filename)
                continue
            if self.writer:
                self.writer.write(frame)