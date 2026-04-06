import cv2
import os
import threading
import queue
import collections
import glob
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class Recorder:
    def __init__(self, output_dir='../videos', fps=20, codec='mp4v',
                 pre_record_seconds=2, max_queue_size=60,
                 max_storage_bytes=None, storage_policy='stop'):
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

        # Limite de armazenamento (bytes) e política
        self.max_storage_bytes = max_storage_bytes
        self.storage_policy = storage_policy
        self._storage_exceeded = False  # Usado apenas na política 'stop'

        # Sentinela para desligamento
        self._shutdown_sentinel = object()
        self._shutdown_started = False

        self.thread = threading.Thread(target=self._record_worker, daemon=False)
        self.thread.start()

        self.on_video_finished = None
        self.lock = threading.Lock()
        self.end_timestamp = None

        logger.info(f"Recorder inicializado: {output_dir}, pré-gravação={pre_record_seconds}s")
        if self.max_storage_bytes:
            logger.info(f"Limite de armazenamento: {self.max_storage_bytes // (1024*1024)} MB, política: {storage_policy}")

    def _get_total_storage_used(self):
        """Retorna o tamanho total (em bytes) de todos os vídeos .mp4 no diretório de saída."""
        pattern = os.path.join(self.output_dir, "*.mp4")
        files = glob.glob(pattern)
        total = 0
        for f in files:
            try:
                total += os.path.getsize(f)
            except OSError as e:
                logger.warning(f"Não foi possível obter tamanho de {f}: {e}")
        return total

    def _enforce_storage_policy(self):
        """Aplica a política de armazenamento após um vídeo ser salvo."""
        if self.max_storage_bytes is None:
            return

        total = self._get_total_storage_used()
        if total <= self.max_storage_bytes:
            # Se estava excedido e agora voltou ao normal, limpa flag
            if self._storage_exceeded:
                self._storage_exceeded = False
                logger.info("Uso de armazenamento voltou ao normal. Novas gravações permitidas.")
            return

        if self.storage_policy == 'delete_oldest':
            # Apaga vídeos mais antigos até ficar abaixo do limite
            pattern = os.path.join(self.output_dir, "*.mp4")
            files = glob.glob(pattern)
            # Ordena por data de modificação (mais antigo primeiro)
            files.sort(key=lambda f: os.path.getmtime(f))
            logger.warning(f"Limite de armazenamento excedido: {total/(1024*1024):.2f} MB > {self.max_storage_bytes/(1024*1024):.2f} MB. Deletando vídeos antigos...")
            while self._get_total_storage_used() > self.max_storage_bytes and files:
                oldest = files.pop(0)
                try:
                    os.remove(oldest)
                    logger.info(f"Removido vídeo antigo: {oldest}")
                except OSError as e:
                    logger.error(f"Erro ao remover {oldest}: {e}")
            # Após deleções, se ainda excedido (ex.: arquivo muito grande), loga
            if self._get_total_storage_used() > self.max_storage_bytes:
                logger.error("Mesmo após deletar vídeos antigos, o limite ainda está excedido. Considere aumentar o limite ou verificar arquivos grandes.")
            else:
                self._storage_exceeded = False
        elif self.storage_policy == 'stop':
            self._storage_exceeded = True
            logger.error(f"Limite de armazenamento excedido: {total/(1024*1024):.2f} MB > {self.max_storage_bytes/(1024*1024):.2f} MB. Novas gravações serão bloqueadas até que espaço seja liberado.")

    def _check_storage_limit(self):
        """Verifica se é permitido iniciar uma nova gravação com base no limite e política."""
        if self.max_storage_bytes is None:
            return True
        if self.storage_policy == 'delete_oldest':
            # Sempre permite gravar, pois a limpeza será feita após cada vídeo
            return True
        elif self.storage_policy == 'stop':
            if self._storage_exceeded:
                total = self._get_total_storage_used()
                if total <= self.max_storage_bytes:
                    self._storage_exceeded = False
                    logger.info("Limite de armazenamento OK. Retomando gravações.")
                    return True
                else:
                    logger.warning("Limite de armazenamento excedido. Gravação não iniciada.")
                    return False
            return True

    def start_recording(self):
        with self.lock:
            if not self.recording:
                if not self._check_storage_limit():
                    return False
                self.recording = True
                start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.filename = os.path.join(self.output_dir, f"clip_{start_timestamp}.mp4")
                self.end_timestamp = None
                logger.debug(f"Iniciando gravação: {self.filename} (buffer: {len(self.frame_buffer)} frames)")
                buffer_copy = list(self.frame_buffer)
                for bframe in buffer_copy:
                    self.frame_queue.put(bframe)
                return True
        return False

    def add_frame(self, frame):
        with self.lock:
            self.frame_buffer.append(frame.copy())
            if self.recording:
                if self.writer is None:
                    h, w = frame.shape[:2]
                    self.writer = cv2.VideoWriter(self.filename, self.fourcc, self.fps, (w, h))
                self.frame_queue.put(frame.copy())

    def stop_recording(self):
        with self.lock:
            if self.recording:
                self.recording = False
                self.end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                logger.debug("Parando gravação")
                self.frame_queue.put(None)

    def shutdown(self):
        """Encerra a thread do gravador aguardando a conclusão de todas as operações."""
        if not self._shutdown_started:
            self._shutdown_started = True
            self.frame_queue.put(self._shutdown_sentinel)
            self.thread.join()
            logger.debug("Recorder encerrado.")

    def _record_worker(self):
        while True:
            item = self.frame_queue.get()
            if item is None:
                if self.writer:
                    self.writer.release()
                    self.writer = None
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

                    # Aplica política de armazenamento após finalizar o vídeo
                    self._enforce_storage_policy()

                    if self.on_video_finished:
                        self.on_video_finished(final_filename)
                continue

            elif item is self._shutdown_sentinel:
                if self.writer:
                    self.writer.release()
                    self.writer = None
                break

            else:
                if self.writer:
                    self.writer.write(item)