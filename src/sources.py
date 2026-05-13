import cv2
import os
import time
import logging
import threading

logger = logging.getLogger(__name__)

class FrameSource:
    is_live = False
    def get_frame(self):
        raise NotImplementedError
    def release(self):
        pass
    def get_fps(self):
        return None
    def get_frame_count(self):
        return None

class CameraSource(FrameSource):
    is_live = True
    def __init__(self, src=0, width=640, height=480, fps=None, codec=None):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise IOError(f"Não foi possível abrir a câmera {src}")
        if codec:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            if self.cap.set(cv2.CAP_PROP_FOURCC, fourcc):
                logger.info(f"Codec da câmera definido: {codec}")
            else:
                logger.warning(f"Falha ao definir codec {codec}, usando padrão.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        logger.info(f"CameraSource: {src} {width}x{height} FPS nativo ~{self.cap.get(cv2.CAP_PROP_FPS):.2f}")

    def get_frame(self):
        ret, frame = self.cap.read()
        return (frame.copy(), time.time()) if ret else (None, None)

    def release(self):
        self.cap.release()
        logger.info("CameraSource liberada")

    def get_fps(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else None

class RTSPSource(FrameSource):
    is_live = True
    def __init__(self, url, timeout=5, max_reconnect=10):
        self.url = url
        self.timeout = timeout
        self.max_reconnect = max_reconnect
        self.cap = None
        self.reconnect_count = 0
        self._connect()

    def _connect(self):
        if self.cap:
            self.cap.release()
        logger.info(f"Conectando RTSP: {self.url}")
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            logger.warning("Falha na conexão inicial")

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            if self.reconnect_count >= self.max_reconnect:
                logger.error("Máximo de reconexões atingido")
                return None, None
            wait = min(2 ** self.reconnect_count, 60)
            logger.info(f"Tentando reconectar em {wait}s...")
            time.sleep(wait)
            self._connect()
            self.reconnect_count += 1
            return None, None
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Frame não recebido, queda de stream")
            self.cap.release()
            self.cap = None
            return None, None
        self.reconnect_count = 0
        return frame.copy(), time.time()

    def release(self):
        if self.cap:
            self.cap.release()
        logger.info("RTSPSource liberada")

    def get_fps(self):
        if self.cap and self.cap.isOpened():
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else None
        return None

class DirectorySource(FrameSource):
    is_live = False
    def __init__(self, path, ext=('.jpg','.png')):
        self.files = sorted([f for f in os.listdir(path) if f.lower().endswith(ext)])
        self.index = 0
        self.path = path
        logger.info(f"DirectorySource: {len(self.files)} imagens em {path}")

    def get_frame(self):
        if self.index >= len(self.files):
            return None
        img_path = os.path.join(self.path, self.files[self.index])
        self.index += 1
        return cv2.imread(img_path)

    def get_frame_count(self):
        return len(self.files)

class VideoFileSource(FrameSource):
    is_live = False
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.path = path
        fps = self.get_fps()
        logger.info(f"VideoFileSource: {path} @ {fps:.2f} fps" if fps else f"VideoFileSource: {path}")

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()

    def get_fps(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else None

    def get_frame_count(self):
        count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(count) if count > 0 else None

class ThreadedFrameSource(FrameSource):
    """
    Wrapper que captura em thread separada e retorna
    o frame mais recente + timestamp de captura (UTC).
    """
    def __init__(self, source, timeout_sec=2.0):
        self.source = source
        self.timeout_sec = timeout_sec
        self.latest_frame = None
        self.latest_timestamp = 0.0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()
        self.is_live = source.is_live
        logger.info("ThreadedFrameSource (latest-frame + timestamp) iniciado")

    def _capture_worker(self):
        while self.running:
            frame = self.source.get_frame()
            now = time.time()
            with self.lock:
                self.latest_frame = frame
                self.latest_timestamp = now
            if frame is None:
                break

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.latest_timestamp
            return None, None

    def release(self):
        self.running = False
        self.source.release()
        self.thread.join(timeout=2.0)
        logger.info("ThreadedFrameSource liberado")

    def get_fps(self):
        if hasattr(self.source, 'get_fps'):
            return self.source.get_fps()
        return None

    def get_frame_count(self):
        if hasattr(self.source, 'get_frame_count'):
            return self.source.get_frame_count()
        return None