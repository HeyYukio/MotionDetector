import cv2
import os
import time
import logging

logger = logging.getLogger(__name__)

class FrameSource:
    def get_frame(self):
        raise NotImplementedError
    def release(self):
        pass

class CameraSource(FrameSource):
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        logger.info(f"CameraSource iniciada: {src}")

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()
        logger.info("CameraSource liberada")

class RTSPSource(FrameSource):
    def __init__(self, url, timeout=5, max_reconnect=10):
        self.url = url
        self.timeout = timeout
        self.max_reconnect = max_reconnect
        self.cap = None
        self.reconnect_count = 0
        self._connect()

    def _connect(self):
        if self.cap is not None:
            self.cap.release()
        logger.info(f"Conectando ao RTSP: {self.url}")
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            logger.warning("Falha na conexão inicial")

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            if self.reconnect_count >= self.max_reconnect:
                logger.error("Número máximo de tentativas de reconexão atingido")
                return None
            wait = min(2 ** self.reconnect_count, 60)
            logger.info(f"Tentando reconectar em {wait}s...")
            time.sleep(wait)
            self._connect()
            self.reconnect_count += 1
            return None  # tenta novamente no próximo ciclo

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Frame não recebido, possível queda de stream")
            self.cap.release()
            self.cap = None
            return None
        self.reconnect_count = 0  # reset ao conseguir frame
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
        logger.info("RTSPSource liberada")

class DirectorySource(FrameSource):
    def __init__(self, path, ext=('.jpg', '.png')):
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

class VideoFileSource(FrameSource):
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        logger.info(f"VideoFileSource: {path}")

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()