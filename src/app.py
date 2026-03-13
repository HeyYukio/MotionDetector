import time
import threading
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MotionRecorderApp:
    def __init__(self, source, motion_detector, recorder, uploader=None,
                 cooldown_sec=2.0, min_motion_frames=5, stop_event=None):
        self.source = source
        self.detector = motion_detector
        self.recorder = recorder
        self.uploader = uploader
        self.cooldown = cooldown_sec
        self.min_motion_frames = min_motion_frames
        self.stop_event = stop_event or threading.Event()

        if uploader:
            self.recorder.on_video_finished = self.uploader.upload

        self.motion_counter = 0
        self.no_motion_start = None
        self.recording = False

    def run(self):
        logger.info("Iniciando monitoramento. Pressione Ctrl+C para parar.")
        while not self.stop_event.is_set():
            frame = self.source.get_frame()
            if frame is None:
                logger.warning("Fonte não forneceu frame (fim da stream?)")
                # Se for fonte finita (vídeo, diretório), podemos encerrar
                # Mas para RTSP/câmera, pode ser queda, então esperamos
                time.sleep(0.5)
                continue

            motion = self.detector.detect(frame)

            if motion:
                self.motion_counter += 1
                self.no_motion_start = None
                if not self.recording and self.motion_counter >= self.min_motion_frames:
                    self.recording = True
                    self.recorder.start_recording()
                    logger.info(f"Movimento detectado - gravando")
            else:
                if self.recording:
                    if self.no_motion_start is None:
                        self.no_motion_start = time.time()
                    elif time.time() - self.no_motion_start > self.cooldown:
                        self.recording = False
                        self.recorder.stop_recording()
                        logger.info(f"Sem movimento - gravação encerrada")
                else:
                    self.motion_counter = max(0, self.motion_counter - 1)

            if self.recording:
                self.recorder.add_frame(frame)

            # Aguarda um curto período ou até o evento de parada
            self.stop_event.wait(0.01)

        logger.info("Parando aplicação...")
        if self.recording:
            self.recorder.stop_recording()
        self.source.release()
        logger.info("Recursos liberados. Programa encerrado.")