import time
import threading
import logging
from datetime import datetime
import cv2  # NOVO: import necessário para desenho

logger = logging.getLogger(__name__)

class MotionRecorderApp:
    def __init__(self, source, motion_detector, recorder, uploader=None,
                 cooldown_sec=2.0, min_motion_frames=5, stop_event=None,
                 show_preview=False):  # NOVO: parâmetro show_preview
        self.source = source
        self.detector = motion_detector
        self.recorder = recorder
        self.uploader = uploader
        self.cooldown = cooldown_sec
        self.min_motion_frames = min_motion_frames
        self.stop_event = stop_event or threading.Event()
        self.show_preview = show_preview  # NOVO

        if uploader:
            self.recorder.on_video_finished = self.uploader.upload

        self.motion_counter = 0
        self.no_motion_start = None
        self.recording = False

        # NOVO: para controle da janela de preview
        self.preview_window_name = "Motion Recorder Preview"

    def run(self):
        logger.info("Iniciando monitoramento. Pressione Ctrl+C para parar.")
        if self.show_preview:
            logger.info("Modo preview ativado. Pressione 'q' na janela para sair.")
            cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)

        while not self.stop_event.is_set():
            frame = self.source.get_frame()
            if frame is None:
                logger.warning("Fonte não forneceu frame (fim da stream?)")
                time.sleep(0.5)
                continue

            # NOVO: detector agora retorna contornos
            contours = self.detector.detect_with_contours(frame)
            motion = len(contours) > 0

            # Lógica de estado (inalterada)
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

            # NOVO: desenha preview se ativado
            if self.show_preview:
                self._draw_preview(frame, motion, contours)
                cv2.imshow(self.preview_window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Comando 'q' recebido na janela de preview. Encerrando...")
                    self.stop_event.set()
                    break

            self.stop_event.wait(0.01)

        logger.info("Parando aplicação...")
        if self.recording:
            self.recorder.stop_recording()
        self.source.release()
        if self.show_preview:
            cv2.destroyAllWindows()
        logger.info("Recursos liberados. Programa encerrado.")

    # método auxiliar para desenhar informações no frame
    def _draw_preview(self, frame, motion, contours):
        # Desenha contornos verdes onde há movimento
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        # Status da gravação
        status = "GRAVANDO" if self.recording else "ESPERANDO"
        color = (0, 0, 255) if self.recording else (255, 255, 255)  # vermelho se gravando
        cv2.putText(frame, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Informação de movimento
        motion_text = f"Movimento: {'SIM' if motion else 'NAO'} ({len(contours)} areas)"
        cv2.putText(frame, motion_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Contador de frames com movimento
        cv2.putText(frame, f"Contador: {self.motion_counter}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Instrução
        cv2.putText(frame, "Pressione 'q' para sair", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)