import time
import threading
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class MotionRecorderApp:
    def __init__(self, source, motion_detector, recorder, uploader=None,
                 cooldown_sec=2.0, min_motion_frames=5, stop_event=None,
                 show_preview=False, roi_polygons_normalized=None):
        self.source = source
        self.detector = motion_detector
        self.recorder = recorder
        self.uploader = uploader
        self.cooldown = cooldown_sec
        self.min_motion_frames = min_motion_frames
        self.stop_event = stop_event or threading.Event()
        self.show_preview = show_preview
        self.roi_polygons_normalized = roi_polygons_normalized or []

        if uploader:
            self.recorder.on_video_finished = self.uploader.upload

        self.motion_counter = 0
        self.no_motion_start = None
        self.recording = False

        self.preview_window_name = "Motion Recorder Preview"
        self.roi_polygons_absolute = []
        self.frame_shape = None

    def _update_roi_absolute(self, frame_shape):
        if self.frame_shape == frame_shape:
            return
        self.frame_shape = frame_shape
        h, w = frame_shape[:2]
        self.roi_polygons_absolute = []
        for poly_norm in self.roi_polygons_normalized:
            abs_poly = np.array([(int(x * w), int(y * h)) for (x, y) in poly_norm], dtype=np.int32)
            self.roi_polygons_absolute.append(abs_poly)

    def run(self):
        logger.info("Iniciando monitoramento. Pressione Ctrl+C para parar.")
        if self.show_preview:
            logger.info("Modo preview ativado. Pressione 'q' na janela para sair.")
            cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)

        target_fps = self.recorder.fps
        if target_fps <= 0:
            target_fps = 20
            logger.warning(f"FPS inválido, usando fallback {target_fps}")

        is_live_source = getattr(self.source, 'is_live', False)

        # =====================================================================
        # INICIALIZAÇÃO DAS VARIÁVEIS DE CONTROLE (AO VIVO OU ARQUIVO)
        # =====================================================================
        if is_live_source:
            frame_interval = 1.0 / target_fps
            logger.info(
                f"Fonte ao vivo – amostrador com preenchimento ativo: "
                f"{target_fps:.2f} fps ({frame_interval*1000:.2f} ms)"
            )
            next_capture_time = time.perf_counter()
            last_processed_frame = None
        else:
            # Fonte de arquivo: reamostrador temporal (duplica/descarta conforme necessário)
            source_fps = None
            if hasattr(self.source, 'get_fps'):
                source_fps = self.source.get_fps()
            total_frames = None
            if hasattr(self.source, 'get_frame_count'):
                total_frames = self.source.get_frame_count()

            if source_fps and source_fps > 0:
                self.source_fps = source_fps
                self.ratio = target_fps / source_fps
                self.output_accum = 0.0
                logger.info(
                    f"Fonte de arquivo com FPS nativo {source_fps:.2f}. "
                    f"Gravando a {target_fps:.2f} fps -> reamostrando (razão {self.ratio:.3f})."
                )
                if total_frames:
                    original_duration = total_frames / source_fps
                    output_frames = int(total_frames * self.ratio)
                    output_duration = output_frames / target_fps
                    logger.info(
                        f"Vídeo original: {total_frames} frames, {original_duration:.2f}s. "
                        f"Saída: ~{output_frames} frames, {output_duration:.2f}s."
                    )
            else:
                self.source_fps = None
                logger.warning(
                    "Não foi possível obter o FPS nativo da fonte de arquivo. "
                    "Todos os frames serão processados sem reamostragem. "
                    "A duração do vídeo resultante pode ser diferente da original."
                )

        # =====================================================================
        # LOOP PRINCIPAL
        # =====================================================================
        while not self.stop_event.is_set():

            # ----------------------------------------------------------------
            # TRATAMENTO PARA FONTES AO VIVO
            # ----------------------------------------------------------------
            if is_live_source:
                now = time.perf_counter()
                frame = self.source.get_frame()
                if frame is None:
                    if not self.source.is_live:
                        logger.info("Fim da fonte de vídeo. Encerrando...")
                        break
                    self.stop_event.wait(0.05)
                    continue

                # Ainda não é hora de emitir um quadro -> espera ativa curta
                if now < next_capture_time:
                    time.sleep(0.001)
                    continue

                # Quantos intervalos de frame_interval se passaram desde a última emissão?
                elapsed_intervals = int((now - next_capture_time) / frame_interval) + 1
                # Proteção contra atraso extremo
                if elapsed_intervals > 100:
                    elapsed_intervals = 100
                    next_capture_time = now

                # Preenche os (elapsed_intervals - 1) intervalos anteriores com duplicatas
                for _ in range(elapsed_intervals - 1):
                    if last_processed_frame is not None and self.recording:
                        self.recorder.add_frame(last_processed_frame.copy())
                    next_capture_time += frame_interval

                # Processa o frame ATUAL como o último intervalo
                if self.show_preview:
                    self._update_roi_absolute(frame.shape)

                contours = self.detector.detect_with_contours(frame)
                motion = len(contours) > 0

                if motion:
                    self.motion_counter += 1
                    self.no_motion_start = None
                    if not self.recording and self.motion_counter >= self.min_motion_frames:
                        self.recording = True
                        self.recorder.start_recording()
                        logger.info("Movimento detectado - gravando")
                else:
                    if self.recording:
                        if self.no_motion_start is None:
                            self.no_motion_start = time.time()
                        elif time.time() - self.no_motion_start > self.cooldown:
                            self.recording = False
                            self.recorder.stop_recording()
                            logger.info("Sem movimento - gravação encerrada")
                    else:
                        self.motion_counter = max(0, self.motion_counter - 1)

                if self.recording:
                    self.recorder.add_frame(frame)

                last_processed_frame = frame.copy()
                next_capture_time += frame_interval

                if self.show_preview:
                    self._draw_preview(frame, motion, contours)
                    cv2.imshow(self.preview_window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Comando 'q' recebido. Encerrando...")
                        self.stop_event.set()
                        break

            # ----------------------------------------------------------------
            # TRATAMENTO PARA FONTES DE ARQUIVO
            # ----------------------------------------------------------------
            else:
                frame = self.source.get_frame()
                if frame is None:
                    logger.info("Fim da fonte de arquivo. Encerrando...")
                    break

                # Detecção de movimento (uma única vez por frame original)
                if self.show_preview:
                    self._update_roi_absolute(frame.shape)

                contours = self.detector.detect_with_contours(frame)
                motion = len(contours) > 0

                if motion:
                    self.motion_counter += 1
                    self.no_motion_start = None
                    if not self.recording and self.motion_counter >= self.min_motion_frames:
                        self.recording = True
                        self.recorder.start_recording()
                        logger.info("Movimento detectado - gravando")
                else:
                    if self.recording:
                        if self.no_motion_start is None:
                            self.no_motion_start = time.time()
                        elif time.time() - self.no_motion_start > self.cooldown:
                            self.recording = False
                            self.recorder.stop_recording()
                            logger.info("Sem movimento - gravação encerrada")
                    else:
                        self.motion_counter = max(0, self.motion_counter - 1)

                # Reamostragem temporal: quantas cópias deste frame enviar ao gravador?
                if self.source_fps is not None:
                    self.output_accum += self.ratio
                    num_copies = int(self.output_accum)
                    self.output_accum -= num_copies
                else:
                    num_copies = 1

                for _ in range(num_copies):
                    if self.recording:
                        self.recorder.add_frame(frame)

                if self.show_preview:
                    self._draw_preview(frame, motion, contours)
                    cv2.imshow(self.preview_window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Comando 'q' recebido. Encerrando...")
                        self.stop_event.set()
                        break

        logger.info("Parando aplicação...")
        if self.recording:
            self.recorder.stop_recording()
        self.source.release()
        if self.show_preview:
            cv2.destroyAllWindows()
        logger.info("Recursos liberados. Programa encerrado.")

    def _draw_preview(self, frame, motion, contours):
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        for poly in self.roi_polygons_absolute:
            cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 255), thickness=2)
        status = "GRAVANDO" if self.recording else "ESPERANDO"
        color = (0, 0, 255) if self.recording else (255, 255, 255)
        cv2.putText(frame, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        motion_text = f"Movimento: {'SIM' if motion else 'NAO'} ({len(contours)} areas)"
        cv2.putText(frame, motion_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Contador: {self.motion_counter}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        fps_text = f"FPS alvo: {self.recorder.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Pressione 'q' para sair", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)