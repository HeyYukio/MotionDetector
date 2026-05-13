import time
import threading
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class MotionRecorderApp:
    def __init__(self, source, motion_detector, recorder, debug_recorder=None,
                 uploader=None, cooldown_sec=2.0, min_motion_frames=5, stop_event=None,
                 show_preview=False, roi_polygons_normalized=None):
        self.source = source
        self.detector = motion_detector
        self.recorder = recorder
        self.debug_recorder = debug_recorder
        self.uploader = uploader
        self.cooldown = cooldown_sec
        self.min_motion_frames = min_motion_frames
        self.stop_event = stop_event or threading.Event()
        self.show_preview = show_preview
        self.roi_polygons_normalized = roi_polygons_normalized or []

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
        cv2.putText(frame, f"FPS alvo: {self.recorder.fps:.1f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Pressione 'q' para sair", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)

    def run(self):
        logger.info("Iniciando monitoramento. Ctrl+C para parar.")
        if self.show_preview:
            cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)

        target_fps = self.recorder.fps
        if target_fps <= 0:
            target_fps = 20
        is_live = getattr(self.source, 'is_live', False)

        if not is_live:
            self._run_file_source(target_fps)
        else:
            self._run_live_source(target_fps)

        logger.info("Parando aplicação...")
        if self.recording:
            self.recorder.stop_recording()
        if self.debug_recorder and self.debug_recorder.recording:
            self.debug_recorder.stop_recording()
        self.source.release()
        if self.show_preview:
            cv2.destroyAllWindows()
        logger.info("Recursos liberados.")

    def _run_live_source(self, target_fps):
        frame_interval = 1.0 / target_fps
        logger.info(f"Gravação: {target_fps:.2f} fps ({frame_interval*1000:.2f} ms)")
        preview_lock = threading.Lock()
        latest_detection_frame = None
        latest_contours = []
        latest_motion_flag = False

        if self.debug_recorder:
            self.debug_recorder.start_recording()
            logger.info("Gravação debug contínua iniciada.")

        def recording_worker():
            next_time = time.perf_counter()
            while not self.stop_event.is_set():
                frame, timestamp = self.source.get_frame()
                if frame is None:
                    if not self.source.is_live:
                        self.stop_event.set()
                        break
                    time.sleep(0.05)
                    continue

                self.recorder.add_frame(frame, timestamp)
                if self.debug_recorder:
                    self.debug_recorder.add_frame(frame, timestamp)

                next_time += frame_interval
                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_time = time.perf_counter() + frame_interval

        def detection_worker():
            nonlocal latest_detection_frame, latest_contours, latest_motion_flag
            while not self.stop_event.is_set():
                frame, _ = self.source.get_frame()
                if frame is None:
                    if not self.source.is_live:
                        break
                    time.sleep(0.05)
                    continue

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

                if self.show_preview:
                    with preview_lock:
                        latest_detection_frame = frame.copy()
                        latest_contours = contours
                        latest_motion_flag = motion

        rec_thread = threading.Thread(target=recording_worker, daemon=False)
        det_thread = threading.Thread(target=detection_worker, daemon=False)
        rec_thread.start()
        det_thread.start()

        if self.show_preview:
            while not self.stop_event.is_set():
                with preview_lock:
                    if latest_detection_frame is not None:
                        display = latest_detection_frame.copy()
                        self._draw_preview(display, latest_motion_flag, latest_contours)
                        cv2.imshow(self.preview_window_name, display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("'q' pressionado, encerrando.")
                    self.stop_event.set()
                    break
        else:
            rec_thread.join()
            det_thread.join()

    def _run_file_source(self, target_fps):
        source_fps = getattr(self.source, 'get_fps', lambda: None)()
        total_frames = getattr(self.source, 'get_frame_count', lambda: None)()
        if source_fps and source_fps > 0:
            self.source_fps = source_fps
            self.ratio = target_fps / source_fps
            self.output_accum = 0.0
            logger.info(f"Arquivo: FPS nativo {source_fps:.2f}, reamostrando para {target_fps:.2f}")
        else:
            self.source_fps = None
            logger.warning("FPS nativo desconhecido, processando todos os frames.")

        if self.debug_recorder:
            self.debug_recorder.start_recording()

        while not self.stop_event.is_set():
            frame = self.source.get_frame()
            if frame is None:
                logger.info("Fim da fonte de arquivo.")
                break

            # Para arquivo, simulamos timestamp com time.time() (ou poderíamos usar um incremento)
            timestamp = time.time()

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

            if self.source_fps is not None:
                self.output_accum += self.ratio
                num_copies = int(self.output_accum)
                self.output_accum -= num_copies
            else:
                num_copies = 1

            for _ in range(num_copies):
                if self.recording:
                    self.recorder.add_frame(frame, timestamp)
                if self.debug_recorder:
                    self.debug_recorder.add_frame(frame, timestamp)

            if self.show_preview:
                self._draw_preview(frame, motion, contours)
                cv2.imshow(self.preview_window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("'q' pressionado, encerrando.")
                    self.stop_event.set()
                    break