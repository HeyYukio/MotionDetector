import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MotionDetector:
    def __init__(self, method='mog2', threshold=25, min_area=500, blur_size=(21,21),
                 history=500, var_threshold=16, detect_shadows=True,
                 roi_polygons_normalized=None):
        """
        method: 'diff' para diferença entre frames, 'mog2' para subtração de fundo MOG2
        roi_polygons_normalized: lista de polígonos no formato normalizado (0-1)
            cada polígono é uma lista de pontos (x, y) normalizados.
        """
        self.method = method
        self.threshold = threshold
        self.min_area = min_area
        self.blur_size = blur_size
        self.prev_gray = None

        # ROIs (coordenadas normalizadas)
        self.roi_polygons_normalized = roi_polygons_normalized if roi_polygons_normalized else []
        # Cache para polígonos absolutos (calculados quando o primeiro frame chega)
        self.roi_polygons_absolute = []
        self.frame_shape = None  # (height, width)

        if method == 'mog2':
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=var_threshold, detectShadows=detect_shadows)
            logger.info("Detector MOG2 inicializado")
        else:
            logger.info("Detector por diferença de frames inicializado")

        if self.roi_polygons_normalized:
            logger.info(f"Detector configurado com {len(self.roi_polygons_normalized)} ROIs")

    def _update_absolute_polygons(self, frame_shape):
        """Converte polígonos normalizados para absolutos com base no shape do frame."""
        if self.frame_shape == frame_shape:
            return
        self.frame_shape = frame_shape
        h, w = frame_shape[:2]
        self.roi_polygons_absolute = []
        for poly_norm in self.roi_polygons_normalized:
            abs_poly = np.array([(int(x * w), int(y * h)) for (x, y) in poly_norm], dtype=np.int32)
            self.roi_polygons_absolute.append(abs_poly)

    def _point_in_rois(self, point):
        """Verifica se um ponto (x, y) está dentro de alguma ROI."""
        if not self.roi_polygons_absolute:
            return True  # Sem ROIs, tudo é considerado
        for poly in self.roi_polygons_absolute:
            if cv2.pointPolygonTest(poly, point, False) >= 0:
                return True
        return False

    def detect(self, frame):
        """Retorna True se movimento for detectado no frame (considerando ROIs)."""
        contours = self.detect_with_contours(frame)
        return len(contours) > 0

    def detect_with_contours(self, frame):
        """
        Retorna uma lista de contornos de movimento que estão dentro de alguma ROI.
        Cada contorno é um array de pontos.
        """
        if frame is None:
            return []

        # Atualiza polígonos absolutos se necessário
        self._update_absolute_polygons(frame.shape)

        if self.method == 'mog2':
            all_contours = self._detect_mog2_contours(frame)
        else:
            all_contours = self._detect_diff_contours(frame)

        # Filtra contornos que estão dentro das ROIs
        valid_contours = []
        for contour in all_contours:
            # Calcula o centróide do contorno
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if self._point_in_rois((cx, cy)):
                    valid_contours.append(contour)
            else:
                # Se não for possível calcular o centróide, ignora
                pass
        return valid_contours

    def _detect_diff_contours(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_size, 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return []

        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.prev_gray = gray

        # Filtra por área mínima
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_area]
        return valid_contours

    def _detect_mog2_contours(self, frame):
        fgmask = self.bg_subtractor.apply(frame)
        if fgmask is not None:
            _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
            fgmask = cv2.erode(fgmask, None, iterations=1)
            fgmask = cv2.dilate(fgmask, None, iterations=2)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_area]
            return valid_contours
        return []