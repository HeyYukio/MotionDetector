import cv2
import logging

logger = logging.getLogger(__name__)

class MotionDetector:
    def __init__(self, method='diff', threshold=25, min_area=500, blur_size=(21,21),
                 history=500, var_threshold=16, detect_shadows=True):
        """
        method: 'diff' para diferença entre frames, 'mog2' para subtração de fundo MOG2
        """
        self.method = method
        self.threshold = threshold
        self.min_area = min_area
        self.blur_size = blur_size
        self.prev_gray = None

        if method == 'mog2':
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=var_threshold, detectShadows=detect_shadows)
            logger.info("Detector MOG2 inicializado")
        else:
            logger.info("Detector por diferença de frames inicializado")

    def detect(self, frame):
        """Retorna True se movimento for detectado no frame."""
        contours = self.detect_with_contours(frame)
        return len(contours) > 0

    # método que retorna os contornos para desenho na tela
    def detect_with_contours(self, frame):
        """Retorna uma lista de contornos de movimento (cada contorno é um array de pontos)."""
        if self.method == 'mog2':
            return self._detect_mog2_contours(frame)
        else:
            return self._detect_diff_contours(frame)

    # extrai a lógica de contornos do método _detect_diff
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

    # extrai a lógica de contornos do método _detect_mog2
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