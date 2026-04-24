#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# ------------------------------------------------------------
# SUPRESSÃO DE MENSAGENS DO QT (DEVE VIR ANTES DE QUALQUER IMPORTAÇÃO DO OPENCV)
# ------------------------------------------------------------

class FilteredStderr:
    """Redireciona stderr, filtrando mensagens indesejadas."""
    def __init__(self, original_stderr, filter_strings):
        self.original_stderr = original_stderr
        self.filter_strings = filter_strings

    def write(self, message):
        if any(pattern in message for pattern in self.filter_strings):
            return
        self.original_stderr.write(message)

    def flush(self):
        self.original_stderr.flush()

    def __getattr__(self, attr):
        return getattr(self.original_stderr, attr)

def suppress_qt_thread_warnings():
    """Ativa o filtro de stderr para ignorar avisos de thread do Qt."""
    filter_patterns = [
        "QObject::moveToThread",
        "Cannot move to target thread"
    ]
    sys.stderr = FilteredStderr(sys.stderr, filter_patterns)

suppress_qt_thread_warnings()
os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.warning=false"
os.environ["QT_FATAL_WARNINGS"] = "0"

# ------------------------------------------------------------
# IMPORTAÇÕES DO PROJETO
# ------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import logging
import signal
import threading
import time
from pprint import pformat

from sources import CameraSource, RTSPSource, DirectorySource, VideoFileSource, ThreadedFrameSource
from detector import MotionDetector
from recorder import Recorder
from uploader import Uploader
from app import MotionRecorderApp
from config import load_config, load_roi

# ------------------------------------------------------------
# CONFIGURAÇÃO DE LOGGING
# ------------------------------------------------------------
def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def log_final_configuration(args, final_fps, roi_polygons):
    logging.info("=" * 60)
    logging.info("CONFIGURAÇÕES FINAIS DA EXECUÇÃO")
    logging.info("=" * 60)
    logging.info(f"Fonte: {args.source_type} -> {args.source_param}")
    if args.source_type == 'camera':
        logging.info(f"Resolução: {args.width}x{args.height}")
        if args.camera_codec:
            logging.info(f"Codec da câmera: {args.camera_codec}")
        else:
            logging.info("Codec da câmera: nativo")
    logging.info(f"FPS de gravação: {final_fps:.2f}")
    logging.info(f"Diretório de saída: {args.output_dir}")
    logging.info(f"Método de detecção: {args.detection_method}")
    logging.info(f"Threshold: {args.threshold} | Área mínima: {args.min_area}")
    logging.info(f"Pré-gravação: {args.pre_record}s | Cooldown: {args.cooldown}s")
    logging.info(f"Frames mínimos para iniciar: {args.min_motion_frames}")
    if args.max_storage_mb > 0:
        logging.info(f"Limite de armazenamento: {args.max_storage_mb} MB | Política: {args.storage_policy}")
    else:
        logging.info("Limite de armazenamento: desativado")
    logging.info(f"Upload: {'Sim' if args.server_url and not args.no_upload else 'Não'}")
    if args.server_url and not args.no_upload:
        logging.info(f"  URL: {args.server_url} | Remover após upload: {args.remove_after_upload}")
    logging.info(f"Preview: {'Sim' if args.show_preview else 'Não'}")
    if roi_polygons:
        logging.info(f"ROIs carregadas: {len(roi_polygons)} polígono(s)")
    logging.info(f"Modo debug: {'Sim' if args.debug else 'Não'}")
    logging.info("=" * 60)

def measure_processing_fps(source, detector, num_frames=50, warmup_frames=10):
    """
    Mede a taxa de processamento real capturando alguns frames
    e aplicando a detecção. Retorna o FPS máximo sustentável.
    """
    logger = logging.getLogger(__name__)
    # Descarta frames iniciais para estabilizar a câmera
    for _ in range(warmup_frames):
        frame = source.get_frame()
        if frame is None:
            logger.warning("Sem frames durante aquecimento do benchmark.")
            return None

    start = time.perf_counter()
    processed = 0
    for _ in range(num_frames):
        frame = source.get_frame()
        if frame is None:
            break
        # Aplica a detecção de movimento (usando o método completo para ser realista)
        _ = detector.detect_with_contours(frame)
        processed += 1
    end = time.perf_counter()

    if processed == 0:
        return None
    elapsed = end - start
    fps = processed / elapsed
    logger.info(f"Benchmark: {processed} frames processados em {elapsed:.2f}s -> {fps:.2f} fps")
    return fps

# ------------------------------------------------------------
# FUNÇÃO PRINCIPAL
# ------------------------------------------------------------
def main():
    prelim_parser = argparse.ArgumentParser(add_help=False)
    prelim_parser.add_argument("--config", type=str, default="config.json")
    prelim_parser.add_argument("--debug", action="store_true")
    prelim_args, remaining_args = prelim_parser.parse_known_args()

    config = load_config(prelim_args.config)
    config = {k: v for k, v in config.items() if v is not None}

    debug_mode = prelim_args.debug or config.get('debug', False)
    setup_logging(debug_mode)
    logging.debug(f"Configurações carregadas do JSON: {pformat(config)}")

    parser = argparse.ArgumentParser(description="Sistema de gravação por detecção de movimento")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--source-type", choices=['camera', 'rtsp', 'dir', 'video'], default='camera')
    parser.add_argument("--source-param", type=str, default="0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--force-fps", action="store_true")
    parser.add_argument("--output-dir", type=str, default="../videos")
    parser.add_argument("--detection-method", choices=['diff', 'mog2'], default='mog2')
    parser.add_argument("--threshold", type=int, default=25)
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--pre-record", type=float, default=2.0)
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--min-motion-frames", type=int, default=5)
    parser.add_argument("--server-url", type=str, default=None)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--remove-after-upload", action="store_true")
    parser.add_argument("--show-preview", action="store_true")
    parser.add_argument("--roi-json", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-storage-mb", type=int, default=0)
    parser.add_argument("--storage-policy", choices=['stop', 'delete_oldest'], default='stop')
    parser.add_argument("--camera-codec", type=str, default=None,
                        help="Codec da câmera (ex: MJPG, YUYV, H264). Se não informado, usa o nativo.")

    parser.set_defaults(**config)
    args = parser.parse_args(remaining_args)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    logging.debug(f"Argumentos finais: {pformat(vars(args))}")

    # ROIs
    roi_polygons = None
    if args.roi_json:
        roi_polygons = load_roi(args.roi_json)
        if roi_polygons:
            logging.info(f"Carregadas {len(roi_polygons)} ROIs do arquivo {args.roi_json}")

    # Uploader
    uploader = None
    if args.server_url and not args.no_upload:
        try:
            import requests
        except ImportError:
            logging.error("Biblioteca 'requests' não instalada.")
            sys.exit(1)
        uploader = Uploader(args.server_url, remove_after_upload=args.remove_after_upload)

    # Fonte de vídeo (bruta)
    raw_source = None
    if args.source_type == 'camera':
        try:
            device = int(args.source_param)
        except ValueError:
            device = args.source_param
        raw_source = CameraSource(device, width=args.width, height=args.height,
                                  fps=args.fps, codec=args.camera_codec)
    elif args.source_type == 'rtsp':
        raw_source = RTSPSource(args.source_param)
    elif args.source_type == 'dir':
        raw_source = DirectorySource(args.source_param)
    elif args.source_type == 'video':
        raw_source = VideoFileSource(args.source_param)
    else:
        logging.error("Tipo de fonte inválido")
        sys.exit(1)

    # Aplica ThreadedFrameSource APENAS para fontes ao vivo (evita aceleração em arquivos)
    if raw_source.is_live:
        source = ThreadedFrameSource(raw_source, timeout_sec=2.0)
        logging.info("ThreadedFrameSource ativado para fonte ao vivo")
    else:
        source = raw_source
        logging.info("Usando fonte direta (sem thread) para arquivo/diretório")

    # Detector
    detector = MotionDetector(
        method=args.detection_method,
        threshold=args.threshold,
        min_area=args.min_area,
        roi_polygons_normalized=roi_polygons
    )

    # Determinação do FPS
    if args.fps is not None:
        final_fps = args.fps
        logging.info(f"Usando FPS definido pelo usuário: {final_fps}")
    else:
        native_fps = None
        if hasattr(source, 'get_fps'):
            native_fps = source.get_fps()
        if native_fps and native_fps > 0:
            final_fps = native_fps
            logging.info(f"Usando FPS nativo da fonte: {final_fps:.2f}")
        else:
            final_fps = 20
            logging.warning(f"Não foi possível obter FPS nativo. Usando fallback: {final_fps}")

    if args.force_fps and args.fps is not None:
        final_fps = args.fps
        logging.info(f"Forçando FPS (--force-fps): {final_fps}")

    # Medição de capacidade de processamento (apenas para fontes ao vivo e se FPS foi definido)
    if source.is_live and args.fps is not None and not args.force_fps:
        logging.info("Medindo capacidade de processamento...")
        measured_fps = measure_processing_fps(source, detector, num_frames=50, warmup_frames=10)
        if measured_fps and measured_fps < final_fps:
            logging.warning(
                f"Hardware sustentou apenas {measured_fps:.2f} fps. "
                f"Reduzindo FPS de gravação de {final_fps:.2f} para {measured_fps:.2f}."
            )
            final_fps = measured_fps
        elif measured_fps:
            logging.info(f"Hardware suporta o FPS desejado ({final_fps:.2f} fps).")

    # Gravador
    max_storage_bytes = args.max_storage_mb * 1024 * 1024 if args.max_storage_mb > 0 else None
    recorder = Recorder(
        output_dir=args.output_dir,
        fps=final_fps,
        pre_record_seconds=args.pre_record,
        max_storage_bytes=max_storage_bytes,
        storage_policy=args.storage_policy
    )

    log_final_configuration(args, final_fps, roi_polygons)

    # Sinais
    stop_event = threading.Event()
    def signal_handler(signum, frame):
        logging.info(f"Sinal {signum} recebido, encerrando...")
        stop_event.set()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Aplicação
    app = MotionRecorderApp(
        source=source,
        motion_detector=detector,
        recorder=recorder,
        uploader=uploader,
        cooldown_sec=args.cooldown,
        min_motion_frames=args.min_motion_frames,
        stop_event=stop_event,
        show_preview=args.show_preview,
        roi_polygons_normalized=roi_polygons
    )

    try:
        app.run()
    finally:
        if recorder is not None and hasattr(recorder, 'shutdown'):
            recorder.shutdown()

if __name__ == "__main__":
    main()