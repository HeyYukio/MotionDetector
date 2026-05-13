#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# ------------------------------------------------------------
# SUPRESSÃO DE MENSAGENS DO QT (ANTES DE QUALQUER IMPORTAÇÃO DO OPENCV)
# ------------------------------------------------------------
class FilteredStderr:
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
    filter_patterns = ["QObject::moveToThread", "Cannot move to target thread"]
    sys.stderr = FilteredStderr(sys.stderr, filter_patterns)

suppress_qt_thread_warnings()
os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.warning=false"
os.environ["QT_FATAL_WARNINGS"] = "0"

# ------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import logging
import signal
import threading
from pprint import pformat

from sources import CameraSource, RTSPSource, DirectorySource, VideoFileSource, ThreadedFrameSource
from detector import MotionDetector
from recorder import Recorder
from uploader import Uploader
from app import MotionRecorderApp
from config import load_config, load_roi

# ------------------------------------------------------------
def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def log_final_configuration(args, final_fps, roi_polygons, debug_record, container_format):
    logging.info("=" * 60)
    logging.info("CONFIGURAÇÕES FINAIS DA EXECUÇÃO")
    logging.info("=" * 60)
    logging.info(f"Fonte: {args.source_type} -> {args.source_param}")
    if args.source_type == 'camera':
        logging.info(f"Resolução: {args.width}x{args.height}")
        logging.info(f"Codec da câmera: {args.camera_codec or 'nativo'}")
    logging.info(f"FPS de gravação: {final_fps:.2f}")
    logging.info(f"Formato do vídeo: {container_format}")
    logging.info(f"Diretório de saída: {args.output_dir}")
    logging.info(f"Método de detecção: {args.detection_method}")
    logging.info(f"Threshold: {args.threshold} | Área mínima: {args.min_area}")
    logging.info(f"Pré-gravação: {args.pre_record}s | Cooldown: {args.cooldown}s")
    logging.info(f"Frames mínimos: {args.min_motion_frames}")
    logging.info(f"ID equipamento: {args.equipment_id or '0000'}")
    if args.max_storage_mb > 0:
        logging.info(f"Armazenamento máx: {args.max_storage_mb} MB | Política: {args.storage_policy}")
    else:
        logging.info("Armazenamento: ilimitado")
    logging.info(f"Upload: {'Sim' if args.server_url and not args.no_upload else 'Não'}")
    logging.info(f"Preview: {'Sim' if args.show_preview else 'Não'}")
    if roi_polygons:
        logging.info(f"ROIs: {len(roi_polygons)} polígono(s)")
    logging.info(f"Modo debug: {'Sim' if args.debug else 'Não'}")
    logging.info(f"Gravação contínua (debug): {'Sim' if debug_record else 'Não'}")
    logging.info("=" * 60)

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

    parser = argparse.ArgumentParser(description="Sistema de gravação por detecção de movimento")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--source-type", choices=['camera','rtsp','dir','video'], default='camera')
    parser.add_argument("--source-param", type=str, default="0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output-dir", type=str, default="../videos")
    parser.add_argument("--detection-method", choices=['diff','mog2'], default='mog2')
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
    parser.add_argument("--storage-policy", choices=['stop','delete_oldest'], default='stop')
    parser.add_argument("--camera-codec", type=str, default=None)
    parser.add_argument("--equipment-id", type=str, default=None)
    parser.add_argument("--debug-record", action="store_true")
    parser.add_argument("--format", choices=['mp4','mkv'], default='mp4',
                        help="Formato do vídeo de saída (padrão: mp4)")

    parser.set_defaults(**config)
    args = parser.parse_args(remaining_args)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    logging.debug(f"Argumentos finais: {pformat(vars(args))}")

    equipment_id = args.equipment_id or "0000"
    roi_polygons = load_roi(args.roi_json) if args.roi_json else None

    uploader = None
    if args.server_url and not args.no_upload:
        try:
            import requests
        except ImportError:
            logging.error("Biblioteca 'requests' não instalada.")
            sys.exit(1)
        uploader = Uploader(args.server_url, remove_after_upload=args.remove_after_upload)

    # Fonte bruta
    raw_source = None
    if args.source_type == 'camera':
        try:
            device = int(args.source_param)
        except ValueError:
            device = args.source_param
        raw_source = CameraSource(device, width=args.width, height=args.height, codec=args.camera_codec)
    elif args.source_type == 'rtsp':
        raw_source = RTSPSource(args.source_param)
    elif args.source_type == 'dir':
        raw_source = DirectorySource(args.source_param)
    elif args.source_type == 'video':
        raw_source = VideoFileSource(args.source_param)
    else:
        logging.error("Tipo de fonte inválido")
        sys.exit(1)

    if raw_source.is_live:
        source = ThreadedFrameSource(raw_source, timeout_sec=2.0)
        logging.info("ThreadedFrameSource ativado")
    else:
        source = raw_source
        logging.info("Fonte direta (sem thread)")

    detector = MotionDetector(method=args.detection_method, threshold=args.threshold,
                              min_area=args.min_area, roi_polygons_normalized=roi_polygons)

    # FPS
    if raw_source.is_live:
        native_fps = getattr(source, 'get_fps', lambda: None)()
        final_fps = native_fps if (native_fps and native_fps > 0) else 20
    else:
        native_fps = getattr(source, 'get_fps', lambda: None)()
        final_fps = native_fps if (native_fps and native_fps > 0) else 20

    max_storage_bytes = args.max_storage_mb * 1024 * 1024 if args.max_storage_mb > 0 else None
    use_mkv = (args.format == 'mkv')

    # Gravador principal (clipes de movimento)
    recorder = Recorder(output_dir=args.output_dir, fps=final_fps,
                        pre_record_seconds=args.pre_record,
                        max_storage_bytes=max_storage_bytes,
                        storage_policy=args.storage_policy,
                        equipment_id=equipment_id,
                        filename_prefix="clip",
                        use_mkv=use_mkv,
                        protected=False)
    if uploader:
        recorder.set_uploader(uploader)

    # Gravador debug contínuo (protegido)
    debug_recorder = None
    if args.debug_record:
        debug_recorder = Recorder(output_dir=args.output_dir, fps=final_fps,
                                  pre_record_seconds=0,
                                  max_storage_bytes=max_storage_bytes,
                                  storage_policy=args.storage_policy,
                                  equipment_id=equipment_id,
                                  filename_prefix="debug",
                                  use_mkv=use_mkv,
                                  protected=True)

    log_final_configuration(args, final_fps, roi_polygons, args.debug_record, args.format)

    stop_event = threading.Event()
    def signal_handler(signum, frame):
        logging.info(f"Sinal {signum} recebido, encerrando.")
        stop_event.set()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app = MotionRecorderApp(source=source, motion_detector=detector,
                            recorder=recorder, debug_recorder=debug_recorder,
                            uploader=uploader, cooldown_sec=args.cooldown,
                            min_motion_frames=args.min_motion_frames,
                            stop_event=stop_event, show_preview=args.show_preview,
                            roi_polygons_normalized=roi_polygons)

    try:
        app.run()
    finally:
        if recorder:
            recorder.shutdown()
        if debug_recorder:
            debug_recorder.shutdown()

if __name__ == "__main__":
    main()