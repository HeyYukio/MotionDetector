#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import logging
import signal
import threading
from types import SimpleNamespace

from sources import CameraSource, RTSPSource, DirectorySource, VideoFileSource, ThreadedFrameSource
from detector import MotionDetector
from recorder import Recorder
from uploader import Uploader
from app import MotionRecorderApp
from config import load_config, load_roi

def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description="Sistema de gravação por detecção de movimento")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Arquivo de configuração JSON (opcional)")
    parser.add_argument("--source-type", choices=['camera', 'rtsp', 'dir', 'video'],
                        default='camera', help="Tipo de fonte de entrada")
    parser.add_argument("--source-param", type=str, default="0",
                        help="Parâmetro da fonte: dispositivo, URL, diretório ou arquivo")
    parser.add_argument("--width", type=int, default=640, help="Largura do frame (câmera)")
    parser.add_argument("--height", type=int, default=480, help="Altura do frame (câmera)")
    parser.add_argument("--fps", type=int, default=20, help="FPS de gravação")
    parser.add_argument("--output-dir", type=str, default="../videos", help="Diretório para salvar vídeos")
    parser.add_argument("--detection-method", choices=['diff', 'mog2'], default='mog2',
                        help="Método de detecção de movimento")
    parser.add_argument("--threshold", type=int, default=25, help="Limiar de diferença (diff)")
    parser.add_argument("--min-area", type=int, default=500, help="Área mínima de movimento")
    parser.add_argument("--pre-record", type=float, default=2.0,
                        help="Segundos de pré-gravação antes do movimento")
    parser.add_argument("--cooldown", type=float, default=2.0,
                        help="Segundos sem movimento para parar gravação")
    parser.add_argument("--min-motion-frames", type=int, default=5,
                        help="Frames consecutivos com movimento para iniciar")
    parser.add_argument("--server-url", type=str, default=None,
                        help="URL do servidor para upload (se não informado, sem upload)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Desabilita upload mesmo se server_url estiver definido")
    parser.add_argument("--remove-after-upload", action="store_true",
                        help="Remove arquivo local após upload bem-sucedido")
    parser.add_argument("--show-preview", action="store_true",
                        help="Mostra janela de preview com detecção de movimento em tempo real")
    parser.add_argument("--roi-json", type=str, default=None,
                        help="Arquivo JSON contendo polígonos de áreas de interesse (ROIs)")
    parser.add_argument("--debug", action="store_true", help="Ativa logging de depuração")

    defaults = {action.dest: action.default for action in parser._actions 
                if action.default is not argparse.SUPPRESS}
    args = parser.parse_args()
    return args, defaults

def main():
    args, defaults = parse_arguments()
    setup_logging(args.debug)

    config = load_config(args.config)

    args_dict = vars(args)
    explicit_args = {k: v for k, v in args_dict.items() if v != defaults.get(k)}

    final_dict = defaults.copy()
    final_dict.update(config)
    final_dict.update(explicit_args)

    args = SimpleNamespace(**final_dict)

    roi_polygons = None
    if args.roi_json:
        roi_polygons = load_roi(args.roi_json)
        if roi_polygons:
            logging.info(f"Carregadas {len(roi_polygons)} ROIs do arquivo {args.roi_json}")

    uploader = None
    if args.server_url and not args.no_upload:
        try:
            import requests
        except ImportError:
            logging.error("Biblioteca 'requests' não instalada. Instale com: pip install requests")
            sys.exit(1)
        uploader = Uploader(args.server_url, remove_after_upload=args.remove_after_upload)

    # Cria a fonte de acordo com o tipo
    source = None
    if args.source_type == 'camera':
        try:
            device = int(args.source_param)
        except ValueError:
            device = args.source_param
        source = CameraSource(device, width=args.width, height=args.height)
    elif args.source_type == 'rtsp':
        source = RTSPSource(args.source_param)
    elif args.source_type == 'dir':
        source = DirectorySource(args.source_param)
    elif args.source_type == 'video':
        source = VideoFileSource(args.source_param)
    else:
        logging.error("Tipo de fonte inválido")
        sys.exit(1)

    # Envolve a fonte com ThreadedFrameSource para não bloquear o loop principal
    if source is not None:
        source = ThreadedFrameSource(source, timeout_sec=2.0)

    detector = MotionDetector(
        method=args.detection_method,
        threshold=args.threshold,
        min_area=args.min_area,
        roi_polygons_normalized=roi_polygons
    )

    recorder = Recorder(
        output_dir=args.output_dir,
        fps=args.fps,
        pre_record_seconds=args.pre_record
    )

    stop_event = threading.Event()

    def signal_handler(signum, frame):
        logging.info(f"Sinal {signum} recebido, encerrando...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

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
        # Garante que o recorder seja encerrado corretamente
        if recorder is not None and hasattr(recorder, 'shutdown'):
            recorder.shutdown()

if __name__ == "__main__":
    main()