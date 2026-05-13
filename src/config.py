import json
import os
import logging

logger = logging.getLogger(__name__)

def load_config(config_file='config.json'):
    if not os.path.exists(config_file):
        logger.info(f"Config {config_file} não encontrada.")
        return {}
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"Config carregada de {config_file}")
        return config
    except Exception as e:
        logger.error(f"Erro ao ler {config_file}: {e}")
        return {}

def load_roi(roi_file):
    if not os.path.exists(roi_file):
        logger.warning(f"ROI {roi_file} não encontrada.")
        return []
    try:
        with open(roi_file, 'r') as f:
            data = json.load(f)
        polygons = []
        if 'polygons_normalized' in data:
            for poly_data in data['polygons_normalized']:
                points = poly_data.get('points', [])
                polygons.append([(p[0], p[1]) for p in points])
        elif 'polygons_absolute' in data and 'image_size' in data:
            w, h = data['image_size']['width'], data['image_size']['height']
            for poly_data in data['polygons_absolute']:
                points = poly_data.get('points', [])
                polygons.append([(p[0]/w, p[1]/h) for p in points])
        return polygons
    except Exception as e:
        logger.error(f"Erro ao carregar ROI: {e}")
        return []