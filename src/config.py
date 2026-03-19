import json
import os
import logging

logger = logging.getLogger(__name__)

def load_config(config_file='config.json'):
    """Carrega configurações de um arquivo JSON. Retorna dict vazio se não existir."""
    if not os.path.exists(config_file):
        logger.info(f"Arquivo de configuração {config_file} não encontrado. Usando padrões.")
        return {}
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuração carregada de {config_file}")
        return config
    except Exception as e:
        logger.error(f"Erro ao ler {config_file}: {e}")
        return {}

def load_roi(roi_file):
    """
    Carrega um arquivo JSON de ROIs e retorna uma lista de polígonos no formato normalizado.
    O arquivo pode conter 'polygons_normalized' ou 'polygons_absolute'.
    Se contiver 'polygons_absolute', converte para normalizado usando 'image_size'.
    Retorna uma lista de polígonos, onde cada polígono é uma lista de pontos (x, y) normalizados.
    """
    if not os.path.exists(roi_file):
        logger.warning(f"Arquivo de ROI {roi_file} não encontrado. Nenhuma ROI será usada.")
        return []

    try:
        with open(roi_file, 'r') as f:
            data = json.load(f)

        polygons = []

        # Prioridade: polygons_normalized
        if 'polygons_normalized' in data:
            for poly_data in data['polygons_normalized']:
                points = poly_data.get('points', [])
                # Garantir que cada ponto é uma tupla (x, y) normalizada
                norm_points = [(p[0], p[1]) for p in points]
                polygons.append(norm_points)
            logger.info(f"Carregadas {len(polygons)} ROIs normalizadas de {roi_file}")

        elif 'polygons_absolute' in data and 'image_size' in data:
            # Converte absoluto para normalizado
            width = data['image_size']['width']
            height = data['image_size']['height']
            for poly_data in data['polygons_absolute']:
                points = poly_data.get('points', [])
                norm_points = [(p[0] / width, p[1] / height) for p in points]
                polygons.append(norm_points)
            logger.info(f"Carregadas {len(polygons)} ROIs absolutas de {roi_file} e convertidas para normalizadas")

        else:
            logger.warning(f"Formato de ROI não reconhecido em {roi_file}")

        return polygons

    except Exception as e:
        logger.error(f"Erro ao ler arquivo de ROI {roi_file}: {e}")
        return []