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