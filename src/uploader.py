import os
import threading
import logging
import requests

logger = logging.getLogger(__name__)

class Uploader:
    def __init__(self, server_url, field_name='file', remove_after_upload=False):
        self.server_url = server_url
        self.field_name = field_name
        self.remove_after_upload = remove_after_upload

    def upload(self, filepath):
        def _upload():
            try:
                with open(filepath, 'rb') as f:
                    files = {self.field_name: (os.path.basename(filepath), f, 'video/mp4')}
                    logger.info(f"Iniciando upload de {filepath} para {self.server_url}")
                    response = requests.post(self.server_url, files=files, timeout=30)
                    if response.status_code in (200, 201):
                        logger.info(f"Upload concluído: {filepath}")
                        if self.remove_after_upload:
                            os.remove(filepath)
                            logger.debug(f"Arquivo removido: {filepath}")
                    else:
                        logger.error(f"Falha no upload: HTTP {response.status_code}")
            except Exception as e:
                logger.exception(f"Erro no upload: {e}")

        threading.Thread(target=_upload, daemon=True).start()