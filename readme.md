# Motion Recording System with Detection

Sistema de gravação automática de vídeos baseado em detecção de movimento.

## Características

- **Múltiplas fontes de vídeo**: câmera USB/Webcam (índice ou dispositivo), stream RTSP, diretório com imagens e arquivo de vídeo.
- **Detecção de movimento**: dois métodos disponíveis (`diff` – diferença entre quadros, `mog2` – fundo adaptativo).
- **Pré‑gravação**: salva os segundos anteriores ao movimento, evitando perder o início do evento.
- **Gravação em vídeo**: arquivos MP4 com nome no formato `clip_AAAAMMDD_HHMMSS_AAAAMMDD_HHMMSS.mp4` (início e fim).
- **Áreas de interesse (ROIs)**: permite definir polígonos onde o movimento é monitorado, ignorando o restante.
- **Upload opcional**: envia os vídeos para um servidor remoto via HTTP.
- **Preview em tempo real**: exibe preview com marcação das áreas de movimento e ROIs (opcional).
- **Configuração via arquivo JSON e linha de comando**: flexibilidade para diferentes cenários.

Instale as dependências:

```bash
pip install opencv-python
pip install requestsS