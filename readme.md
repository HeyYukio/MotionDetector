# MotionRecorder – Sistema inteligente de gravação por detecção de movimento

Sistema modular em Python para captura de vídeo, detecção de movimento, gravação seletiva (clipes) e contínua (debug), com upload automático e gerenciamento de armazenamento. Suporta câmeras USB/IP, streams RTSP, arquivos de vídeo e sequências de imagens. Produz vídeos **MP4** ou **MKV com timestamps precisos por frame**.

## 📋 Funcionalidades

- **Múltiplas fontes de vídeo**  
  Câmera local (V4L2/DirectShow), RTSP, arquivo de vídeo, diretório de imagens.

- **Detecção de movimento configurável**  
  Métodos: diferença entre quadros ou MOG2 (subtração de fundo).  
  Parâmetros ajustáveis: threshold, área mínima, pré‑gravação, cooldown, número mínimo de frames com movimento.

- **Regiões de Interesse (ROIs)**  
  Defina polígonos normalizados via arquivo JSON para ignorar áreas irrelevantes.

- **Gravação inteligente**  
  Inicia gravação após `min_motion_frames` consecutivos com movimento; para após `cooldown` segundos sem movimento.  
  **Pré‑gravação**: buffer circular que inclui os segundos anteriores ao início do movimento no clipe.

- **Gravação contínua de debug** *(opcional)*  
  Gera um vídeo completo de toda a sessão, independente de movimento, com prefixo `debug_`.

- **Upload automático**  
  Envia os clipes de movimento (`clip_*.mp4`/`.mkv`) para um servidor HTTP. Opção de remover o arquivo local após upload.

- **Gerenciamento de armazenamento**  
  Limite máximo de espaço com políticas **parar** (stop) ou **deletar arquivos antigos** (delete_oldest).

- **Preview em tempo real**  
  Janela OpenCV exibindo detecções, status e FPS. Pressione `q` para sair.

- **Nomes de arquivo padronizados**  
  `clip_<início>_<fim>_<id>.mp4` / `debug_<início>_<fim>_<id>.mkv`  
  IDs de equipamento personalizáveis.

- **DOIS FORMATOS DE SAÍDA**  
  - **MP4** (padrão): usando OpenCV, sem timestamps por frame.
  - **MKV** (PyAV): cada frame carrega seu instante de captura em tempo real (UTC com precisão de microssegundos). Permite rastreamento temporal preciso e recuperação via `ffprobe` ou API PyAV.

## 🧱 Arquitetura simplificada
