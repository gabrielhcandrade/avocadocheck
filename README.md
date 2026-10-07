# AvocadoCheck

App de celular que estima a maturação de abacates Hass por foto e recomenda o destino do lote.
Projeto da disciplina SIN 323 (Inteligência Artificial, UFV).

```
avocadocheck/
├── web/               → telas do app (vai para o Netlify)
│   └── config.js      → endereço da API (colocar a URL do Render aqui)
├── backend/           → API com os modelos (vai para o Render)
│   ├── cloud_server.py
│   ├── app.py, pipeline.py
│   ├── models/        → rede neural em ONNX + Random Forest + métricas
│   └── requirements.txt
├── netlify.toml       → configuração do Netlify
├── render.yaml        → configuração do Render
└── export_onnx.py     → converte um novo cnn_final.pt para ONNX
```

## Antes de publicar: colocar os modelos em `backend/models/`

Os modelos não vêm no zip (são grandes). Gere e copie a partir da sua pasta `POC - IA`:

1. Copie `export_onnx.py` para a pasta `POC - IA` (a mesma do `pipeline.py`) e rode lá:
   ```
   python -m pip install onnx onnxruntime
   python export_onnx.py --models-dir outputs_avocado
   ```
   Ele confere se o ONNX dá as mesmas respostas do PyTorch e cria `cnn_final.onnx` e
   `cnn_head.npz` dentro de `outputs_avocado`.
2. Copie estes 6 arquivos de `outputs_avocado` para `avocadocheck/backend/models/`:
   `cnn_final.onnx`, `cnn_head.npz`, `final_model_stage.joblib`, `final_model_days.joblib`,
   `metrics_stage.csv`, `metrics_days.csv`.

No PowerShell, de dentro da pasta `POC - IA` (ajuste o caminho do avocadocheck):
```
$dest = "C:\caminho\para\avocadocheck\backend\models"
copy outputs_avocado\cnn_final.onnx, outputs_avocado\cnn_head.npz, outputs_avocado\final_model_stage.joblib, outputs_avocado\final_model_days.joblib, outputs_avocado\metrics_stage.csv, outputs_avocado\metrics_days.csv $dest
```

## Publicar

### 1. GitHub
Crie um repositório e envie esta pasta inteira. Os modelos têm cerca de 43 MB cada,
abaixo do limite de 100 MB por arquivo do GitHub, então não precisa de Git LFS.

```
git init
git add .
git commit -m "AvocadoCheck"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/avocadocheck.git
git push -u origin main
```

### 2. Render (API)
1. No Render: **New > Blueprint** e escolha o repositório. Ele lê o `render.yaml` sozinho.
   Se preferir **New > Web Service**, use:
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn cloud_server:api --host 0.0.0.0 --port $PORT`
   - Variável de ambiente `PYTHON_VERSION` = `3.12.8`
2. Espere o deploy terminar e abra `https://SEU-SERVICO.onrender.com/api/info`.
   Se aparecer um JSON com as condições e os modelos, a API está no ar.

### 3. Netlify (telas)
1. Edite `web/config.js` e coloque a URL do Render:
   ```js
   window.AVOCHECK_API = 'https://SEU-SERVICO.onrender.com';
   ```
   Faça commit e push.
2. No Netlify: **Add new site > Import an existing project**, escolha o repositório.
   O `netlify.toml` já diz para publicar a pasta `web`, sem comando de build.
3. Abra o endereço `https://....netlify.app` no celular. Pelo HTTPS do Netlify a câmera
   abre ao vivo dentro do app. No menu do navegador, use "Adicionar à tela inicial".

### 4. (Opcional) Restringir quem usa a API
No Render, mude a variável `ALLOWED_ORIGINS` de `*` para o endereço do Netlify,
por exemplo `https://avocadocheck.netlify.app`.

## Plano grátis do Render
- O servidor dorme depois de 15 minutos sem uso. A primeira abertura depois disso leva
  até 1 minuto, e o app mostra "Conectando ao servidor" enquanto isso.
- O limite é 512 MB de memória. Por isso a API usa a rede neural em ONNX, sem PyTorch:
  medida aqui, ela usa cerca de 405 MB no pico e dá exatamente as mesmas respostas.
- Na nuvem, o detector de frutos é o detector por cor do pipeline (YOLO-World precisa
  do PyTorch). Ele funciona bem com fundo claro.

## Atualizar os modelos depois de treinar de novo
No computador, depois de rodar o `pipeline.py --fit-final-cnn`:
```
python export_onnx.py --models-dir outputs_avocado
```
Copie para `backend/models/`: `cnn_final.onnx`, `cnn_head.npz`, `final_model_stage.joblib`,
`final_model_days.joblib`, `metrics_stage.csv` e `metrics_days.csv`. Depois faça commit e push.
O `requirements.txt` fixa `scikit-learn==1.8.0`, a versão usada para salvar os modelos.
Se você treinar com outra versão, mude esse número.

## Rodar tudo no computador (sem nuvem)
```
cd backend
pip install -r requirements.txt
uvicorn cloud_server:api --port 8000
```
Abra `http://localhost:8000/app/`.
