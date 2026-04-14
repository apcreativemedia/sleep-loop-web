# Sleep Loop Web

Aplicación web que convierte un audio corto en un loop perfecto de 1h u 8h usando la técnica de crossfade equal-power (qsin) de 8 segundos.

## Features

- Muestra los **sonidos de sueño más buscados hoy** al entrar (Google Trends + YouTube + lista curada)
- Subir audio → elegir duración (1h / 8h) → genera MP3 sin costuras
- Barra de progreso en vivo
- Se despliega en Railway con un solo Dockerfile

## Pipeline de audio (idéntico al que validaste en el test de 10 min)

1. Trim de 0.5s en cada punta (quitar clicks de grabación)
2. Split en 8 copias → chain de acrossfade `d=8 c1=qsin c2=qsin` entre cada par → genera "loop unit" sin costuras
3. `stream_loop` del unit hasta la duración objetivo (1h = 3600s, 8h = 28800s)
4. `loudnorm I=-18 TP=-1.5 LRA=11` + fade in 5s / fade out 10s
5. Encode MP3 192 kbps estéreo 44.1 kHz

## Deploy en Railway

1. Crea un **nuevo servicio** en tu proyecto Railway existente
2. Subí el proyecto a GitHub (o usá Railway CLI: `railway up` desde esta carpeta)
3. Railway detecta el `Dockerfile` y el `railway.toml` automáticamente
4. No requiere variables de entorno
5. Click en "Generate Domain" para obtener URL pública

### Por qué Docker y no buildpack

`ffmpeg` es obligatorio y los buildpacks de Railway no lo incluyen por defecto. El Dockerfile instala ffmpeg del repo oficial de Debian.

## Correr local

```bash
cd sleep-loop-web
pip install -r requirements.txt
# También necesitás ffmpeg instalado en tu sistema
python app.py
# abrir http://localhost:5000
```

## Notas de rendimiento

- 1h: ~2-4 min en CPU Railway
- 8h: ~15-25 min en CPU Railway
- Gunicorn timeout está en 1800s (30 min) para soportar renders largos
- Los archivos se borran solos al reiniciar el contenedor (efímero en Railway)

## Próximos pasos sugeridos

- Guardar los renders en S3 / R2 para que sobrevivan reinicios
- Login con usuario (para que cada persona vea solo sus renders)
- Integrar con el bot de Telegram existente (un botón "Enviar a Telegram")
- Preview de 30 segundos antes de comprometerse a las 8h
