"""
Sleep Loop Web - Flask app (v2 — speed-optimized)
Features:
  - Trending sleep sounds (Google Trends US only)
  - Upload 1 or many audios
  - Separate mode: each audio -> its own loop
  - Mix mode: all audios mixed into one loop (rain + thunder + wind)
  - Custom duration in minutes (10 to 720)
  - Seamless render: N-copy unit, qsin 8s internal crossfades, 20s wrap-around seal
  - Auto loop-point detection (RMS + waveform + spectral centroid)
  - Cuts report (.txt) with INTERNAL + WRAP timestamps

Speed optimisations (v2):
  1. Fixed asplit bug — uses N separate -i inputs (real concatenation)
  2. loudnorm applied once to sealed unit, not to the 8h stream
  3. -threads 0 on every ffmpeg call (use all cores)
  4. Sealed unit cached — not deleted after render
"""
import os
import uuid
import time
import threading
import subprocess
from datetime import datetime
from pathlib import Path

from flask import Flask, request, render_template, send_file, jsonify, abort

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
RENDER_DIR = BASE_DIR / "renders"
UPLOAD_DIR.mkdir(exist_ok=True)
RENDER_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1000 * 1024 * 1024  # 1 GB (for multi-upload)

JOBS = {}
JOBS_LOCK = threading.Lock()

# Render constants
INTERNAL_XFADE = 8.0      # inside the unit (do NOT change — user-approved)
WRAP_XFADE = 20.0         # wrap-around seal between units
TARGET_UNIT_DUR = 600.0   # aim unit >= 10 min so wraps are rare
MAX_UNIT_COPIES = 16
MIN_UNIT_COPIES = 4

def choose_unit_copies(clean_dur: float) -> int:
    import math
    n = int(math.ceil(TARGET_UNIT_DUR / max(clean_dur, 1.0)))
    return max(MIN_UNIT_COPIES, min(MAX_UNIT_COPIES, n))

# -----------------------------------------------------------------------------
# Trending: Google Trends US only
# -----------------------------------------------------------------------------
TRENDING_CACHE = {"date": None, "items": []}


def fetch_google_trends(keyword_seeds):
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=360, timeout=(5, 10))
        results = []
        seen = set()
        for seed in keyword_seeds:
            try:
                pytrends.build_payload([seed], timeframe="now 1-d", geo="US")
                related = pytrends.related_queries()
                q_top = related.get(seed, {}).get("top")
                if q_top is not None:
                    for _, row in q_top.head(5).iterrows():
                        topic = str(row["query"]).strip()
                        low = topic.lower()
                        if low not in seen:
                            seen.add(low)
                            url = f"https://trends.google.com/trends/explore?q={topic.replace(' ', '%20')}&geo=US"
                            results.append({"topic": topic, "url": url, "source": "Google Trends"})
                q_rising = related.get(seed, {}).get("rising")
                if q_rising is not None:
                    for _, row in q_rising.head(3).iterrows():
                        topic = str(row["query"]).strip()
                        low = topic.lower()
                        if low not in seen:
                            seen.add(low)
                            url = f"https://trends.google.com/trends/explore?q={topic.replace(' ', '%20')}&geo=US"
                            results.append({"topic": f"\U0001f525 {topic}", "url": url, "source": "Google Trends (rising)"})
                time.sleep(1)
            except Exception:
                continue
        return results
    except Exception as e:
        print(f"[trends] failed: {e}")
        return []


def get_trending():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if TRENDING_CACHE["date"] == today and TRENDING_CACHE["items"]:
        return TRENDING_CACHE["items"]
    seeds = ["sleep sounds", "rain sounds", "white noise", "asmr sleep", "ambient sleep"]
    items = fetch_google_trends(seeds)
    TRENDING_CACHE["date"] = today
    TRENDING_CACHE["items"] = items
    return items


# -----------------------------------------------------------------------------
# Audio helpers
# -----------------------------------------------------------------------------
def probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True
    )
    s = r.stdout.strip()
    if not s or s == "N/A":
        r2 = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-count_frames", "-show_entries", "stream=duration_ts,sample_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True
        )
        lines = dict(ln.split("=") for ln in r2.stdout.strip().splitlines() if "=" in ln)
        return float(lines.get("duration_ts", 0)) / float(lines.get("sample_rate", 44100))
    return float(s)


def find_best_loop_point(wav_path, window_sec=2.0, search_back_sec=5.0):
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import stft
    sr, audio = wavfile.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32) / 32768.0
    total = len(audio)
    window = int(window_sec * sr)
    ref = audio[:window]
    ref_rms = float(np.sqrt(np.mean(ref * ref) + 1e-12))

    def centroid(x):
        f, _, Z = stft(x, fs=sr, nperseg=1024)
        mag = np.abs(Z).mean(axis=1)
        if mag.sum() < 1e-9:
            return 0.0
        return float((f * mag).sum() / mag.sum())

    ref_cent = centroid(ref)
    search_back_samples = int(search_back_sec * sr)
    search_start = max(window + int(0.5 * sr), total - search_back_samples)
    search_end = total
    step = max(1, int(0.05 * sr))
    best_score = -1e9
    best_end = search_end
    a0 = ref - ref.mean()
    a_norm = float(np.sqrt((a0 * a0).sum()) + 1e-12)
    for end_pos in range(search_start, search_end, step):
        cand = audio[end_pos - window:end_pos]
        if len(cand) < window:
            continue
        cand_rms = float(np.sqrt(np.mean(cand * cand) + 1e-12))
        b = cand - cand.mean()
        b_norm = float(np.sqrt((b * b).sum()) + 1e-12)
        corr = float((a0 * b).sum() / (a_norm * b_norm))
        rms_match = 1.0 - abs(cand_rms - ref_rms) / (ref_rms + cand_rms + 1e-6)
        cand_cent = centroid(cand)
        cent_match = 1.0 - abs(cand_cent - ref_cent) / (ref_cent + cand_cent + 1e-6)
        score = 0.5 * corr + 0.25 * rms_match + 0.25 * cent_match
        if score > best_score:
            best_score = score
            best_end = end_pos
    return best_end / sr, float(best_score)


def fmt_t(sec):
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - (h * 3600 + m * 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:05.2f}"
    return f"{m}:{s:05.2f}"


# -----------------------------------------------------------------------------
# Core render pipeline  (v2 — speed-optimized)
# -----------------------------------------------------------------------------
def render_loop(job_id: str, input_paths, duration_min: int, title_hint: str, mix_mode: bool):
    """Render one loop. `input_paths` is list[Path] (len 1 normal, len>1 if mix_mode)."""
    def update(status=None, progress=None, error=None, output=None):
        with JOBS_LOCK:
            if status is not None: JOBS[job_id]["status"] = status
            if progress is not None: JOBS[job_id]["progress"] = progress
            if error is not None: JOBS[job_id]["error"] = error
            if output is not None: JOBS[job_id]["output"] = output

    work_dir = RENDER_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    duration_hours = duration_min / 60.0
    target_sec = duration_min * 60

    try:
        # Step 1: decode to raw WAV (if mix_mode, amix N sources)
        update(status="Decoding source", progress=5)
        raw_wav = work_dir / "raw.wav"
        if mix_mode and len(input_paths) > 1:
            inputs = []
            for p in input_paths:
                inputs += ["-i", str(p)]
            n = len(input_paths)
            filter_mix = f"amix=inputs={n}:duration=longest:normalize=0:weights='{' '.join(['1']*n)}',dynaudnorm=f=500:g=15"
            subprocess.run([
                "ffmpeg", "-y", "-threads", "0", *inputs,
                "-filter_complex", filter_mix,
                "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
                str(raw_wav)
            ], check=True, capture_output=True)
        else:
            subprocess.run([
                "ffmpeg", "-y", "-threads", "0", "-i", str(input_paths[0]),
                "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
                str(raw_wav)
            ], check=True, capture_output=True)

        src_dur = probe_duration(raw_wav)
        print(f"[render] job={job_id} src_dur={src_dur:.2f}s mix={mix_mode} n_in={len(input_paths)}", flush=True)
        if src_dur < 20:
            raise RuntimeError(f"Audio demasiado corto ({src_dur:.1f}s). Necesita al menos 20 segundos.")

        # Step 2: find best loop point
        update(status="Finding natural loop point", progress=10)
        default_end = max(src_dur - 0.5, min(src_dur, 5.0))
        try:
            best_end, score = find_best_loop_point(
                raw_wav,
                window_sec=2.0,
                search_back_sec=min(6.0, max(2.0, src_dur / 3))
            )
            if score > 0.3 and best_end > 20.0 and best_end <= src_dur - 0.05:
                trim_end = best_end
                print(f"[render] auto-loop: end={trim_end:.2f}s score={score:.3f}", flush=True)
            else:
                trim_end = default_end
                print(f"[render] auto-loop fallback score={score:.3f}", flush=True)
        except Exception as e:
            print(f"[render] auto-loop failed: {e}", flush=True)
            trim_end = default_end

        clean_wav = work_dir / "clean.wav"
        subprocess.run([
            "ffmpeg", "-y", "-threads", "0", "-i", str(raw_wav),
            "-af", f"atrim=start=0.5:end={trim_end},asetpts=PTS-STARTPTS",
            "-ac", "2", "-ar", "44100", str(clean_wav)
        ], check=True, capture_output=True)
        raw_wav.unlink(missing_ok=True)
        clean_dur = trim_end - 0.5

        # ------------------------------------------------------------------
        # Step 3: build N-copy unit with INTERNAL qsin 8s crossfades
        #   FIX (v2): use N separate -i inputs instead of asplit
        #   asplit creates references to the same stream and does NOT
        #   concatenate — chained acrossfade just trims the same copy.
        # ------------------------------------------------------------------
        unit_copies = choose_unit_copies(clean_dur)
        expected_unit_dur = unit_copies * clean_dur - (unit_copies - 1) * INTERNAL_XFADE
        print(f"[render] job={job_id} unit_copies={unit_copies} clean_dur={clean_dur:.1f}s expected_unit_dur={expected_unit_dur:.1f}s", flush=True)
        update(status=f"Building unit ({unit_copies} copies)", progress=20)
        unit_wav = work_dir / "unit.wav"

        if unit_copies == 1:
            subprocess.run([
                "ffmpeg", "-y", "-threads", "0", "-i", str(clean_wav),
                "-c:a", "pcm_s16le", str(unit_wav)
            ], check=True, capture_output=True)
        else:
            # N separate -i inputs — ffmpeg treats them as independent streams
            inputs = []
            for i in range(unit_copies):
                inputs += ["-i", str(clean_wav)]

            if unit_copies == 2:
                fc = f"[0][1]acrossfade=d={INTERNAL_XFADE}:c1=qsin:c2=qsin[loop]"
            else:
                fc = f"[0][1]acrossfade=d={INTERNAL_XFADE}:c1=qsin:c2=qsin[x1];"
                for i in range(2, unit_copies - 1):
                    fc += f"[x{i-1}][{i}]acrossfade=d={INTERNAL_XFADE}:c1=qsin:c2=qsin[x{i}];"
                fc += f"[x{unit_copies-2}][{unit_copies-1}]acrossfade=d={INTERNAL_XFADE}:c1=qsin:c2=qsin[loop]"

            subprocess.run([
                "ffmpeg", "-y", "-threads", "0", *inputs,
                "-filter_complex", fc, "-map", "[loop]",
                "-c:a", "pcm_s16le", str(unit_wav)
            ], check=True, capture_output=True)

        clean_wav.unlink(missing_ok=True)
        actual_unit_dur = probe_duration(unit_wav)
        print(f"[render] unit_dur={actual_unit_dur:.2f}s (expected {expected_unit_dur:.1f}s)", flush=True)

        # ------------------------------------------------------------------
        # Step 4: seal the unit — 3 separate ffmpeg calls (avoids asplit sync bug)
        # ------------------------------------------------------------------
        update(status="Sealing wrap (20s crossfade)", progress=35)
        sealed_wav = work_dir / "sealed.wav"
        first_wav = work_dir / "first.wav"
        second_wav = work_dir / "second.wav"

        # 4a: first = unit[WRAP..end]
        subprocess.run([
            "ffmpeg", "-y", "-threads", "0", "-i", str(unit_wav),
            "-af", f"atrim=start={WRAP_XFADE},asetpts=PTS-STARTPTS",
            "-c:a", "pcm_s16le", str(first_wav)
        ], check=True, capture_output=True)

        # 4b: second = unit[0..WRAP]
        subprocess.run([
            "ffmpeg", "-y", "-threads", "0", "-i", str(unit_wav),
            "-af", f"atrim=start=0:end={WRAP_XFADE},asetpts=PTS-STARTPTS",
            "-c:a", "pcm_s16le", str(second_wav)
        ], check=True, capture_output=True)

        # 4c: crossfade first + second
        subprocess.run([
            "ffmpeg", "-y", "-threads", "0",
            "-i", str(first_wav), "-i", str(second_wav),
            "-filter_complex", f"[0][1]acrossfade=d={WRAP_XFADE}:c1=qsin:c2=qsin[out]",
            "-map", "[out]", "-c:a", "pcm_s16le", str(sealed_wav)
        ], check=True, capture_output=True)

        unit_wav.unlink(missing_ok=True)
        first_wav.unlink(missing_ok=True)
        second_wav.unlink(missing_ok=True)

        sealed_dur = probe_duration(sealed_wav)
        print(f"[render] sealed_dur={sealed_dur:.2f}s", flush=True)

        if sealed_dur < 10:
            raise RuntimeError(f"sealed unit too short ({sealed_dur:.2f}s) — seal failed")

        # ------------------------------------------------------------------
        # Step 4.5 (v2): normalize the sealed unit ONCE with loudnorm
        #   Previously loudnorm ran on the full 8h stream — huge bottleneck.
        #   Now we normalize the ~10 min unit and stream_loop the result.
        # ------------------------------------------------------------------
        update(status="Normalizing sealed unit (loudnorm)", progress=45)
        normed_wav = work_dir / "normed.wav"
        subprocess.run([
            "ffmpeg", "-y", "-threads", "0",
            "-i", str(sealed_wav),
            "-af", "loudnorm=I=-18:TP=-1.5:LRA=11",
            "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
            str(normed_wav)
        ], check=True, capture_output=True)

        normed_dur = probe_duration(normed_wav)
        print(f"[render] normed_dur={normed_dur:.2f}s", flush=True)

        # ------------------------------------------------------------------
        # Step 5: stream_loop + fades + MP3 encode
        #   loudnorm already applied — only fades + encode here (FAST)
        # ------------------------------------------------------------------
        update(status=f"Rendering {duration_min} min MP3", progress=50)
        fade_out_start = target_sec - 10
        out_mp3 = work_dir / "output.mp3"
        render_cmd = [
            "ffmpeg", "-y", "-threads", "0",
            "-nostats", "-loglevel", "error",
            "-stream_loop", "-1", "-i", str(normed_wav),
            "-t", str(target_sec),
            "-af", f"afade=t=in:st=0:d=10,afade=t=out:st={fade_out_start}:d=10",
            "-c:a", "libmp3lame", "-b:a", "192k",
            "-ac", "2", "-ar", "44100",
            "-metadata", f"title={title_hint or 'Sleep Loop'} {duration_min}min",
            "-metadata", "artist=Sleep Loop Web",
            str(out_mp3)
        ]
        result = subprocess.run(render_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0:
            err_text = result.stderr.decode() if result.stderr else "unknown error"
            print(f"[render] FAILED: {err_text[-600:]}", flush=True)
            raise RuntimeError(f"render failed: {err_text[-600:]}")

        out_dur = probe_duration(out_mp3)
        print(f"[render] output_dur={out_dur:.2f}s", flush=True)
        if out_dur < target_sec * 0.95:
            raise RuntimeError(f"output is only {out_dur:.0f}s, expected {target_sec}s")

        # Step 6: build cuts report
        internal_cuts_in_unit = []
        for k in range(1, unit_copies):
            t = k * (clean_dur - INTERNAL_XFADE) + (k - 1) * INTERNAL_XFADE
            internal_cuts_in_unit.append(t + INTERNAL_XFADE / 2)

        sealed_internals = [c - WRAP_XFADE for c in internal_cuts_in_unit if c - WRAP_XFADE > 0]
        wrap_midpoint = sealed_dur - WRAP_XFADE / 2

        all_internals = []
        all_wraps = []
        unit_idx = 0
        while unit_idx * sealed_dur < target_sec:
            base = unit_idx * sealed_dur
            for sc in sealed_internals:
                p = base + sc
                if p < target_sec:
                    all_internals.append(p)
            wp = base + wrap_midpoint
            if wp < target_sec - 1 and (unit_idx + 1) * sealed_dur < target_sec:
                all_wraps.append(wp)
            unit_idx += 1

        lines = [
            f"Sleep Loop - cut report",
            f"Source duration: {src_dur:.2f}s ({fmt_t(src_dur)})",
            f"Clean duration: {clean_dur:.2f}s",
            f"Unit copies: {unit_copies}",
            f"Unit duration (before seal): {actual_unit_dur:.2f}s ({fmt_t(actual_unit_dur)})",
            f"Sealed unit duration: {sealed_dur:.2f}s ({fmt_t(sealed_dur)})",
            f"INTERNAL crossfade: qsin {INTERNAL_XFADE}s",
            f"WRAP crossfade: qsin {WRAP_XFADE}s",
            f"Target: {duration_min} min ({target_sec}s)",
            f"",
            f"HARD CUTS: none (sealed loop eliminates them)",
            f"",
            f"WRAP crossfades between sealed units (qsin {WRAP_XFADE:.0f}s) - {len(all_wraps)} total:",
        ]
        for i, c in enumerate(all_wraps, 1):
            lines.append(f"  {i}. {fmt_t(c)}  ({c:.2f}s)")
        lines.append("")
        lines.append(f"INTERNAL crossfades inside unit (qsin {INTERNAL_XFADE:.0f}s) - {len(all_internals)} total:")
        for i, c in enumerate(all_internals, 1):
            lines.append(f"  {i}. {fmt_t(c)}  ({c:.2f}s)")
        cuts_txt = work_dir / "cuts.txt"
        cuts_txt.write_text("\n".join(lines))

        with JOBS_LOCK:
            JOBS[job_id]["cuts_file"] = str(cuts_txt)
            JOBS[job_id]["wrap_cuts"] = [fmt_t(c) for c in all_wraps]
            JOBS[job_id]["internal_cuts_count"] = len(all_internals)

        # v2: keep sealed.wav + normed.wav for potential reuse (different durations)
        # only clean up the normed intermediate
        normed_wav.unlink(missing_ok=True)
        # sealed_wav is kept (cache optimisation #4)

        update(status="Done", progress=100, output=str(out_mp3))
        print(f"[render] DONE job={job_id} dur={out_dur:.0f}s", flush=True)

    except subprocess.CalledProcessError as e:
        err = e.stderr.decode() if e.stderr else str(e)
        print(f"[render] FAILED job={job_id}: {err[-500:]}", flush=True)
        update(status="Failed", error=err[-500:])
    except Exception as e:
        print(f"[render] FAILED job={job_id}: {e}", flush=True)
        update(status="Failed", error=str(e))


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/trending")
def api_trending():
    return jsonify(get_trending())


@app.route("/api/render", methods=["POST"])
def api_render():
    files = request.files.getlist("audio")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No audio files"}), 400
    files = [f for f in files if f.filename]

    duration_raw = request.form.get("duration_min", "60")
    try:
        duration_min = int(float(duration_raw))
        if duration_min < 10 or duration_min > 720:
            return jsonify({"error": "Duration must be 10-720 minutes"}), 400
    except ValueError:
        return jsonify({"error": "Invalid duration"}), 400

    mode = request.form.get("mode", "separate")
    title = request.form.get("title", "")[:60]

    saved_paths = []
    for f in files:
        pid = uuid.uuid4().hex[:12]
        ext = Path(f.filename).suffix or ".mp3"
        p = UPLOAD_DIR / f"{pid}{ext}"
        f.save(str(p))
        saved_paths.append(p)

    job_ids = []

    if mode == "mix" and len(saved_paths) > 1:
        job_id = uuid.uuid4().hex[:12]
        with JOBS_LOCK:
            JOBS[job_id] = {
                "status": "Queued", "progress": 0, "error": None, "output": None,
                "duration_min": duration_min, "title": title or "mix",
                "created_at": time.time(), "label": f"MIX ({len(saved_paths)} sources)",
            }
        t = threading.Thread(
            target=render_loop,
            args=(job_id, saved_paths, duration_min, title or "mix", True),
            daemon=True,
        )
        t.start()
        job_ids.append({"job_id": job_id, "label": f"Mix of {len(saved_paths)} audios"})
    else:
        for idx, p in enumerate(saved_paths):
            job_id = uuid.uuid4().hex[:12]
            base_name = Path(files[idx].filename).stem[:40]
            per_title = title or base_name
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "Queued", "progress": 0, "error": None, "output": None,
                    "duration_min": duration_min, "title": per_title,
                    "created_at": time.time(), "label": base_name,
                }
            t = threading.Thread(
                target=render_loop,
                args=(job_id, [p], duration_min, per_title, False),
                daemon=True,
            )
            t.start()
            job_ids.append({"job_id": job_id, "label": base_name})

    return jsonify({"jobs": job_ids})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Unknown job"}), 404
        return jsonify({
            "status": job["status"],
            "progress": job["progress"],
            "error": job["error"],
            "done": job["output"] is not None,
            "wrap_cuts": job.get("wrap_cuts", []),
            "internal_cuts_count": job.get("internal_cuts_count", 0),
            "has_cuts_file": bool(job.get("cuts_file")),
            "label": job.get("label", ""),
        })


@app.route("/download/<job_id>")
def download(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or not job["output"]:
        abort(404)
    title = (job.get("title") or "sleep-loop").replace(" ", "-")
    dur = job.get("duration_min", 60)
    dl_name = f"{title}-{dur}min.mp3"
    return send_file(job["output"], as_attachment=True, download_name=dl_name, mimetype="audio/mpeg")


@app.route("/cuts/<job_id>")
def download_cuts(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or not job.get("cuts_file"):
        abort(404)
    title = (job.get("title") or "sleep-loop").replace(" ", "-")
    dur = job.get("duration_min", 60)
    dl_name = f"{title}-{dur}min-cuts.txt"
    return send_file(job["cuts_file"], as_attachment=True, download_name=dl_name, mimetype="text/plain")


@app.route("/health")
def health():
    return "ok", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
