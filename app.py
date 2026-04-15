"""
Sleep Loop Web - Flask app
Features:
  - Shows today's most-searched sleep sounds (Google Trends + YouTube)
  - User uploads audio, picks duration (1h or 8h)
  - Renders seamless loop using qsin 8s crossfade technique
  - Download link when ready
"""
import os
import uuid
import json
import time
import shutil
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
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload cap

# In-memory job tracker: {job_id: {"status": "...", "progress": 0-100, "output": path, "error": None}}
JOBS = {}
JOBS_LOCK = threading.Lock()

# -----------------------------------------------------------------------------
# Trending: Google Trends + YouTube search
# -----------------------------------------------------------------------------
TRENDING_CACHE = {"date": None, "items": []}

FALLBACK_TOPICS = [
    "rain sounds", "ocean waves", "thunderstorm", "forest night",
    "white noise", "pink noise", "brown noise", "fireplace crackling",
    "fan sound", "wind in trees", "babbling brook", "mountain stream",
    "tropical rain", "beach waves", "crickets night", "owl forest",
    "campfire", "waterfall", "heavy rain on tent", "gentle stream"
]


def fetch_google_trends(keyword_seeds):
    """Try pytrends; return list of (topic, url) or [] on failure."""
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=0, timeout=(5, 10))
        results = []
        for seed in keyword_seeds[:3]:
            try:
                pytrends.build_payload([seed], timeframe="now 1-d", geo="")
                related = pytrends.related_queries()
                q = related.get(seed, {}).get("top")
                if q is not None:
                    for _, row in q.head(3).iterrows():
                        topic = str(row["query"])
                        url = f"https://trends.google.com/trends/explore?q={topic.replace(' ', '%20')}"
                        results.append({"topic": topic, "url": url, "source": "Google Trends"})
                time.sleep(1)
            except Exception:
                continue
        return results[:5]
    except Exception as e:
        print(f"[trends] failed: {e}")
        return []


def fetch_youtube_trending(query_list):
    """Use yt-dlp to search YouTube for popular sleep sound videos."""
    try:
        import yt_dlp
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "default_search": "ytsearch5",
            "skip_download": True,
        }
        results = []
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for q in query_list[:3]:
                try:
                    info = ydl.extract_info(f"ytsearch5:{q} 8 hours sleep", download=False)
                    for entry in info.get("entries", [])[:2]:
                        title = entry.get("title", "")
                        vid = entry.get("id", "")
                        url = f"https://youtu.be/{vid}" if vid else entry.get("url", "")
                        results.append({"topic": title[:80], "url": url, "source": "YouTube"})
                except Exception:
                    continue
        return results[:5]
    except Exception as e:
        print(f"[youtube] failed: {e}")
        return []


def get_trending():
    """Return today's trending sleep sounds. Cached for the day."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if TRENDING_CACHE["date"] == today and TRENDING_CACHE["items"]:
        return TRENDING_CACHE["items"]

    seeds = ["sleep sounds", "rain sounds", "white noise"]
    items = []

    trends = fetch_google_trends(seeds)
    items.extend(trends)

    youtube = fetch_youtube_trending(seeds)
    items.extend(youtube)

    # Always add 5 curated fallbacks rotated by day
    import hashlib
    day_hash = int(hashlib.md5(today.encode()).hexdigest(), 16)
    rotated = FALLBACK_TOPICS[day_hash % len(FALLBACK_TOPICS):] + FALLBACK_TOPICS[:day_hash % len(FALLBACK_TOPICS)]
    for t in rotated[:5]:
        items.append({
            "topic": t,
            "url": f"https://www.pond5.com/search?kw={t.replace(' ', '+')}&media=sfx",
            "source": "Pond5 search"
        })

    TRENDING_CACHE["date"] = today
    TRENDING_CACHE["items"] = items
    return items


# -----------------------------------------------------------------------------
# Audio pipeline - seamless loop with qsin 8s crossfade
# -----------------------------------------------------------------------------
def render_loop(job_id: str, input_path: Path, duration_hours: float, title_hint: str = ""):
    """Background rendering job."""
    def update(status=None, progress=None, error=None, output=None):
        with JOBS_LOCK:
            if status is not None:
                JOBS[job_id]["status"] = status
            if progress is not None:
                JOBS[job_id]["progress"] = progress
            if error is not None:
                JOBS[job_id]["error"] = error
            if output is not None:
                JOBS[job_id]["output"] = output

    work_dir = RENDER_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        update(status="Cleaning source", progress=5)

        # Step 1: decode source to normalized WAV FIRST, then measure from WAV.
        # This handles any input format and formats missing duration metadata.
        raw_wav = work_dir / "raw.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_path),
            "-ac", "2", "-ar", "44100",
            "-c:a", "pcm_s16le",
            str(raw_wav)
        ], check=True, capture_output=True)

        # Measure from the decoded WAV (reliable)
        def probe_duration(path: Path) -> float:
            r = subprocess.run(
                ["ffprobe", "-v", "error",
                 "-select_streams", "a:0",
                 "-show_entries", "stream=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, check=True
            )
            s = r.stdout.strip()
            if not s or s == "N/A":
                # Fallback: count samples
                r2 = subprocess.run(
                    ["ffprobe", "-v", "error",
                     "-select_streams", "a:0",
                     "-count_frames", "-show_entries", "stream=duration_ts,sample_rate",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                    capture_output=True, text=True, check=True
                )
                lines = dict(ln.split("=") for ln in r2.stdout.strip().splitlines() if "=" in ln)
                return float(lines.get("duration_ts", 0)) / float(lines.get("sample_rate", 44100))
            return float(s)

        src_dur = probe_duration(raw_wav)
        print(f"[render] job={job_id} src_dur={src_dur:.2f}s", flush=True)
        if src_dur < 20:
            raise RuntimeError(f"Audio demasiado corto ({src_dur:.1f}s). Necesita al menos 20 segundos para hacer un loop sin costuras.")

        trim_end = max(src_dur - 0.5, min(src_dur, 5.0))
        clean_wav = work_dir / "clean.wav"

        subprocess.run([
            "ffmpeg", "-y", "-i", str(raw_wav),
            "-af", f"atrim=start=0.5:end={trim_end},asetpts=PTS-STARTPTS",
            "-ac", "2", "-ar", "44100",
            str(clean_wav)
        ], check=True, capture_output=True)

        raw_wav.unlink(missing_ok=True)  # free disk
        clean_dur = trim_end - 0.5  # seconds

        # Step 2: build seamless loop unit (chain 8 copies with qsin 8s crossfade)
        update(status="Building loop unit", progress=15)
        XFADE = 8.0
        unit_dur = 8 * clean_dur - 7 * XFADE  # length after 8-chain
        unit_wav = work_dir / "unit.wav"

        filter_complex = "[0]asplit=8[a1][a2][a3][a4][a5][a6][a7][a8];"
        filter_complex += "[a1][a2]acrossfade=d=8:c1=qsin:c2=qsin[x1];"
        for i in range(3, 8):
            filter_complex += f"[x{i-2}][a{i}]acrossfade=d=8:c1=qsin:c2=qsin[x{i-1}];"
        filter_complex += "[x6][a8]acrossfade=d=8:c1=qsin:c2=qsin[loop]"

        subprocess.run([
            "ffmpeg", "-y", "-i", str(clean_wav),
            "-filter_complex", filter_complex,
            "-map", "[loop]",
            "-c:a", "pcm_s16le",
            str(unit_wav)
        ], check=True, capture_output=True)

        # Step 3: ONE-SHOT: stream_loop + loudnorm + fade + MP3 encode
        # No intermediate 600 MB WAV. ffmpeg streams everything through.
        update(status=f"Rendering {duration_hours}h MP3", progress=40)
        target_sec = int(duration_hours * 3600)
        fade_out_start = target_sec - 10
        out_mp3 = work_dir / "output.mp3"

        actual_unit_dur = probe_duration(unit_wav)
        print(f"[render] job={job_id} unit_dur={actual_unit_dur:.1f}s target={target_sec}s", flush=True)

        render_cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", str(unit_wav),
            "-t", str(target_sec),
            "-af", f"loudnorm=I=-18:TP=-1.5:LRA=11,afade=t=in:st=0:d=5,afade=t=out:st={fade_out_start}:d=10",
            "-c:a", "libmp3lame", "-b:a", "192k",
            "-ac", "2", "-ar", "44100",
            "-metadata", f"title={title_hint or 'Sleep Loop'} {duration_hours}h",
            "-metadata", "artist=Sleep Loop Web",
            str(out_mp3)
        ]
        result = subprocess.run(render_cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"render failed: {result.stderr.decode()[-600:]}")

        # Verify output reached target
        out_dur = probe_duration(out_mp3)
        print(f"[render] job={job_id} output_dur={out_dur:.1f}s", flush=True)
        if out_dur < target_sec * 0.95:
            raise RuntimeError(f"output is only {out_dur:.0f}s, expected {target_sec}s")

        # Compute cut points (smooth + hard) and write a .txt next to the MP3
        def fmt(sec):
            sec = max(0.0, sec)
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = sec - (h * 3600 + m * 60)
            if h > 0:
                return f"{h}:{m:02d}:{s:05.2f}"
            return f"{m}:{s:05.2f}"

        smooth_cuts = []  # inside the unit (qsin 8s crossfades)
        # inside the FIRST unit, crossfades start at these times:
        for k in range(1, 8):
            t = k * (clean_dur - XFADE) + (k - 1) * XFADE
            # crossfade spans from t to t+XFADE; midpoint is t+XFADE/2
            smooth_cuts.append(t + XFADE / 2)

        hard_cuts = []  # between units (no crossfade)
        t = unit_dur
        while t < target_sec - 1:
            hard_cuts.append(t)
            t += unit_dur

        # All-cuts across the full duration (smooth+hard, repeating per unit)
        all_smooth = []
        unit_idx = 0
        while unit_idx * unit_dur < target_sec:
            base = unit_idx * unit_dur
            for sc in smooth_cuts:
                p = base + sc
                if p < target_sec:
                    all_smooth.append(p)
            unit_idx += 1

        cuts_lines = [
            f"Sleep Loop - cut report",
            f"Source duration: {src_dur:.2f}s ({fmt(src_dur)})",
            f"Clean duration (trimmed): {clean_dur:.2f}s",
            f"Unit duration (8 copies + qsin 8s crossfades): {unit_dur:.2f}s ({fmt(unit_dur)})",
            f"Target: {duration_hours}h ({target_sec}s)",
            f"",
            f"HARD CUTS between units (listen here first — potential seams):",
        ]
        if hard_cuts:
            for i, c in enumerate(hard_cuts, 1):
                cuts_lines.append(f"  {i}. {fmt(c)}  ({c:.2f}s)")
        else:
            cuts_lines.append("  (none — target fits inside one unit)")
        cuts_lines.append("")
        cuts_lines.append(f"SMOOTH crossfades (qsin 8s, should be inaudible) — {len(all_smooth)} total:")
        for i, c in enumerate(all_smooth, 1):
            cuts_lines.append(f"  {i}. {fmt(c)}  ({c:.2f}s)")
        cuts_txt = work_dir / "cuts.txt"
        cuts_txt.write_text("\n".join(cuts_lines))

        with JOBS_LOCK:
            JOBS[job_id]["cuts_file"] = str(cuts_txt)
            JOBS[job_id]["hard_cuts"] = [fmt(c) for c in hard_cuts]
            JOBS[job_id]["smooth_cuts_count"] = len(all_smooth)

        # Cleanup intermediate WAVs to save disk
        clean_wav.unlink(missing_ok=True)
        unit_wav.unlink(missing_ok=True)

        update(status="Done", progress=100, output=str(out_mp3))

    except subprocess.CalledProcessError as e:
        err = e.stderr.decode() if e.stderr else str(e)
        update(status="Failed", error=err[-500:])
    except Exception as e:
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
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    duration = request.form.get("duration", "1")
    try:
        duration_h = float(duration)
        if duration_h not in (1.0, 8.0):
            return jsonify({"error": "Duration must be 1 or 8"}), 400
    except ValueError:
        return jsonify({"error": "Invalid duration"}), 400

    title = request.form.get("title", "")[:60]

    job_id = uuid.uuid4().hex[:12]
    ext = Path(file.filename).suffix or ".mp3"
    saved = UPLOAD_DIR / f"{job_id}{ext}"
    file.save(str(saved))

    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "Queued",
            "progress": 0,
            "error": None,
            "output": None,
            "duration_h": duration_h,
            "title": title,
            "created_at": time.time(),
        }

    t = threading.Thread(target=render_loop, args=(job_id, saved, duration_h, title), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


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
            "hard_cuts": job.get("hard_cuts", []),
            "smooth_cuts_count": job.get("smooth_cuts_count", 0),
            "has_cuts_file": bool(job.get("cuts_file")),
        })


@app.route("/download/<job_id>")
def download(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or not job["output"]:
        abort(404)
    title = (job.get("title") or "sleep-loop").replace(" ", "-")
    dl_name = f"{title}-{int(job['duration_h'])}h.mp3"
    return send_file(job["output"], as_attachment=True, download_name=dl_name, mimetype="audio/mpeg")


@app.route("/cuts/<job_id>")
def download_cuts(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or not job.get("cuts_file"):
        abort(404)
    title = (job.get("title") or "sleep-loop").replace(" ", "-")
    dl_name = f"{title}-{int(job['duration_h'])}h-cuts.txt"
    return send_file(job["cuts_file"], as_attachment=True, download_name=dl_name, mimetype="text/plain")


@app.route("/health")
def health():
    return "ok", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
