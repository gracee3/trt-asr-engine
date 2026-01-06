#!/usr/bin/env bash
set -euo pipefail

# =========================
# Configuration (env overrides)
# =========================
TRT_REPO="${TRT_REPO:-$HOME/git/trt-asr-engine}"
MAG_REPO="${MAG_REPO:-}"

LIBRISPEECH_ROOT="${LIBRISPEECH_ROOT:-}"
LIBRISPEECH_SPLIT="${LIBRISPEECH_SPLIT:-dev-clean}"
NUM_UTTS="${NUM_UTTS:-8}"
ROUNDS="${ROUNDS:-2}"

SUITE_ROOT="${SUITE_ROOT:-/tmp/magnolia_stt_suite}"

# ALSA loopback
LOOP_ID="${LOOP_ID:-LoopSTT}"
LOOP_INDEX="${LOOP_INDEX:-10}"   # used only if we try to modprobe
PLAY_PCM="${PLAY_PCM:-loopstt_play_mono16k}"   # defined via .asoundrc snippet
CAP_PCM="${CAP_PCM:-loopstt_cap_mono16k}"      # defined via .asoundrc snippet
BEEP_ENABLED="${BEEP_ENABLED:-1}"

# Magnolia run command template (required for E2E run)
# Must include placeholders: {{DEVICE}} and {{DURATION}}
MAGNOLIA_CMD_TEMPLATE="${MAGNOLIA_CMD_TEMPLATE:-}"
MAGNOLIA_DEVICE="${MAGNOLIA_DEVICE:-$CAP_PCM}"
MAGNOLIA_EXTRA_ENV="${MAGNOLIA_EXTRA_ENV:-}"     # optional, extra env vars

# Transcript extraction regex (best-effort defaults)
# You can override if Magnolia prints differently.
MAG_FINAL_REGEX="${MAG_FINAL_REGEX:-^Final:[[:space:]]*(.*)$}"

# Parakeet / debug knobs (suite variants apply these)
PARAKEET_MODEL_DIR="${PARAKEET_MODEL_DIR:-$TRT_REPO/models/parakeet-tdt-0.6b-v3}"
PARAKEET_MAX_FRAMES_PER_PUSH="${PARAKEET_MAX_FRAMES_PER_PUSH:-256}"
PARAKEET_DEBUG_BLANK_SCAN="${PARAKEET_DEBUG_BLANK_SCAN:-1}"
PARAKEET_DEBUG_EMIT_TOKENS="${PARAKEET_DEBUG_EMIT_TOKENS:-1}"

# Enable tap capture if Magnolia supports it
AUDIO_TAP_ENABLE_DEFAULT="${AUDIO_TAP_ENABLE:-1}"

# Variants
VARIANTS="${VARIANTS:-base nopunct nocache nocache_nopunct}"

# Build control
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_MAGNOLIA="${SKIP_MAGNOLIA:-0}"  # set 1 to only run dataset prep + offline checks

# =========================
# Logging helpers
# =========================
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }
die() { echo "[$(ts)] ERROR: $*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

# =========================
# Dependency checks
# =========================
need_cmd bash
need_cmd python3
need_cmd arecord
need_cmd aplay
need_cmd timeout
need_cmd awk
need_cmd sed
need_cmd grep
need_cmd find

# conversion tool: ffmpeg OR sox OR flac
HAS_FFMPEG=0
HAS_SOX=0
HAS_FLAC=0
command -v ffmpeg >/dev/null 2>&1 && HAS_FFMPEG=1
command -v sox >/dev/null 2>&1 && HAS_SOX=1
command -v flac >/dev/null 2>&1 && HAS_FLAC=1

if [[ "$HAS_FFMPEG" -eq 0 && "$HAS_SOX" -eq 0 && "$HAS_FLAC" -eq 0 ]]; then
  die "Need one of: ffmpeg, sox, or flac (for LibriSpeech conversion)"
fi

# =========================
# Resolve Magnolia repo if not set
# =========================
if [[ -z "$MAG_REPO" ]]; then
  if [[ -d "$HOME/git/magnolia" ]]; then
    MAG_REPO="$HOME/git/magnolia"
  elif [[ -d "$HOME/git/magnolia integration" ]]; then
    MAG_REPO="$HOME/git/magnolia integration"
  fi
fi

# =========================
# Check repos
# =========================
[[ -d "$TRT_REPO" ]] || die "TRT_REPO not found: $TRT_REPO"
if [[ "$SKIP_MAGNOLIA" -eq 0 ]]; then
  [[ -d "$MAG_REPO" ]] || die "MAG_REPO not found (set MAG_REPO): $MAG_REPO"
  [[ -n "$MAGNOLIA_CMD_TEMPLATE" ]] || die "MAGNOLIA_CMD_TEMPLATE not set. See AGENTS.md."
fi

# =========================
# Dataset root
# =========================
[[ -n "$LIBRISPEECH_ROOT" ]] || die "LIBRISPEECH_ROOT not set (e.g. ~/datasets/LibriSpeech)"
[[ -d "$LIBRISPEECH_ROOT" ]] || die "LIBRISPEECH_ROOT does not exist: $LIBRISPEECH_ROOT"

# =========================
# Suite run directory
# =========================
SUITE_DIR="$SUITE_ROOT/suite_$(date +%Y%m%d_%H%M%S)_pid$$"
mkdir -p "$SUITE_DIR"
log "Suite dir: $SUITE_DIR"

# =========================
# Ensure loopback module (best-effort)
# =========================
ensure_loopback() {
  if aplay -l | grep -qi "$LOOP_ID"; then
    log "ALSA loopback card appears present (matches '$LOOP_ID')."
    return 0
  fi

  # Try generic "Loopback" too (some distros ignore id=)
  if aplay -l | grep -qi "Loopback"; then
    log "ALSA loopback card present (Loopback)."
    return 0
  fi

  log "Loopback card not detected. Attempting to load snd-aloop (may prompt for sudo)."
  if command -v sudo >/dev/null 2>&1; then
    set +e
    sudo modprobe snd-aloop "index=$LOOP_INDEX" "id=$LOOP_ID" pcm_substreams=1
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
      die "modprobe snd-aloop failed. Manually run: sudo modprobe snd-aloop index=$LOOP_INDEX id=$LOOP_ID pcm_substreams=1"
    fi
  else
    die "sudo not available and loopback not present. Load manually: modprobe snd-aloop ..."
  fi

  aplay -l | grep -qiE "$LOOP_ID|Loopback" || die "snd-aloop loaded but card still not visible in aplay -l"
  log "snd-aloop loaded."
}

# =========================
# Install .asoundrc snippet (safe append; no default changes)
# =========================
ASOUNDRC_MARK_BEGIN="# BEGIN LOOPSTT (auto)"
ASOUNDRC_MARK_END="# END LOOPSTT (auto)"

install_asoundrc_snippet() {
  local asoundrc="$HOME/.asoundrc"
  if [[ -f "$asoundrc" ]] && grep -qF "$ASOUNDRC_MARK_BEGIN" "$asoundrc"; then
    log "~/.asoundrc already contains LOOPSTT snippet."
    return 0
  fi

  if [[ -f "$asoundrc" ]]; then
    cp "$asoundrc" "$SUITE_DIR/asoundrc.backup"
    log "Backed up existing ~/.asoundrc to $SUITE_DIR/asoundrc.backup"
  fi

  cat >> "$asoundrc" <<EOF

$ASOUNDRC_MARK_BEGIN
# Dedicated ALSA loopback endpoints for STT testing.
# Playback into:   $PLAY_PCM  -> hw:$LOOP_ID,0,0
# Capture from:    $CAP_PCM   -> hw:$LOOP_ID,1,0
#
# This does NOT change defaults; it only adds named PCMs.

pcm.$PLAY_PCM {
  type plug
  slave {
    pcm "hw:$LOOP_ID,0,0"
    rate 16000
    channels 1
    format S16_LE
  }
}

pcm.$CAP_PCM {
  type plug
  slave {
    pcm "hw:$LOOP_ID,1,0"
    rate 16000
    channels 1
    format S16_LE
  }
}

ctl.$PLAY_PCM {
  type hw
  card "$LOOP_ID"
}
$ASOUNDRC_MARK_END
EOF

  log "Appended LOOPSTT PCM definitions to ~/.asoundrc"
}

# =========================
# Generate a beep WAV (for loopback sanity)
# =========================
BEEP_WAV="$SUITE_DIR/beep_1khz_250ms.wav"

make_beep_wav() {
  python3 - <<PY
import wave, math, struct
out = "$BEEP_WAV"
sr = 16000
dur = 0.25
freq = 1000.0
amp = 0.2
n = int(sr*dur)
with wave.open(out, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    for i in range(n):
        s = amp*math.sin(2*math.pi*freq*i/sr)
        w.writeframes(struct.pack("<h", int(max(-1,min(1,s))*32767)))
print(out)
PY
}

# =========================
# Verify loopback (beep play -> record)
# =========================
verify_loopback() {
  local rec="$SUITE_DIR/loopback_check.wav"
  log "Loopback sanity check: record from $CAP_PCM while playing beep into $PLAY_PCM"
  rm -f "$rec"

  # record in background
  ( arecord -q -D "$CAP_PCM" -f S16_LE -r 16000 -c 1 -d 2 "$rec" ) &
  local rec_pid=$!
  sleep 0.15

  if [[ "$BEEP_ENABLED" -eq 1 ]]; then
    aplay -q -D "$PLAY_PCM" "$BEEP_WAV" || true
    sleep 0.1
    aplay -q -D "$PLAY_PCM" "$BEEP_WAV" || true
  fi

  wait "$rec_pid" || true

  [[ -s "$rec" ]] || die "Loopback check failed: no recorded file at $rec"

  # quick energy check
  python3 - <<PY
import wave, struct, math
p = "$rec"
w = wave.open(p, "rb")
n = w.getnframes()
x = w.readframes(n)
w.close()
s = struct.unpack("<%dh" % n, x)
rms = math.sqrt(sum(v*v for v in s)/max(1,n))/32768.0
peak = max(abs(v) for v in s)/32768.0
print(f"[loopback_check] frames={n} rms={rms:.6f} peak={peak:.6f}")
if peak < 0.01:
    raise SystemExit("Loopback check: peak too low (silent?)")
PY

  log "Loopback sanity check passed."
}

# =========================
# Build steps (best-effort)
# =========================
build_trt_repo() {
  if [[ "$SKIP_BUILD" -eq 1 ]]; then
    log "SKIP_BUILD=1; skipping builds."
    return 0
  fi

  # C++ build
  if [[ -d "$TRT_REPO/cpp" ]]; then
    log "Building C++ (cmake)..."
    mkdir -p "$TRT_REPO/cpp/build"
    if [[ ! -f "$TRT_REPO/cpp/build/build.ninja" && ! -f "$TRT_REPO/cpp/build/Makefile" ]]; then
      (cd "$TRT_REPO/cpp" && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release) || die "CMake configure failed"
    fi
    (cd "$TRT_REPO/cpp" && cmake --build build -j) || die "CMake build failed"
  fi

  # Rust CLI build (if present)
  if [[ -d "$TRT_REPO/rust" && -f "$TRT_REPO/rust/Cargo.toml" ]]; then
    log "Building Rust CLI (cargo)..."
    (cd "$TRT_REPO/rust" && cargo build -p cli) || die "cargo build -p cli failed"
  elif [[ -f "$TRT_REPO/Cargo.toml" ]]; then
    log "Building Rust CLI from repo root (cargo)..."
    (cd "$TRT_REPO" && cargo build -p cli) || die "cargo build -p cli failed"
  else
    log "Rust CLI not detected (Cargo.toml not found). Skipping."
  fi
}

# =========================
# Locate TRT CLI binary
# =========================
find_trt_cli() {
  local c1="$TRT_REPO/target/release/cli"
  local c2="$TRT_REPO/target/debug/cli"
  local c3="$TRT_REPO/rust/target/release/cli"
  local c4="$TRT_REPO/rust/target/debug/cli"
  if [[ -x "$c1" ]]; then echo "$c1"; return 0; fi
  if [[ -x "$c2" ]]; then echo "$c2"; return 0; fi
  if [[ -x "$c3" ]]; then echo "$c3"; return 0; fi
  if [[ -x "$c4" ]]; then echo "$c4"; return 0; fi
  die "Could not find cli binary. Build it or set PATH. Looked in target/{debug,release}/cli."
}

# =========================
# LibriSpeech manifest generation
# =========================
MANIFEST="$SUITE_DIR/manifest.tsv"
DATASET_DIR="$SUITE_DIR/dataset"

make_manifest() {
  log "Generating LibriSpeech manifest (split=$LIBRISPEECH_SPLIT, num_utts=$NUM_UTTS) ..."
  mkdir -p "$DATASET_DIR"
  python3 "$TRT_REPO/tools/stt_suite/make_librispeech_manifest.py" \
    --librispeech-root "$LIBRISPEECH_ROOT" \
    --split "$LIBRISPEECH_SPLIT" \
    --num-utts "$NUM_UTTS" \
    --out-dir "$DATASET_DIR" \
    --manifest "$MANIFEST"
  [[ -s "$MANIFEST" ]] || die "Manifest not created: $MANIFEST"
  log "Manifest written: $MANIFEST"
}

# =========================
# Extract transcript from logs (best-effort)
# =========================
extract_final_from_log() {
  local log_path="$1"
  python3 - <<PY
import re, sys
p = sys.argv[1]
rx = re.compile(r"""$MAG_FINAL_REGEX""")
last = ""
try:
    with open(p, "r", errors="ignore") as f:
        for line in f:
            m = rx.match(line.rstrip("\n"))
            if m:
                last = m.group(1).strip()
except FileNotFoundError:
    pass
print(last)
PY "$log_path"
}

# =========================
# WER helper
# =========================
wer() {
  local ref="$1"
  local hyp="$2"
  python3 - <<'PY'
import re, sys

def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

ref = norm(sys.argv[1])
hyp = norm(sys.argv[2])

r = ref.split() if ref else []
h = hyp.split() if hyp else []

# edit distance
dp = list(range(len(h)+1))
for i, rw in enumerate(r, 1):
    prev = dp[0]
    dp[0] = i
    for j, hw in enumerate(h, 1):
        cur = dp[j]
        cost = 0 if rw == hw else 1
        dp[j] = min(
            dp[j] + 1,       # del
            dp[j-1] + 1,     # ins
            prev + cost      # sub
        )
        prev = cur

ed = dp[-1]
den = max(1, len(r))
print(f"{ed/den:.6f}")
PY "$ref" "$hyp"
}

# =========================
# Variant env
# =========================
variant_env() {
  local variant="$1"
  # base env always
  export PARAKEET_MODEL_DIR
  export PARAKEET_MAX_FRAMES_PER_PUSH
  export PARAKEET_DEBUG_BLANK_SCAN
  export PARAKEET_DEBUG_EMIT_TOKENS

  # reset toggles
  unset PARAKEET_DISABLE_CACHE || true
  unset PARAKEET_DISABLE_PUNCT_SUPPRESSION || true

  case "$variant" in
    base)
      ;;
    nopunct)
      export PARAKEET_DISABLE_PUNCT_SUPPRESSION=1
      ;;
    nocache)
      export PARAKEET_DISABLE_CACHE=1
      ;;
    nocache_nopunct)
      export PARAKEET_DISABLE_CACHE=1
      export PARAKEET_DISABLE_PUNCT_SUPPRESSION=1
      ;;
    *)
      die "Unknown variant: $variant"
      ;;
  esac
}

# =========================
# Main: prepare + run
# =========================
log "TRT_REPO=$TRT_REPO"
log "MAG_REPO=$MAG_REPO"
log "LIBRISPEECH_ROOT=$LIBRISPEECH_ROOT"
log "LIBRISPEECH_SPLIT=$LIBRISPEECH_SPLIT"

ensure_loopback
install_asoundrc_snippet
make_beep_wav
verify_loopback
build_trt_repo
make_manifest

TRT_CLI="$(find_trt_cli)"
log "Using TRT CLI: $TRT_CLI"
log "Using model dir: $PARAKEET_MODEL_DIR"

RESULTS_TSV="$SUITE_DIR/results.tsv"
echo -e "variant\tround\tutt_id\tduration_sec\tref_text\tcli_text\tcli_wer\tmagnolia_text\tmagnolia_wer\tutt_dir" > "$RESULTS_TSV"

# Read manifest
# Columns: utt_id \t wav_path \t duration_sec \t ref_text
tail -n +2 "$MANIFEST" > "$SUITE_DIR/manifest.body.tsv"

for variant in $VARIANTS; do
  log "=== VARIANT: $variant ==="
  variant_env "$variant"

  for round in $(seq 1 "$ROUNDS"); do
    log "--- round $round/$ROUNDS (variant=$variant) ---"

    while IFS=$'\t' read -r utt_id wav_path duration_sec ref_text; do
      [[ -n "$utt_id" ]] || continue

      utt_dir="$SUITE_DIR/$variant/round_$round/$utt_id"
      mkdir -p "$utt_dir"
      cp -f "$wav_path" "$utt_dir/input.wav" || true

      # --- Offline TRT CLI on WAV ---
      cli_log="$utt_dir/trt_cli.log"
      log "[${variant} r${round} $utt_id] TRT CLI (offline wav) ..."
      set +e
      ( "$TRT_CLI" "$wav_path" --model-dir "$PARAKEET_MODEL_DIR" -v ) >"$cli_log" 2>&1
      cli_rc=$?
      set -e
      if [[ "$cli_rc" -ne 0 ]]; then
        log "[${variant} r${round} $utt_id] TRT CLI exit=$cli_rc (continuing). See $cli_log"
      fi
      cli_text="$(grep -E '^Transcript:' "$cli_log" | tail -n 1 | sed -E 's/^Transcript:[[:space:]]*//')"
      cli_wer_val="$(wer "$ref_text" "$cli_text")"

      magn_text=""
      magn_wer_val=""

      # --- Magnolia E2E via loopback ---
      if [[ "$SKIP_MAGNOLIA" -eq 0 ]]; then
        mag_log="$utt_dir/magnolia.log"

        # duration padding
        dur_pad="$(python3 - <<PY
import math
d=float("$duration_sec")
print(int(math.ceil(d+6.0)))
PY
)"
        cmd="${MAGNOLIA_CMD_TEMPLATE//\{\{DEVICE\}\}/$MAGNOLIA_DEVICE}"
        cmd="${cmd//\{\{DURATION\}\}/$dur_pad}"

        log "[${variant} r${round} $utt_id] Magnolia start (timeout=${dur_pad}s) ..."
        (
          export AUDIO_TAP_ENABLE="$AUDIO_TAP_ENABLE_DEFAULT"
          export AUDIO_TAP_DIR="$utt_dir/taps"
          export AUDIO_TAP_FEATURES=1
          export AUDIO_TAP_CAPTURE=1
          export AUDIO_TAP_POST_DSP=1
          export AUDIO_TAP_ENABLE=1

          # Parakeet env is already set by variant_env()
          # Extra user-specified env injection:
          if [[ -n "$MAGNOLIA_EXTRA_ENV" ]]; then
            eval "export $MAGNOLIA_EXTRA_ENV"
          fi

          # Start Magnolia under timeout, capture stdout/stderr
          timeout "${dur_pad}s" bash -lc "$cmd" >"$mag_log" 2>&1
        ) &
        mag_pid=$!

        # give Magnolia a moment to initialize
        sleep 0.3

        # Play beep + utterance + beep into loopback playback endpoint
        if [[ "$BEEP_ENABLED" -eq 1 ]]; then
          aplay -q -D "$PLAY_PCM" "$BEEP_WAV" || true
          sleep 0.05
        fi

        log "[${variant} r${round} $utt_id] Playback into loopback ($PLAY_PCM): $(basename "$wav_path")"
        aplay -q -D "$PLAY_PCM" "$wav_path" || true

        if [[ "$BEEP_ENABLED" -eq 1 ]]; then
          sleep 0.05
          aplay -q -D "$PLAY_PCM" "$BEEP_WAV" || true
        fi

        # wait for Magnolia (timeout will stop it)
        wait "$mag_pid" || true

        magn_text="$(extract_final_from_log "$mag_log")"
        magn_wer_val="$(wer "$ref_text" "$magn_text")"

        # Optional: if Magnolia wrote tap_FEATURES.json, run replay too (kept as artifact)
        tap_json="$(find "$utt_dir" -name 'tap_FEATURES.json' -print -quit 2>/dev/null || true)"
        if [[ -n "$tap_json" && -f "$tap_json" ]]; then
          replay_log="$utt_dir/feature_replay.log"
          log "[${variant} r${round} $utt_id] Feature replay from tap_FEATURES.json ..."
          set +e
          ( "$TRT_CLI" "$tap_json" --features-input --model-dir "$PARAKEET_MODEL_DIR" -v ) >"$replay_log" 2>&1
          set -e
        fi
      fi

      echo -e "${variant}\t${round}\t${utt_id}\t${duration_sec}\t${ref_text}\t${cli_text}\t${cli_wer_val}\t${magn_text}\t${magn_wer_val}\t${utt_dir}" >> "$RESULTS_TSV"

      log "[${variant} r${round} $utt_id] ref='${ref_text}'"
      log "[${variant} r${round} $utt_id] cli='${cli_text}' wer=${cli_wer_val}"
      if [[ "$SKIP_MAGNOLIA" -eq 0 ]]; then
        log "[${variant} r${round} $utt_id] mag='${magn_text}' wer=${magn_wer_val}"
      fi
    done < "$SUITE_DIR/manifest.body.tsv"
  done
done

# Summary aggregation
SUMMARY="$SUITE_DIR/summary.txt"
python3 - <<'PY' "$RESULTS_TSV" "$SUMMARY"
import sys, csv, statistics
tsv_path = sys.argv[1]
out_path = sys.argv[2]

rows=[]
with open(tsv_path, newline="", errors="ignore") as f:
    r=csv.DictReader(f, delimiter="\t")
    for row in r:
        rows.append(row)

def ffloat(x):
    try:
        return float(x)
    except:
        return None

by_var={}
for row in rows:
    v=row["variant"]
    by_var.setdefault(v, {"cli":[], "mag":[]})
    cw=ffloat(row.get("cli_wer",""))
    mw=ffloat(row.get("magnolia_wer",""))
    if cw is not None: by_var[v]["cli"].append(cw)
    if mw is not None: by_var[v]["mag"].append(mw)

lines=[]
lines.append(f"rows={len(rows)} variants={sorted(by_var.keys())}")
lines.append("")
for v in sorted(by_var.keys()):
    cli=by_var[v]["cli"]
    mag=by_var[v]["mag"]
    if cli:
        lines.append(f"[{v}] CLI WER mean={statistics.mean(cli):.4f} median={statistics.median(cli):.4f} n={len(cli)}")
    else:
        lines.append(f"[{v}] CLI WER: n=0")
    if mag:
        lines.append(f"[{v}] MAG WER mean={statistics.mean(mag):.4f} median={statistics.median(mag):.4f} n={len(mag)}")
    else:
        lines.append(f"[{v}] MAG WER: n=0")
    lines.append("")

with open(out_path,"w") as f:
    f.write("\n".join(lines)+"\n")

print("\n".join(lines))
PY

log "Done."
log "Results TSV: $RESULTS_TSV"
log "Summary:     $SUMMARY"
log "Suite dir:   $SUITE_DIR"