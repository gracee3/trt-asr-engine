use clap::Parser;
use features::{
    FeatureConfig,
    LogMelExtractor,
    StreamingLogMelExtractor,
    compute_per_feature_stats,
    apply_per_feature_norm,
    NormalizationMode,
};
use parakeet_trt::{ParakeetSessionSafe, TranscriptionEvent};
use serde_json::{Value, json};
use std::env;
use std::path::PathBuf;
use std::time::Duration;
use std::thread;
use std::fs::File;
use std::io::{Read, BufReader};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(help = "Path to input file (WAV, raw PCM, or feature dump)")]
    input_path: PathBuf,

    #[arg(long, help = "Path to model directory")]
    model_dir: PathBuf,

    #[arg(long, help = "Simulated streaming interval in seconds")]
    stream_sim: Option<f32>,

    #[arg(long, help = "Disable sleeping between stream chunks (no-sleep replay)")]
    no_sleep: bool,

    #[arg(long, default_value_t = 0)]
    device_id: i32,

    // Replay harness options
    #[arg(long, help = "Input is raw PCM (f32le, 16kHz mono) instead of WAV")]
    raw_pcm: bool,

    #[arg(long, help = "Input is raw PCM with this sample rate (requires --raw-pcm)")]
    sample_rate: Option<u32>,

    #[arg(long, help = "Input is pre-computed features (tap raw/json, bypass feature extraction)")]
    features_input: bool,

    #[arg(long, help = "Number of mel bins when using --features-input (overrides JSON sidecar if set)")]
    n_mels: Option<usize>,

    #[arg(long, short, help = "Verbose output (log all events and timing)")]
    verbose: bool,

    #[arg(long, help = "Dump computed features to this file (f32le, [C,T])")]
    dump_features: Option<PathBuf>,

    #[arg(long, help = "Dump per-chunk frame counts to this JSON file (streaming only)")]
    dump_chunk_frames: Option<PathBuf>,

    #[arg(long, value_parser = ["none", "per_feature"], help = "Feature normalization (overrides PARAKEET_FEATURE_NORM)")]
    feature_norm: Option<String>,

    #[arg(long, help = "Compute/dump features but skip runtime inference")]
    features_only: bool,
}

#[derive(Debug, Default)]
struct FeatureMeta {
    mel_bins: Option<usize>,
    num_frames: Option<usize>,
    layout: Option<String>,
    format: Option<String>,
    kind: Option<String>,
    shape: Option<Vec<usize>>,
}

fn load_raw_pcm(path: &PathBuf) -> anyhow::Result<Vec<f32>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;

    if bytes.len() % 4 != 0 {
        anyhow::bail!("Raw PCM file size must be multiple of 4 (f32le)");
    }

    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(samples)
}

fn frames_major_to_bins_major(features_tc: &[f32], n_mels: usize, frames: usize) -> Vec<f32> {
    // Input is [T, C] (frame-major), output is [C, T] (mel-major) to match encoder [B,C,T].
    let mut out = vec![0.0f32; n_mels * frames];
    for t in 0..frames {
        let in_base = t * n_mels;
        for m in 0..n_mels {
            out[m * frames + t] = features_tc[in_base + m];
        }
    }
    out
}

fn load_features_raw(path: &PathBuf, n_mels: usize) -> anyhow::Result<(Vec<f32>, usize)> {
    // Load pre-computed features (f32le) and infer frames from n_mels.
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;

    if bytes.len() % 4 != 0 {
        anyhow::bail!("Feature file size must be multiple of 4 (f32le)");
    }

    let total_floats = bytes.len() / 4;
    if total_floats % n_mels != 0 {
        anyhow::bail!("Feature file size {} floats not divisible by n_mels={}", total_floats, n_mels);
    }

    let num_frames = total_floats / n_mels;
    let features: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok((features, num_frames))
}

fn resolve_feature_paths(input_path: &PathBuf) -> (PathBuf, Option<PathBuf>) {
    let ext = input_path.extension().and_then(|s| s.to_str()).unwrap_or("");
    if ext.eq_ignore_ascii_case("json") {
        let mut raw_path = input_path.clone();
        raw_path.set_extension("raw");
        return (raw_path, Some(input_path.clone()));
    }

    let mut json_path = input_path.clone();
    json_path.set_extension("json");
    if json_path.exists() {
        (input_path.clone(), Some(json_path))
    } else {
        (input_path.clone(), None)
    }
}

fn parse_feature_meta(path: &PathBuf) -> anyhow::Result<FeatureMeta> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let value: Value = serde_json::from_reader(reader)?;

    let mut meta = FeatureMeta::default();
    meta.kind = value.get("kind").and_then(|v| v.as_str()).map(|s| s.to_string());
    meta.format = value.get("format").and_then(|v| v.as_str()).map(|s| s.to_string());
    meta.layout = value.get("layout")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_ascii_lowercase());
    meta.mel_bins = value.get("mel_bins").and_then(|v| v.as_u64()).map(|v| v as usize);
    meta.num_frames = value.get("num_frames").and_then(|v| v.as_u64()).map(|v| v as usize);
    meta.shape = value.get("shape").and_then(|v| v.as_array()).map(|arr| {
        arr.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect()
    });

    if (meta.mel_bins.is_none() || meta.num_frames.is_none()) && meta.shape.as_ref().map(|s| s.len()) == Some(2) {
        let shape = meta.shape.as_ref().unwrap();
        match meta.layout.as_deref() {
            Some("bins_major") => {
                if meta.mel_bins.is_none() { meta.mel_bins = Some(shape[0]); }
                if meta.num_frames.is_none() { meta.num_frames = Some(shape[1]); }
            }
            Some("frames_major") => {
                if meta.mel_bins.is_none() { meta.mel_bins = Some(shape[1]); }
                if meta.num_frames.is_none() { meta.num_frames = Some(shape[0]); }
            }
            _ => {}
        }
    }

    Ok(meta)
}

fn dump_features_to_file(features_bct: &[f32], path: &PathBuf) -> anyhow::Result<()> {
    use std::io::Write;
    let mut file = File::create(path)?;
    for &f in features_bct {
        file.write_all(&f.to_le_bytes())?;
    }
    Ok(())
}

fn slice_bct_frames(features_bct: &[f32], n_mels: usize, total_frames: usize, start: usize, frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_mels * frames];
    for m in 0..n_mels {
        let src_offset = m * total_frames + start;
        let dst_offset = m * frames;
        out[dst_offset..dst_offset + frames]
            .copy_from_slice(&features_bct[src_offset..src_offset + frames]);
    }
    out
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let start_time = std::time::Instant::now();
    let env_norm = env::var("PARAKEET_FEATURE_NORM").ok();
    let feature_norm = args
        .feature_norm
        .or(env_norm)
        .unwrap_or_else(|| "none".to_string());
    let feature_norm = match feature_norm.as_str() {
        "none" => NormalizationMode::None,
        "per_feature" => NormalizationMode::PerFeature,
        other => anyhow::bail!("Unsupported feature normalization: {other}"),
    };

    if args.verbose {
        eprintln!("[replay] Input: {:?}", args.input_path);
        eprintln!("[replay] Model: {:?}", args.model_dir);
        eprintln!("[replay] Mode: {}", if args.features_input { "features" } else if args.raw_pcm { "raw_pcm" } else { "wav" });
        eprintln!("[replay] Feature normalization: {:?}", feature_norm);
    }

    // Handle direct feature input (bypass audio loading and feature extraction)
    if args.features_input {
        let input_ext = args.input_path.extension().and_then(|s| s.to_str()).unwrap_or("");
        let input_is_json = input_ext.eq_ignore_ascii_case("json");
        let (feature_path, meta_path) = resolve_feature_paths(&args.input_path);
        if !feature_path.exists() {
            anyhow::bail!("Feature raw file not found: {:?}", feature_path);
        }

        let meta = match meta_path {
            Some(ref path) if path.exists() => Some(parse_feature_meta(path)?),
            Some(ref path) if input_is_json => {
                anyhow::bail!("Feature JSON not found: {:?}", path);
            }
            _ => None,
        };

        let mut n_mels = args.n_mels.unwrap_or(128);
        let mut layout = "bins_major".to_string();
        let mut meta_num_frames = None;

        if let Some(ref m) = meta {
            if let Some(kind) = m.kind.as_deref() {
                if kind != "mel_features" && args.verbose {
                    eprintln!("[replay] Warning: JSON kind='{}' (expected mel_features)", kind);
                }
            }
            if let Some(format) = m.format.as_deref() {
                if format != "f32le" {
                    anyhow::bail!("Feature JSON format '{}' not supported (expected f32le)", format);
                }
            }
            if args.n_mels.is_none() {
                if let Some(bins) = m.mel_bins {
                    n_mels = bins;
                }
            }
            if let Some(l) = m.layout.as_ref() {
                layout = l.clone();
            }
            meta_num_frames = m.num_frames;
        }

        let (features_raw, num_frames) = load_features_raw(&feature_path, n_mels)?;
        if let Some(expected_frames) = meta_num_frames {
            if expected_frames != num_frames && args.verbose {
                eprintln!("[replay] Warning: JSON num_frames={} but raw implies {}", expected_frames, num_frames);
            }
        }

        let features_bct = if layout == "frames_major" {
            frames_major_to_bins_major(&features_raw, n_mels, num_frames)
        } else {
            if layout != "bins_major" && args.verbose {
                eprintln!("[replay] Warning: Unknown layout '{}', assuming bins_major", layout);
            }
            features_raw
        };

        if args.verbose {
            if let Some(ref path) = meta_path {
                eprintln!("[replay] Feature meta: json={:?} layout={} mel_bins={}", path, layout, n_mels);
            }
            eprintln!("[replay] Loaded {} frames of {} mel features", num_frames, n_mels);

            // Compute stats on features
            let mut nan_ct = 0;
            let mut inf_ct = 0;
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            for &v in &features_bct {
                if v.is_nan() { nan_ct += 1; }
                else if v.is_infinite() { inf_ct += 1; }
                else {
                    min_v = min_v.min(v);
                    max_v = max_v.max(v);
                }
            }
            eprintln!("[replay] Feature stats: nan={} inf={} min={:.4} max={:.4}", nan_ct, inf_ct, min_v, max_v);
        }

        let session = ParakeetSessionSafe::new(
            args.model_dir.to_str().unwrap(),
            args.device_id,
            true,
        )?;

        println!("Starting transcription (feature replay)...");
        let max_chunk_frames = 256usize;
        if num_frames > max_chunk_frames && args.verbose {
            eprintln!("[replay] Chunking features: {} frames per push", max_chunk_frames);
        }

        if num_frames <= max_chunk_frames {
            session.push_features(&features_bct, num_frames)?;
            while let Some(event) = session.poll_event() {
                match event {
                    TranscriptionEvent::FinalText { text, .. } => println!("Transcript: {}", text),
                    TranscriptionEvent::PartialText { text, .. } if args.verbose => eprintln!("[replay] Partial: {}", text),
                    TranscriptionEvent::Error { message } => eprintln!("Error: {}", message),
                    _ => {}
                }
            }
        } else {
            let mut start = 0usize;
            let mut chunk_idx = 0usize;
            while start < num_frames {
                let frames = std::cmp::min(max_chunk_frames, num_frames - start);
                let chunk_bct = slice_bct_frames(&features_bct, n_mels, num_frames, start, frames);
                if args.verbose {
                    eprintln!("[replay] feature_chunk={} start={} frames={}", chunk_idx, start, frames);
                }
                session.push_features(&chunk_bct, frames)?;
                while let Some(event) = session.poll_event() {
                    match event {
                        TranscriptionEvent::FinalText { text, .. } => println!("Transcript: {}", text),
                        TranscriptionEvent::PartialText { text, .. } if args.verbose => eprintln!("[replay] Partial: {}", text),
                        TranscriptionEvent::Error { message } => eprintln!("Error: {}", message),
                        _ => {}
                    }
                }
                start += frames;
                chunk_idx += 1;
            }
        }

        if args.verbose {
            eprintln!("[replay] Completed in {:.2}s", start_time.elapsed().as_secs_f32());
        }
        return Ok(());
    }

    // 1. Load Audio (WAV or raw PCM)
    let audio: Vec<f32> = if args.raw_pcm {
        let samples = load_raw_pcm(&args.input_path)?;
        if args.verbose {
            let sr = args.sample_rate.unwrap_or(16000);
            eprintln!("[replay] Loaded {} samples ({:.2}s at {} Hz)",
                     samples.len(), samples.len() as f32 / sr as f32, sr);
        }
        samples
    } else {
        let mut reader = hound::WavReader::open(&args.input_path)?;
        let spec = reader.spec();
        if spec.sample_rate != 16000 || spec.channels != 1 {
            anyhow::bail!("WAV must be 16kHz mono (got {} Hz, {} channels)", spec.sample_rate, spec.channels);
        }
        if args.verbose {
            eprintln!("[replay] WAV: {} Hz, {} ch, {} bits", spec.sample_rate, spec.channels, spec.bits_per_sample);
        }
        match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
            hound::SampleFormat::Int => {
                let bits = spec.bits_per_sample;
                if bits == 0 || bits > 32 {
                    anyhow::bail!("Unsupported PCM bit depth: {}", bits);
                }
                let denom = (1u64 << (bits - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.unwrap() as f32 / denom)
                    .collect()
            }
        }
    };

    if args.verbose {
        // Compute audio stats
        let mut nan_ct = 0;
        let mut peak = 0.0f32;
        let mut sum_sq = 0.0f64;
        for &v in &audio {
            if v.is_nan() { nan_ct += 1; }
            else {
                peak = peak.max(v.abs());
                sum_sq += (v as f64).powi(2);
            }
        }
        let rms = (sum_sq / audio.len() as f64).sqrt();
        eprintln!("[replay] Audio stats: samples={} peak={:.4} rms={:.6} nan={}",
                 audio.len(), peak, rms, nan_ct);
    }

    // 2. Setup Feature Extractor
    let config = FeatureConfig::default();
    let extractor = LogMelExtractor::new(config);
    let mut stream_extractor = StreamingLogMelExtractor::new(config);
    let n_mels = extractor.n_mels();
    let mut per_feature_stats = None;
    let mut offline_features_tc = None;

    if feature_norm == NormalizationMode::PerFeature {
        let features_tc = extractor.compute(&audio);
        let total_frames = features_tc.len() / n_mels;
        per_feature_stats = Some(compute_per_feature_stats(&features_tc, n_mels, total_frames));
        if args.stream_sim.is_none() {
            offline_features_tc = Some(features_tc);
        }
    }

    // 3. Setup Runtime (optional)
    let session = if args.features_only {
        None
    } else {
        Some(ParakeetSessionSafe::new(
            args.model_dir.to_str().unwrap(),
            args.device_id,
            true,
        )?)
    };

    if session.is_some() {
        println!("Starting transcription...");
    }

    if let Some(interval) = args.stream_sim {
        // Simulated streaming
        let samples_per_chunk = (interval * 16000.0) as usize;
        let mut pos = 0;
        let mut chunk_idx = 0;
        let mut all_features_tc = Vec::new();
        let mut chunk_frames_list: Vec<usize> = Vec::new();

        while pos < audio.len() {
            let end = (pos + samples_per_chunk).min(audio.len());
            let chunk = &audio[pos..end];

            let mut features_tc = stream_extractor.push(chunk);
            let num_frames = features_tc.len() / n_mels;

            if num_frames > 0 {
                chunk_frames_list.push(num_frames);
                if feature_norm == NormalizationMode::PerFeature {
                    if let Some(stats) = per_feature_stats.as_ref() {
                        apply_per_feature_norm(&mut features_tc, n_mels, num_frames, stats);
                    }
                }
                let features_bct = frames_major_to_bins_major(&features_tc, n_mels, num_frames);

                if args.dump_features.is_some() {
                    all_features_tc.extend_from_slice(&features_tc);
                }

                if args.verbose {
                    eprintln!("[replay] chunk={} pos={} samples={} frames={}",
                             chunk_idx, pos, chunk.len(), num_frames);
                }

                if let Some(ref session) = session {
                    session.push_features(&features_bct, num_frames)?;

                    // Poll for events
                    while let Some(event) = session.poll_event() {
                        match event {
                            TranscriptionEvent::PartialText { text, .. } => {
                                if args.verbose {
                                    eprintln!("[replay] Partial: {}", text);
                                } else {
                                    print!("\rPartial: {}", text);
                                    use std::io::{self, Write};
                                    io::stdout().flush().unwrap();
                                }
                            }
                            TranscriptionEvent::FinalText { text, .. } => {
                                println!("\nFinal: {}", text);
                            }
                            TranscriptionEvent::Error { message } => {
                                eprintln!("\nError: {}", message);
                            }
                        }
                    }
                }
            }

            pos = end;
            chunk_idx += 1;
            if !args.no_sleep {
                thread::sleep(Duration::from_secs_f32(interval));
            }
        }

        // Flush remaining frames with right padding to match offline framing.
        let mut tail_features_tc = stream_extractor.finalize();
        let tail_frames = tail_features_tc.len() / n_mels;
        if tail_frames > 0 {
            chunk_frames_list.push(tail_frames);
            if feature_norm == NormalizationMode::PerFeature {
                if let Some(stats) = per_feature_stats.as_ref() {
                    apply_per_feature_norm(&mut tail_features_tc, n_mels, tail_frames, stats);
                }
            }
            let features_bct = frames_major_to_bins_major(&tail_features_tc, n_mels, tail_frames);

            if args.dump_features.is_some() {
                all_features_tc.extend_from_slice(&tail_features_tc);
            }

            if args.verbose {
                eprintln!("[replay] tail_chunk={} frames={}", chunk_idx, tail_frames);
            }

            if let Some(ref session) = session {
                session.push_features(&features_bct, tail_frames)?;
                while let Some(event) = session.poll_event() {
                    match event {
                        TranscriptionEvent::PartialText { text, .. } => {
                            if args.verbose {
                                eprintln!("[replay] Partial: {}", text);
                            } else {
                                print!("\rPartial: {}", text);
                                use std::io::{self, Write};
                                io::stdout().flush().unwrap();
                            }
                        }
                        TranscriptionEvent::FinalText { text, .. } => {
                            println!("\nFinal: {}", text);
                        }
                        TranscriptionEvent::Error { message } => {
                            eprintln!("\nError: {}", message);
                        }
                    }
                }
            }
        }

        // Dump features if requested
        if let Some(ref dump_path) = args.dump_features {
            let total_frames = all_features_tc.len() / n_mels;
            let features_bct = frames_major_to_bins_major(&all_features_tc, n_mels, total_frames);
            dump_features_to_file(&features_bct, dump_path)?;
            if args.verbose {
                eprintln!("[replay] Dumped {} frames to {:?}", total_frames, dump_path);
            }
        }

        if let Some(ref dump_path) = args.dump_chunk_frames {
            let total_frames = chunk_frames_list.iter().sum::<usize>();
            let payload = json!({
                "chunk_frames": chunk_frames_list,
                "total_frames": total_frames,
                "mel_bins": n_mels,
                "layout": "bins_major",
            });
            let file = File::create(dump_path)?;
            serde_json::to_writer_pretty(file, &payload)?;
            if args.verbose {
                eprintln!("[replay] Dumped chunk frames to {:?}", dump_path);
            }
        }
    } else {
        // Offline whole file
        let mut features_tc = if let Some(features_tc) = offline_features_tc {
            features_tc
        } else {
            extractor.compute(&audio)
        };
        let num_frames = features_tc.len() / n_mels;
        if feature_norm == NormalizationMode::PerFeature {
            if let Some(stats) = per_feature_stats.as_ref() {
                apply_per_feature_norm(&mut features_tc, n_mels, num_frames, stats);
            }
        }
        let features_bct = frames_major_to_bins_major(&features_tc, n_mels, num_frames);

        if args.verbose {
            eprintln!("[replay] Computed {} feature frames", num_frames);

            // Compute feature stats
            let mut nan_ct = 0;
            let mut inf_ct = 0;
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            for &v in &features_bct {
                if v.is_nan() { nan_ct += 1; }
                else if v.is_infinite() { inf_ct += 1; }
                else {
                    min_v = min_v.min(v);
                    max_v = max_v.max(v);
                }
            }
            eprintln!("[replay] Feature stats: nan={} inf={} min={:.4} max={:.4}", nan_ct, inf_ct, min_v, max_v);
        }

        // Dump features if requested
        if let Some(ref dump_path) = args.dump_features {
            dump_features_to_file(&features_bct, dump_path)?;
            if args.verbose {
                eprintln!("[replay] Dumped {} frames to {:?}", num_frames, dump_path);
            }
        }

        if let Some(ref session) = session {
            session.push_features(&features_bct, num_frames)?;

            while let Some(event) = session.poll_event() {
                match event {
                    TranscriptionEvent::FinalText { text, .. } => println!("Transcript: {}", text),
                    TranscriptionEvent::PartialText { text, .. } if args.verbose => eprintln!("[replay] Partial: {}", text),
                    TranscriptionEvent::Error { message } => eprintln!("Error: {}", message),
                    _ => {}
                }
            }
        }
    }

    if args.verbose {
        eprintln!("[replay] Completed in {:.2}s", start_time.elapsed().as_secs_f32());
    }

    Ok(())
}
