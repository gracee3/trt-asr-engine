use clap::Parser;
use features::{FeatureConfig, LogMelExtractor};
use parakeet_trt::{ParakeetSessionSafe, TranscriptionEvent};
use serde_json::Value;
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

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let start_time = std::time::Instant::now();

    if args.verbose {
        eprintln!("[replay] Input: {:?}", args.input_path);
        eprintln!("[replay] Model: {:?}", args.model_dir);
        eprintln!("[replay] Mode: {}", if args.features_input { "features" } else if args.raw_pcm { "raw_pcm" } else { "wav" });
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
        session.push_features(&features_bct, num_frames)?;

        while let Some(event) = session.poll_event() {
            match event {
                TranscriptionEvent::FinalText { text, .. } => println!("Transcript: {}", text),
                TranscriptionEvent::PartialText { text, .. } if args.verbose => eprintln!("[replay] Partial: {}", text),
                TranscriptionEvent::Error { message } => eprintln!("Error: {}", message),
                _ => {}
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
    let n_mels = extractor.n_mels();

    // 3. Setup Runtime
    let session = ParakeetSessionSafe::new(
        args.model_dir.to_str().unwrap(),
        args.device_id,
        true,
    )?;

    println!("Starting transcription...");

    if let Some(interval) = args.stream_sim {
        // Simulated streaming
        let samples_per_chunk = (interval * 16000.0) as usize;
        let mut pos = 0;
        let mut chunk_idx = 0;
        let mut all_features_bct = Vec::new();

        while pos < audio.len() {
            let end = (pos + samples_per_chunk).min(audio.len());
            let chunk = &audio[pos..end];

            let features_tc = extractor.compute(chunk);
            let num_frames = features_tc.len() / n_mels;

            if num_frames > 0 {
                let features_bct = frames_major_to_bins_major(&features_tc, n_mels, num_frames);

                if args.dump_features.is_some() {
                    all_features_bct.extend_from_slice(&features_bct);
                }

                if args.verbose {
                    eprintln!("[replay] chunk={} pos={} samples={} frames={}",
                             chunk_idx, pos, chunk.len(), num_frames);
                }

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

            pos = end;
            chunk_idx += 1;
            thread::sleep(Duration::from_secs_f32(interval));
        }

        // Dump features if requested
        if let Some(ref dump_path) = args.dump_features {
            let total_frames = all_features_bct.len() / n_mels;
            dump_features_to_file(&all_features_bct, dump_path)?;
            if args.verbose {
                eprintln!("[replay] Dumped {} frames to {:?}", total_frames, dump_path);
            }
        }
    } else {
        // Offline whole file
        let features_tc = extractor.compute(&audio);
        let num_frames = features_tc.len() / n_mels;
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

    if args.verbose {
        eprintln!("[replay] Completed in {:.2}s", start_time.elapsed().as_secs_f32());
    }

    Ok(())
}
