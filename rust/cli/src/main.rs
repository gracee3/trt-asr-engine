use clap::Parser;
use features::{FeatureConfig, LogMelExtractor};
use parakeet_trt::{ParakeetSessionSafe, TranscriptionEvent};
use std::path::PathBuf;
use std::time::Duration;
use std::thread;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(help = "Path to input WAV file")]
    wav_path: PathBuf,

    #[arg(long, help = "Path to model directory")]
    model_dir: PathBuf,

    #[arg(long, help = "Simulated streaming interval in seconds")]
    stream_sim: Option<f32>,

    #[arg(long, default_value_t = 0)]
    device_id: i32,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // 1. Load Audio
    let mut reader = hound::WavReader::open(&args.wav_path)?;
    let spec = reader.spec();
    if spec.sample_rate != 16000 || spec.channels != 1 {
        anyhow::bail!("WAV must be 16kHz mono (got {} Hz, {} channels)", spec.sample_rate, spec.channels);
    }
    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            // Normalize to [-1, 1] for common PCM bit depths.
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
    };

    // 2. Setup Feature Extractor
    let config = FeatureConfig::default();
    let extractor = LogMelExtractor::new(config);
    let n_mels = extractor.n_mels();

    let to_bct = |feat_tc: &[f32], frames: usize| -> Vec<f32> {
        // Input is [T, C] (frame-major), output is [C, T] (mel-major) to match encoder [B,C,T].
        let mut out = vec![0.0f32; n_mels * frames];
        for t in 0..frames {
            for m in 0..n_mels {
                out[m * frames + t] = feat_tc[t * n_mels + m];
            }
        }
        out
    };

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

        while pos < audio.len() {
            let end = (pos + samples_per_chunk).min(audio.len());
            let chunk = &audio[pos..end];
            
            let features_tc = extractor.compute(chunk);
            let num_frames = features_tc.len() / n_mels;
            
            if num_frames > 0 {
                let features_bct = to_bct(&features_tc, num_frames);
                session.push_features(&features_bct, num_frames)?;
                
                // Poll for events
                while let Some(event) = session.poll_event() {
                    match event {
                        TranscriptionEvent::PartialText { text, .. } => {
                            print!("\rPartial: {}", text);
                            use std::io::{self, Write};
                            io::stdout().flush().unwrap();
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
            thread::sleep(Duration::from_secs_f32(interval));
        }
    } else {
        // Offline whole file
        let features_tc = extractor.compute(&audio);
        let num_frames = features_tc.len() / n_mels;
        let features_bct = to_bct(&features_tc, num_frames);
        session.push_features(&features_bct, num_frames)?;
        
        while let Some(event) = session.poll_event() {
            match event {
                TranscriptionEvent::FinalText { text, .. } => println!("Transcript: {}", text),
                TranscriptionEvent::Error { message } => eprintln!("Error: {}", message),
                _ => {}
            }
        }
    }

    Ok(())
}
