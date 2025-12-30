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
    if spec.sample_rate != 16000 || spec.channels != 1 || spec.sample_format != hound::SampleFormat::Float {
        anyhow::bail!("WAV must be 16kHz mono f32");
    }
    let audio: Vec<f32> = reader.samples::<f32>().map(|s| s.unwrap()).collect();

    // 2. Setup Feature Extractor
    let config = FeatureConfig::default();
    let extractor = LogMelExtractor::new(config);

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
            
            let features = extractor.compute(chunk);
            let num_frames = features.len() / 80;
            
            if num_frames > 0 {
                session.push_features(&features, num_frames)?;
                
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
        let features = extractor.compute(&audio);
        let num_frames = features.len() / 80;
        session.push_features(&features, num_frames)?;
        
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
