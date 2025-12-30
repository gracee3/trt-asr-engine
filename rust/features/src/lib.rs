use realfft::RealFftPlanner;
use std::sync::Arc;

pub struct FeatureConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub preemphasis: f32,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 512,
            win_length: 400, // 25ms
            hop_length: 160, // 10ms
            n_mels: 128,
            preemphasis: 0.97,
        }
    }
}

pub struct LogMelExtractor {
    config: FeatureConfig,
    mel_filterbank: Vec<Vec<f32>>,
    window: Vec<f32>,
    fft_plan: Arc<dyn realfft::RealToComplex<f32>>,
}

impl LogMelExtractor {
    pub fn new(config: FeatureConfig) -> Self {
        let window = hann_window(config.win_length);
        let mel_filterbank = create_mel_filterbank(
            config.n_mels,
            config.n_fft,
            config.sample_rate as f32,
            0.0,
            (config.sample_rate / 2) as f32,
        );
        let mut fft_planner = RealFftPlanner::new();
        let fft_plan = fft_planner.plan_fft_forward(config.n_fft);

        Self {
            config,
            mel_filterbank,
            window,
            fft_plan,
        }
    }

    pub fn compute(&self, audio: &[f32]) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }

        let mut features = Vec::new();
        let mut pos = 0;

        let mut fft_input = self.fft_plan.make_input_vec();
        let mut fft_output = self.fft_plan.make_output_vec();

        while pos + self.config.win_length <= audio.len() {
            let frame = &audio[pos..pos + self.config.win_length];
            
            // Pre-emphasis and Windowing
            for i in 0..self.config.win_length {
                let val = if i == 0 {
                    frame[i]
                } else {
                    frame[i] - self.config.preemphasis * frame[i-1]
                };
                fft_input[i] = val * self.window[i];
            }
            // Zero padding if win_length < n_fft
            for i in self.config.win_length..self.config.n_fft {
                fft_input[i] = 0.0;
            }

            self.fft_plan.process(&mut fft_input, &mut fft_output).unwrap();

            // Power spectrum
            let mut power_spec = Vec::with_capacity(fft_output.len());
            for c in fft_output.iter() {
                let mag_sq = c.re * c.re + c.im * c.im;
                power_spec.push(mag_sq);
            }

            // Mel filtering
            for mel_bin in &self.mel_filterbank {
                let mut energy = 0.0;
                for (i, &weight) in mel_bin.iter().enumerate() {
                    energy += power_spec[i] * weight;
                }
                // Log compression (with small epsilon to avoid log(0))
                let log_energy = (energy + 1e-6).ln();
                features.push(log_energy);
            }

            pos += self.config.hop_length;
        }

        features
    }

    pub fn n_mels(&self) -> usize {
        self.config.n_mels
    }
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos()))
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

fn create_mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
) -> Vec<Vec<f32>> {
    let min_mel = hz_to_mel(f_min);
    let max_mel = hz_to_mel(f_max);

    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..(n_mels + 2) {
        let mel = min_mel + (max_mel - min_mel) * (i as f32 / (n_mels + 1) as f32);
        mel_points.push(mel_to_hz(mel));
    }

    let bin_count = n_fft / 2 + 1;
    let mut filterbank = vec![vec![0.0; bin_count]; n_mels];

    for m in 0..n_mels {
        let left = mel_points[m];
        let center = mel_points[m + 1];
        let right = mel_points[m + 2];

        for i in 0..bin_count {
            let freq = i as f32 * sample_rate / n_fft as f32;
            if freq > left && freq < center {
                filterbank[m][i] = (freq - left) / (center - left);
            } else if freq >= center && freq < right {
                filterbank[m][i] = (right - freq) / (right - center);
            }
        }
    }

    filterbank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_stability() {
        let config = FeatureConfig::default();
        let extractor = LogMelExtractor::new(config);
        
        // 1 second of audio @ 16k
        let audio = vec![0.0; 16000];
        let features = extractor.compute(&audio);
        
        // Expected frames: (16000 - 400) / 160 + 1 = 15600 / 160 + 1 = 97 + 1 = 98
        let expected_frames = (16000 - 400) / 160 + 1;
        assert_eq!(features.len(), expected_frames * 128);
    }
}
