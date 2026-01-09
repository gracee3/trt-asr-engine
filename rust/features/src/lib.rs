use realfft::RealFftPlanner;
use realfft::num_complex::Complex;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelNorm {
    Slaney,
}

#[derive(Debug, Clone, Copy)]
pub struct FeatureConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub preemphasis: f32,
    pub f_min: f32,
    pub f_max: f32,
    pub mel_norm: MelNorm,
    pub log_zero_guard_value: f32,
    pub mag_power: f32,
    pub center: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 512,
            win_length: 400, // 25ms
            hop_length: 160, // 10ms
            n_mels: 128,
            // Match NeMo FilterbankFeatures defaults for this checkpoint.
            preemphasis: 0.97,
            f_min: 0.0,
            f_max: 0.0, // resolved to Nyquist in LogMelExtractor::new
            mel_norm: MelNorm::Slaney,
            log_zero_guard_value: 5.9604645e-8, // 2^-24
            mag_power: 2.0,
            center: true,
        }
    }
}

pub struct LogMelExtractor {
    config: FeatureConfig,
    mel_filterbank: Vec<Vec<f32>>,
    window: Vec<f32>,
    fft_plan: Arc<dyn realfft::RealToComplex<f32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationMode {
    None,
    PerFeature,
}

pub struct FeatureStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl LogMelExtractor {
    pub fn new(config: FeatureConfig) -> Self {
        let window = pad_window(&hann_window(config.win_length), config.n_fft);
        let f_max = if config.f_max > 0.0 {
            config.f_max
        } else {
            (config.sample_rate / 2) as f32
        };
        let mel_filterbank = create_mel_filterbank(
            config.n_mels,
            config.n_fft,
            config.sample_rate as f32,
            config.f_min,
            f_max,
            config.mel_norm,
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

        let valid_frames = self.valid_frames(audio.len());
        if valid_frames == 0 {
            return Vec::new();
        }

        let preemph = apply_preemphasis(audio, self.config.preemphasis);
        let padded = pad_centered(&preemph, self.config.center, self.config.n_fft, self.config.hop_length, valid_frames);

        let mut features = Vec::with_capacity(valid_frames * self.config.n_mels);
        let mut pos = 0usize;

        let mut fft_input = self.fft_plan.make_input_vec();
        let mut fft_output = self.fft_plan.make_output_vec();

        let n_fft = self.config.n_fft;

        for _frame_idx in 0..valid_frames {
            let frame = &padded[pos..pos + n_fft];

            let frame_out = self.compute_frame(frame, &mut fft_input, &mut fft_output);
            features.extend_from_slice(&frame_out);

            pos += self.config.hop_length;
        }

        features
    }

    pub fn n_mels(&self) -> usize {
        self.config.n_mels
    }

    fn valid_frames(&self, audio_len: usize) -> usize {
        if audio_len == 0 {
            return 0;
        }
        if self.config.center {
            audio_len / self.config.hop_length
        } else if audio_len < self.config.n_fft {
            0
        } else {
            (audio_len - self.config.n_fft) / self.config.hop_length + 1
        }
    }

    fn compute_frame(
        &self,
        frame: &[f32],
        fft_input: &mut [f32],
        fft_output: &mut [Complex<f32>],
    ) -> Vec<f32> {
        let n_fft = self.config.n_fft;
        let win = &self.window;
        for i in 0..n_fft {
            fft_input[i] = frame[i] * win[i];
        }
        self.fft_plan.process(fft_input, fft_output).unwrap();

        let mut power_spec = Vec::with_capacity(fft_output.len());
        for c in fft_output.iter() {
            let mag_sq = c.re * c.re + c.im * c.im;
            let val = if (self.config.mag_power - 2.0).abs() < f32::EPSILON {
                mag_sq
            } else if (self.config.mag_power - 1.0).abs() < f32::EPSILON {
                mag_sq.sqrt()
            } else {
                mag_sq.sqrt().powf(self.config.mag_power)
            };
            power_spec.push(val);
        }

        let mut out = Vec::with_capacity(self.config.n_mels);
        for mel_bin in &self.mel_filterbank {
            let mut energy = 0.0;
            for (i, &weight) in mel_bin.iter().enumerate() {
                energy += power_spec[i] * weight;
            }
            let log_energy = (energy + self.config.log_zero_guard_value).ln();
            out.push(log_energy);
        }
        out
    }
}

pub struct StreamingLogMelExtractor {
    extractor: LogMelExtractor,
    buffer: Vec<f32>,
    total_samples: usize,
    next_frame: usize,
    preemph_prev: f32,
    fft_input: Vec<f32>,
    fft_output: Vec<Complex<f32>>,
}

impl StreamingLogMelExtractor {
    pub fn new(config: FeatureConfig) -> Self {
        let extractor = LogMelExtractor::new(config);
        let left_pad = if extractor.config.center {
            extractor.config.n_fft / 2
        } else {
            0
        };
        let buffer = vec![0.0f32; left_pad];
        let fft_input = extractor.fft_plan.make_input_vec();
        let fft_output = extractor.fft_plan.make_output_vec();
        Self {
            extractor,
            buffer,
            total_samples: 0,
            next_frame: 0,
            preemph_prev: 0.0,
            fft_input,
            fft_output,
        }
    }

    pub fn n_mels(&self) -> usize {
        self.extractor.n_mels()
    }

    pub fn push(&mut self, audio: &[f32]) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }
        let preemph = self.apply_preemphasis_stream(audio);
        self.buffer.extend_from_slice(&preemph);
        self.total_samples += audio.len();

        let target_frames = self.extractor.valid_frames(self.total_samples);
        self.compute_available_frames(target_frames, false)
    }

    pub fn finalize(&mut self) -> Vec<f32> {
        let target_frames = self.extractor.valid_frames(self.total_samples);
        self.compute_available_frames(target_frames, true)
    }

    fn apply_preemphasis_stream(&mut self, audio: &[f32]) -> Vec<f32> {
        if self.extractor.config.preemphasis == 0.0 {
            return audio.to_vec();
        }
        let mut out = Vec::with_capacity(audio.len());
        for &sample in audio {
            out.push(sample - self.extractor.config.preemphasis * self.preemph_prev);
            self.preemph_prev = sample;
        }
        out
    }

    fn compute_available_frames(&mut self, target_frames: usize, pad_right: bool) -> Vec<f32> {
        let n_fft = self.extractor.config.n_fft;
        let hop = self.extractor.config.hop_length;
        if pad_right && target_frames > 0 {
            let last_start = (target_frames - 1) * hop;
            let required_len = last_start + n_fft;
            if self.buffer.len() < required_len {
                self.buffer.resize(required_len, 0.0);
            }
        }

        let mut out = Vec::new();
        while self.next_frame < target_frames {
            let start = self.next_frame * hop;
            let end = start + n_fft;
            if end > self.buffer.len() {
                break;
            }
            let frame = &self.buffer[start..end];
            let frame_out = self
                .extractor
                .compute_frame(frame, &mut self.fft_input, &mut self.fft_output);
            out.extend_from_slice(&frame_out);
            self.next_frame += 1;
        }
        out
    }
}

pub fn compute_per_feature_stats(features_tc: &[f32], n_mels: usize, frames: usize) -> FeatureStats {
    let mut mean = vec![0.0f32; n_mels];
    let mut std = vec![0.0f32; n_mels];

    if frames == 0 || n_mels == 0 {
        return FeatureStats { mean, std };
    }

    for t in 0..frames {
        let base = t * n_mels;
        for m in 0..n_mels {
            mean[m] += features_tc[base + m];
        }
    }
    let denom = frames as f32;
    for m in 0..n_mels {
        mean[m] /= denom;
    }

    let denom_std = if frames > 1 { (frames - 1) as f32 } else { 1.0 };
    for t in 0..frames {
        let base = t * n_mels;
        for m in 0..n_mels {
            let diff = features_tc[base + m] - mean[m];
            std[m] += diff * diff;
        }
    }
    for m in 0..n_mels {
        std[m] = (std[m] / denom_std).sqrt() + 1e-5;
    }

    FeatureStats { mean, std }
}

pub fn apply_per_feature_norm(features_tc: &mut [f32], n_mels: usize, frames: usize, stats: &FeatureStats) {
    if frames == 0 || n_mels == 0 {
        return;
    }
    for t in 0..frames {
        let base = t * n_mels;
        for m in 0..n_mels {
            let idx = base + m;
            features_tc[idx] = (features_tc[idx] - stats.mean[m]) / stats.std[m];
        }
    }
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos()))
        .collect()
}

fn pad_window(window: &[f32], n_fft: usize) -> Vec<f32> {
    if window.len() == n_fft {
        return window.to_vec();
    }
    let mut padded = vec![0.0f32; n_fft];
    let left = (n_fft - window.len()) / 2;
    padded[left..left + window.len()].copy_from_slice(window);
    padded
}

fn apply_preemphasis(audio: &[f32], preemph: f32) -> Vec<f32> {
    if preemph == 0.0 {
        return audio.to_vec();
    }
    let mut out = Vec::with_capacity(audio.len());
    let mut prev = 0.0f32;
    for &sample in audio {
        out.push(sample - preemph * prev);
        prev = sample;
    }
    out
}

fn pad_centered(
    audio: &[f32],
    center: bool,
    n_fft: usize,
    hop_length: usize,
    valid_frames: usize,
) -> Vec<f32> {
    let left_pad = if center { n_fft / 2 } else { 0 };
    let mut padded = vec![0.0f32; left_pad];
    padded.extend_from_slice(audio);
    if valid_frames > 0 {
        let last_start = (valid_frames - 1) * hop_length;
        let required_len = last_start + n_fft;
        if padded.len() < required_len {
            padded.resize(required_len, 0.0);
        }
    }
    padded
}

fn hz_to_mel_slaney(hz: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if hz < min_log_hz {
        hz / f_sp
    } else {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz_slaney(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if mel < min_log_mel {
        mel * f_sp
    } else {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    }
}

fn create_mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
    mel_norm: MelNorm,
) -> Vec<Vec<f32>> {
    let min_mel = hz_to_mel_slaney(f_min);
    let max_mel = hz_to_mel_slaney(f_max);

    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..(n_mels + 2) {
        let mel = min_mel + (max_mel - min_mel) * (i as f32 / (n_mels + 1) as f32);
        mel_points.push(mel_to_hz_slaney(mel));
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

    if mel_norm == MelNorm::Slaney {
        for m in 0..n_mels {
            let enorm = 2.0 / (mel_points[m + 2] - mel_points[m]);
            for i in 0..bin_count {
                filterbank[m][i] *= enorm;
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
        
        // Expected frames: floor(16000 / 160) = 100 (centered STFT, valid frames only)
        let expected_frames = 16000 / 160;
        assert_eq!(features.len(), expected_frames * 128);
    }
}
