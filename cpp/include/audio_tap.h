// audio_tap.h - RAII audio tap writer for pipeline debugging
//
// Writes raw PCM + JSON sidecar for offline analysis.
// Env-var gated: set AUDIO_TAP_ENABLE=1 to activate.
//
// IMPORTANT: All counts are in FRAMES (not samples).
// A frame contains `channels` samples. For stereo: 1 frame = 2 samples.
//
// Usage:
//   AudioTapWriter tap("post_dsp", 16000, 1, AudioTapWriter::Format::F32LE);
//   tap.write_frames(samples, num_frames);  // num_frames, not num_samples!
//   // On destruction, JSON sidecar is finalized
//
// Convert to WAV offline:
//   ffmpeg -f f32le -ar 16000 -ac 1 -i tap_post_dsp.raw tap_post_dsp.wav

#ifndef AUDIO_TAP_H
#define AUDIO_TAP_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <string>
#include <atomic>
#include <mutex>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace audio_tap {

// ============================================================================
// Global configuration (resolved once at first use)
// ============================================================================

struct TapConfig {
    bool enabled = false;
    bool fill_gaps = false;
    bool per_chunk_ndjson = false;
    std::string base_dir;
    std::string run_dir;  // base_dir/run_<timestamp>_<pid>/

    static TapConfig& instance() {
        static TapConfig cfg = init();
        return cfg;
    }

private:
    static bool env_bool(const char* name, bool def = false) {
        const char* v = std::getenv(name);
        if (!v) return def;
        return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0;
    }

    static TapConfig init() {
        TapConfig cfg;
        cfg.enabled = env_bool("AUDIO_TAP_ENABLE");
        cfg.fill_gaps = env_bool("AUDIO_TAP_FILL_GAPS", true);  // Default: fill gaps
        cfg.per_chunk_ndjson = env_bool("AUDIO_TAP_PER_CHUNK");

        const char* dir_env = std::getenv("AUDIO_TAP_DIR");
        cfg.base_dir = dir_env ? dir_env : ".";

        // Create run-isolated directory: <base>/run_<timestamp>_<pid>/
        if (cfg.enabled) {
            char run_suffix[64];
            time_t now = time(nullptr);
            struct tm tm_buf;
            localtime_r(&now, &tm_buf);
            snprintf(run_suffix, sizeof(run_suffix), "run_%04d%02d%02d_%02d%02d%02d_%d",
                     tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                     tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
                     static_cast<int>(getpid()));
            cfg.run_dir = cfg.base_dir + "/" + run_suffix;
            mkdir(cfg.run_dir.c_str(), 0755);
        }

        return cfg;
    }
};

// Sanitize tap name to env var: uppercase, replace non-alnum with _, collapse
inline std::string sanitize_tap_name(const std::string& name) {
    std::string result;
    result.reserve(name.size());
    bool last_was_underscore = false;
    for (char c : name) {
        if ((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
            result += c;
            last_was_underscore = false;
        } else if (c >= 'a' && c <= 'z') {
            result += (c - 'a' + 'A');  // uppercase
            last_was_underscore = false;
        } else {
            if (!last_was_underscore && !result.empty()) {
                result += '_';
                last_was_underscore = true;
            }
        }
    }
    // Trim trailing underscore
    while (!result.empty() && result.back() == '_') result.pop_back();
    return result;
}

// Check if specific tap is enabled: AUDIO_TAP_<SANITIZED_NAME>=1
inline bool is_tap_enabled(const std::string& name) {
    if (!TapConfig::instance().enabled) return false;
    std::string env_name = "AUDIO_TAP_" + sanitize_tap_name(name);
    const char* v = std::getenv(env_name.c_str());
    return v && (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0);
}

// ============================================================================
// Sample format
// ============================================================================

enum class Format {
    S16LE,   // Signed 16-bit little-endian (int16_t)
    S32LE,   // Signed 32-bit little-endian (int32_t)
    F32LE,   // Float 32-bit little-endian (float)
    F64LE,   // Float 64-bit little-endian (double)
};

inline const char* format_to_string(Format fmt) {
    switch (fmt) {
        case Format::S16LE: return "s16le";
        case Format::S32LE: return "s32le";
        case Format::F32LE: return "f32le";
        case Format::F64LE: return "f64le";
    }
    return "unknown";
}

inline size_t format_bytes_per_sample(Format fmt) {
    switch (fmt) {
        case Format::S16LE: return 2;
        case Format::S32LE: return 4;
        case Format::F32LE: return 4;
        case Format::F64LE: return 8;
    }
    return 0;
}

inline double format_fullscale(Format fmt) {
    switch (fmt) {
        case Format::S16LE: return 32768.0;
        case Format::S32LE: return 2147483648.0;
        case Format::F32LE: return 1.0;
        case Format::F64LE: return 1.0;
    }
    return 1.0;
}

// ============================================================================
// Per-tap statistics (accumulated during writes)
// ============================================================================

struct TapStats {
    double peak = 0.0;           // Max absolute value (in native scale)
    double sum = 0.0;            // Running sum for mean
    double sum_sq = 0.0;         // Running sum of squares for RMS
    uint64_t total_frames = 0;   // Frames written (excluding gaps)
    uint64_t total_samples = 0;  // Total scalar samples = frames * channels
    uint64_t clipped_samples = 0;
    uint64_t nan_count = 0;
    uint64_t inf_count = 0;
    uint64_t gap_frames = 0;     // Gap frames (zeros written if fill_gaps)
    uint64_t write_count = 0;    // Number of write calls

    void reset() {
        peak = sum = sum_sq = 0.0;
        total_frames = total_samples = clipped_samples = 0;
        nan_count = inf_count = gap_frames = write_count = 0;
    }

    double mean() const {
        return total_samples > 0 ? sum / total_samples : 0.0;
    }

    double rms() const {
        return total_samples > 0 ? std::sqrt(sum_sq / total_samples) : 0.0;
    }

    double dc_offset() const { return mean(); }

    // RMS in dBFS (relative to fullscale)
    double rms_dbfs(double fullscale) const {
        double r = rms();
        if (r <= 0.0 || fullscale <= 0.0) return -120.0;  // Floor
        return 20.0 * std::log10(r / fullscale);
    }

    double peak_dbfs(double fullscale) const {
        if (peak <= 0.0 || fullscale <= 0.0) return -120.0;
        return 20.0 * std::log10(peak / fullscale);
    }
};

// ============================================================================
// RAII Audio Tap Writer
// ============================================================================

class AudioTapWriter {
public:
    // Create a tap writer.
    // stage_name: identifies this tap point (e.g., "capture", "post_dsp")
    // sample_rate: Hz
    // channels: number of interleaved channels
    // fmt: sample format
    // notes: optional description
    AudioTapWriter(const std::string& stage_name,
                   uint32_t sample_rate,
                   uint16_t channels,
                   Format fmt,
                   const std::string& notes = "")
        : stage_name_(stage_name)
        , sanitized_name_(sanitize_tap_name(stage_name))
        , sample_rate_(sample_rate)
        , channels_(channels)
        , format_(fmt)
        , fullscale_(format_fullscale(fmt))
        , notes_(notes)
        , enabled_(false)
        , raw_file_(nullptr)
        , ndjson_file_(nullptr)
    {
        // Early return if globally disabled (zero overhead path)
        if (!TapConfig::instance().enabled) return;
        if (!is_tap_enabled(stage_name)) return;

        enabled_ = true;
        // Cache run_dir now to avoid static destruction order issues
        run_dir_ = TapConfig::instance().run_dir;

        // Record start time
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        start_monotonic_ns_ = ts.tv_sec * 1000000000ULL + ts.tv_nsec;

        // Open files in run-isolated directory
        std::string base_path = run_dir_ + "/tap_" + sanitized_name_;

        raw_file_ = std::fopen((base_path + ".raw").c_str(), "wb");
        if (!raw_file_) {
            fprintf(stderr, "[AudioTap] ERROR: Failed to open %s.raw\n", base_path.c_str());
            enabled_ = false;
            return;
        }

        if (TapConfig::instance().per_chunk_ndjson) {
            ndjson_file_ = std::fopen((base_path + ".ndjson").c_str(), "w");
        }

        fprintf(stderr, "[AudioTap] Opened tap '%s' -> %s.raw (%u Hz, %u ch, %s)\n",
                stage_name_.c_str(), base_path.c_str(),
                sample_rate_, channels_, format_to_string(fmt));
    }

    ~AudioTapWriter() {
        if (!enabled_) return;

        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t end_ns = ts.tv_sec * 1000000000ULL + ts.tv_nsec;

        if (raw_file_) std::fclose(raw_file_);
        if (ndjson_file_) std::fclose(ndjson_file_);

        // Write JSON sidecar (use cached run_dir_ to avoid static destruction order issues)
        std::string json_path = run_dir_ + "/tap_" + sanitized_name_ + ".json";
        FILE* json_file = std::fopen(json_path.c_str(), "w");
        if (json_file) {
            uint64_t total_frames_with_gaps = stats_.total_frames + stats_.gap_frames;
            double duration_sec = static_cast<double>(stats_.total_frames) / sample_rate_;
            double duration_with_gaps_sec = static_cast<double>(total_frames_with_gaps) / sample_rate_;

            fprintf(json_file,
                "{\n"
                "  \"tap_name\": \"%s\",\n"
                "  \"sanitized_name\": \"%s\",\n"
                "  \"sample_rate_hz\": %u,\n"
                "  \"channels\": %u,\n"
                "  \"format\": \"%s\",\n"
                "  \"bytes_per_sample\": %zu,\n"
                "  \"fullscale\": %.1f,\n"
                "  \"layout\": \"interleaved_frames\",\n"
                "  \"start_monotonic_ns\": %llu,\n"
                "  \"end_monotonic_ns\": %llu,\n"
                "  \"frames_written\": %llu,\n"
                "  \"gap_frames\": %llu,\n"
                "  \"total_frames_including_gaps\": %llu,\n"
                "  \"total_samples\": %llu,\n"
                "  \"duration_sec\": %.6f,\n"
                "  \"duration_with_gaps_sec\": %.6f,\n"
                "  \"write_count\": %llu,\n"
                "  \"gaps_filled\": %s,\n"
                "  \"stats\": {\n"
                "    \"peak\": %.9g,\n"
                "    \"peak_dbfs\": %.2f,\n"
                "    \"rms\": %.9g,\n"
                "    \"rms_dbfs\": %.2f,\n"
                "    \"dc_offset\": %.9g,\n"
                "    \"clipped_samples\": %llu,\n"
                "    \"nan_count\": %llu,\n"
                "    \"inf_count\": %llu\n"
                "  },\n"
                "  \"notes\": \"%s\"\n"
                "}\n",
                stage_name_.c_str(),
                sanitized_name_.c_str(),
                sample_rate_,
                channels_,
                format_to_string(format_),
                format_bytes_per_sample(format_),
                fullscale_,
                (unsigned long long)start_monotonic_ns_,
                (unsigned long long)end_ns,
                (unsigned long long)stats_.total_frames,
                (unsigned long long)stats_.gap_frames,
                (unsigned long long)total_frames_with_gaps,
                (unsigned long long)stats_.total_samples,
                duration_sec,
                duration_with_gaps_sec,
                (unsigned long long)stats_.write_count,
                TapConfig::instance().fill_gaps ? "true" : "false",
                stats_.peak,
                stats_.peak_dbfs(fullscale_),
                stats_.rms(),
                stats_.rms_dbfs(fullscale_),
                stats_.dc_offset(),
                (unsigned long long)stats_.clipped_samples,
                (unsigned long long)stats_.nan_count,
                (unsigned long long)stats_.inf_count,
                notes_.c_str()
            );
            std::fclose(json_file);
        }

        fprintf(stderr, "[AudioTap] Closed '%s': %llu frames (%.3fs), peak=%.4f (%.1f dBFS), rms=%.6f (%.1f dBFS)\n",
                stage_name_.c_str(),
                (unsigned long long)stats_.total_frames,
                static_cast<double>(stats_.total_frames) / sample_rate_,
                stats_.peak, stats_.peak_dbfs(fullscale_),
                stats_.rms(), stats_.rms_dbfs(fullscale_));
    }

    // Non-copyable, movable
    AudioTapWriter(const AudioTapWriter&) = delete;
    AudioTapWriter& operator=(const AudioTapWriter&) = delete;
    AudioTapWriter(AudioTapWriter&& other) noexcept { *this = std::move(other); }
    AudioTapWriter& operator=(AudioTapWriter&& other) noexcept {
        if (this != &other) {
            stage_name_ = std::move(other.stage_name_);
            sanitized_name_ = std::move(other.sanitized_name_);
            sample_rate_ = other.sample_rate_;
            channels_ = other.channels_;
            format_ = other.format_;
            fullscale_ = other.fullscale_;
            notes_ = std::move(other.notes_);
            enabled_ = other.enabled_;
            start_monotonic_ns_ = other.start_monotonic_ns_;
            raw_file_ = other.raw_file_;
            ndjson_file_ = other.ndjson_file_;
            stats_ = other.stats_;
            other.raw_file_ = nullptr;
            other.ndjson_file_ = nullptr;
            other.enabled_ = false;
        }
        return *this;
    }

    bool is_enabled() const { return enabled_; }
    const TapStats& stats() const { return stats_; }
    uint32_t sample_rate() const { return sample_rate_; }
    uint16_t channels() const { return channels_; }

    // ========================================================================
    // Write methods - all use FRAMES (not samples)
    // A frame contains `channels` interleaved samples.
    // ========================================================================

    // Write interleaved frames (generic template)
    template<typename T>
    void write_frames(const T* interleaved_data, size_t num_frames) {
        // Zero-overhead early return when disabled
        if (!enabled_) return;
        if (!raw_file_ || num_frames == 0) return;

        std::lock_guard<std::mutex> lock(mutex_);

        const size_t num_samples = num_frames * channels_;

        // Update stats
        for (size_t i = 0; i < num_samples; ++i) {
            double val = static_cast<double>(interleaved_data[i]);

            if (std::isnan(val)) {
                stats_.nan_count++;
                continue;
            }
            if (std::isinf(val)) {
                stats_.inf_count++;
                continue;
            }

            double abs_val = std::fabs(val);
            if (abs_val > stats_.peak) stats_.peak = abs_val;
            stats_.sum += val;
            stats_.sum_sq += val * val;

            // Clipping detection for integer formats
            if constexpr (std::is_same_v<T, int16_t>) {
                if (interleaved_data[i] == INT16_MAX || interleaved_data[i] == INT16_MIN) {
                    stats_.clipped_samples++;
                }
            } else if constexpr (std::is_same_v<T, int32_t>) {
                if (interleaved_data[i] == INT32_MAX || interleaved_data[i] == INT32_MIN) {
                    stats_.clipped_samples++;
                }
            }
        }

        stats_.total_frames += num_frames;
        stats_.total_samples += num_samples;
        stats_.write_count++;

        // Write raw bytes
        size_t bytes = num_samples * format_bytes_per_sample(format_);
        std::fwrite(interleaved_data, 1, bytes, raw_file_);

        // Optional per-chunk NDJSON
        if (ndjson_file_) {
            write_chunk_ndjson(num_frames);
        }
    }

    // Convenience wrappers with type checking
    void write_frames_f32(const float* data, size_t num_frames) {
        if (format_ != Format::F32LE && enabled_) {
            fprintf(stderr, "[AudioTap] WARNING: write_frames_f32 but format is %s\n",
                    format_to_string(format_));
        }
        write_frames(data, num_frames);
    }

    void write_frames_s16(const int16_t* data, size_t num_frames) {
        if (format_ != Format::S16LE && enabled_) {
            fprintf(stderr, "[AudioTap] WARNING: write_frames_s16 but format is %s\n",
                    format_to_string(format_));
        }
        write_frames(data, num_frames);
    }

    // ========================================================================
    // Gap recording - also uses FRAMES
    // ========================================================================

    // Record a gap (dropped frames).
    // If AUDIO_TAP_FILL_GAPS=1 (default), writes zeros to preserve timeline.
    void record_gap(size_t num_frames) {
        if (!enabled_) return;
        if (!raw_file_ || num_frames == 0) return;

        std::lock_guard<std::mutex> lock(mutex_);

        stats_.gap_frames += num_frames;

        if (TapConfig::instance().fill_gaps) {
            // Write zeros to preserve time alignment
            size_t num_samples = num_frames * channels_;
            size_t bytes = num_samples * format_bytes_per_sample(format_);
            std::vector<uint8_t> zeros(bytes, 0);
            std::fwrite(zeros.data(), 1, bytes, raw_file_);
            // Note: zeros don't contribute to sum/sum_sq but do affect timeline
        }

        // Log gap in NDJSON if enabled
        if (ndjson_file_) {
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            uint64_t now_ns = ts.tv_sec * 1000000000ULL + ts.tv_nsec;
            fprintf(ndjson_file_,
                "{\"type\":\"gap\",\"ts_ns\":%llu,\"gap_frames\":%zu,\"filled\":%s}\n",
                (unsigned long long)now_ns,
                num_frames,
                TapConfig::instance().fill_gaps ? "true" : "false");
        }
    }

    // Flush to disk
    void flush() {
        if (enabled_ && raw_file_) {
            std::fflush(raw_file_);
        }
        if (ndjson_file_) {
            std::fflush(ndjson_file_);
        }
    }

private:
    void write_chunk_ndjson(size_t num_frames) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t now_ns = ts.tv_sec * 1000000000ULL + ts.tv_nsec;

        // Compute chunk RMS (approximate from recent stats delta)
        double chunk_rms = stats_.total_samples > 0 ? stats_.rms() : 0.0;

        fprintf(ndjson_file_,
            "{\"type\":\"chunk\",\"ts_ns\":%llu,\"idx\":%llu,\"frames\":%zu,"
            "\"total_frames\":%llu,\"rms\":%.6g,\"peak\":%.6g}\n",
            (unsigned long long)now_ns,
            (unsigned long long)stats_.write_count,
            num_frames,
            (unsigned long long)stats_.total_frames,
            chunk_rms,
            stats_.peak);
    }

    std::string stage_name_;
    std::string sanitized_name_;
    std::string run_dir_;  // Cached at construction to avoid static destruction order issues
    uint32_t sample_rate_;
    uint16_t channels_;
    Format format_;
    double fullscale_;
    std::string notes_;
    bool enabled_;
    uint64_t start_monotonic_ns_ = 0;
    FILE* raw_file_;
    FILE* ndjson_file_;
    TapStats stats_;
    std::mutex mutex_;
};

// ============================================================================
// Feature Tap Writer (specialized for mel features)
// ============================================================================

class FeatureTapWriter {
public:
    // Create a feature tap.
    // mel_bins: number of mel frequency bins (typically 128)
    // frame_shift_ms: hop size in milliseconds (typically 10.0)
    // window_ms: window size in milliseconds (typically 25.0)
    // audio_sample_rate: sample rate of input audio (typically 16000)
    // layout: "bins_major" ([C,T]) or "frames_major" ([T,C])
    FeatureTapWriter(const std::string& name,
                     size_t mel_bins,
                     float frame_shift_ms,
                     float window_ms,
                     uint32_t audio_sample_rate,
                     const std::string& layout = "bins_major",
                     const std::string& notes = "")
        : name_(name)
        , sanitized_name_(sanitize_tap_name(name))
        , mel_bins_(mel_bins)
        , frame_shift_ms_(frame_shift_ms)
        , window_ms_(window_ms)
        , audio_sample_rate_(audio_sample_rate)
        , layout_(layout)
        , notes_(notes)
        , enabled_(false)
        , raw_file_(nullptr)
    {
        if (!TapConfig::instance().enabled) return;
        if (!is_tap_enabled(name)) return;

        enabled_ = true;
        // Cache run_dir now to avoid static destruction order issues
        run_dir_ = TapConfig::instance().run_dir;

        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        start_monotonic_ns_ = ts.tv_sec * 1000000000ULL + ts.tv_nsec;

        std::string base_path = run_dir_ + "/tap_" + sanitized_name_;

        raw_file_ = std::fopen((base_path + ".raw").c_str(), "wb");
        if (!raw_file_) {
            fprintf(stderr, "[FeatureTap] ERROR: Failed to open %s.raw\n", base_path.c_str());
            enabled_ = false;
            return;
        }

        fprintf(stderr, "[FeatureTap] Opened '%s' -> %s.raw (%zu bins, %.1fms shift, %s)\n",
                name_.c_str(), base_path.c_str(), mel_bins_, frame_shift_ms_, layout_.c_str());
    }

    ~FeatureTapWriter() {
        if (!enabled_) return;

        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t end_ns = ts.tv_sec * 1000000000ULL + ts.tv_nsec;

        if (raw_file_) std::fclose(raw_file_);

        // Use cached run_dir_ to avoid static destruction order issues
        std::string json_path = run_dir_ + "/tap_" + sanitized_name_ + ".json";
        FILE* json_file = std::fopen(json_path.c_str(), "w");
        if (json_file) {
            double frame_rate_hz = 1000.0 / frame_shift_ms_;
            double duration_sec = total_frames_ / frame_rate_hz;
            double mean_val = total_count_ > 0 ? stats_sum_ / total_count_ : 0.0;
            double var_val = total_count_ > 0 ? (stats_sum_sq_ / total_count_) - (mean_val * mean_val) : 0.0;

            // Build shape string (must persist through fprintf)
            std::string shape_str = (layout_ == "bins_major")
                ? (std::to_string(mel_bins_) + ", " + std::to_string(total_frames_))
                : (std::to_string(total_frames_) + ", " + std::to_string(mel_bins_));

            fprintf(json_file,
                "{\n"
                "  \"kind\": \"mel_features\",\n"
                "  \"tap_name\": \"%s\",\n"
                "  \"mel_bins\": %zu,\n"
                "  \"num_frames\": %llu,\n"
                "  \"shape\": [%s],\n"
                "  \"layout\": \"%s\",\n"
                "  \"frame_shift_ms\": %.2f,\n"
                "  \"window_ms\": %.2f,\n"
                "  \"frame_rate_hz\": %.2f,\n"
                "  \"audio_sample_rate_hz\": %u,\n"
                "  \"format\": \"f32le\",\n"
                "  \"start_monotonic_ns\": %llu,\n"
                "  \"end_monotonic_ns\": %llu,\n"
                "  \"duration_sec\": %.6f,\n"
                "  \"stats\": {\n"
                "    \"min\": %.9g,\n"
                "    \"max\": %.9g,\n"
                "    \"mean\": %.9g,\n"
                "    \"variance\": %.9g,\n"
                "    \"nan_count\": %llu,\n"
                "    \"inf_count\": %llu,\n"
                "    \"zero_frames\": %llu\n"
                "  },\n"
                "  \"notes\": \"%s\"\n"
                "}\n",
                name_.c_str(),
                mel_bins_,
                (unsigned long long)total_frames_,
                shape_str.c_str(),
                layout_.c_str(),
                frame_shift_ms_,
                window_ms_,
                frame_rate_hz,
                audio_sample_rate_,
                (unsigned long long)start_monotonic_ns_,
                (unsigned long long)end_ns,
                duration_sec,
                stats_min_,
                stats_max_,
                mean_val,
                var_val,
                (unsigned long long)nan_count_,
                (unsigned long long)inf_count_,
                (unsigned long long)zero_frames_,
                notes_.c_str());
            std::fclose(json_file);
        }

        fprintf(stderr, "[FeatureTap] Closed '%s': %llu frames, min=%.4f max=%.4f\n",
                name_.c_str(), (unsigned long long)total_frames_, stats_min_, stats_max_);
    }

    bool is_enabled() const { return enabled_; }

    // Write features. Expected layout depends on constructor parameter.
    // data: pointer to float features
    // num_frames: number of time frames
    void write_frames(const float* data, size_t num_frames) {
        if (!enabled_) return;
        if (!raw_file_ || num_frames == 0) return;

        std::lock_guard<std::mutex> lock(mutex_);

        size_t total_floats = mel_bins_ * num_frames;

        // Compute stats
        for (size_t i = 0; i < total_floats; ++i) {
            float v = data[i];
            if (std::isnan(v)) { nan_count_++; continue; }
            if (std::isinf(v)) { inf_count_++; continue; }
            if (v < stats_min_) stats_min_ = v;
            if (v > stats_max_) stats_max_ = v;
            stats_sum_ += v;
            stats_sum_sq_ += v * v;
            total_count_++;
        }

        // Count near-zero frames (all bins < threshold)
        const float zero_thresh = 1e-6f;
        for (size_t f = 0; f < num_frames; ++f) {
            bool is_zero = true;
            for (size_t b = 0; b < mel_bins_ && is_zero; ++b) {
                size_t idx = (layout_ == "bins_major") ? (b * num_frames + f) : (f * mel_bins_ + b);
                if (std::fabs(data[idx]) > zero_thresh) is_zero = false;
            }
            if (is_zero) zero_frames_++;
        }

        total_frames_ += num_frames;

        std::fwrite(data, sizeof(float), total_floats, raw_file_);
    }

private:
    std::string name_;
    std::string sanitized_name_;
    std::string run_dir_;  // Cached at construction to avoid static destruction order issues
    size_t mel_bins_;
    float frame_shift_ms_;
    float window_ms_;
    uint32_t audio_sample_rate_;
    std::string layout_;
    std::string notes_;
    bool enabled_;
    uint64_t start_monotonic_ns_ = 0;
    FILE* raw_file_;
    std::mutex mutex_;

    // Stats
    uint64_t total_frames_ = 0;
    uint64_t total_count_ = 0;
    double stats_min_ = std::numeric_limits<double>::infinity();
    double stats_max_ = -std::numeric_limits<double>::infinity();
    double stats_sum_ = 0.0;
    double stats_sum_sq_ = 0.0;
    uint64_t nan_count_ = 0;
    uint64_t inf_count_ = 0;
    uint64_t zero_frames_ = 0;
};

} // namespace audio_tap

#endif // AUDIO_TAP_H
