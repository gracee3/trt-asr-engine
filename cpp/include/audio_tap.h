// audio_tap.h - RAII audio tap writer for pipeline debugging
//
// Writes raw PCM + JSON sidecar for offline analysis.
// Env-var gated: set AUDIO_TAP_ENABLE=1 to activate.
//
// Usage:
//   AudioTapWriter tap("post_dsp", 16000, 1, AudioTapWriter::Format::F32LE);
//   tap.write(samples, num_samples);
//   // On destruction, JSON sidecar is finalized with total sample count
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

namespace audio_tap {

// Check if taps are globally enabled via environment variable
inline bool taps_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("AUDIO_TAP_ENABLE");
        enabled = (env && (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0));
    }
    return enabled == 1;
}

// Get tap output directory (default: current directory)
inline std::string tap_directory() {
    const char* env = std::getenv("AUDIO_TAP_DIR");
    return env ? std::string(env) : ".";
}

// Sample format specifier
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

// Per-chunk statistics (accumulated during write)
struct TapStats {
    double peak = 0.0;           // Max absolute value
    double sum = 0.0;            // Running sum for mean
    double sum_sq = 0.0;         // Running sum of squares for RMS
    uint64_t total_samples = 0;
    uint64_t clipped_samples = 0;  // Only for integer formats
    uint64_t nan_count = 0;
    uint64_t inf_count = 0;
    uint64_t gap_samples = 0;      // Explicit zero-fill for gaps

    void reset() {
        peak = sum = sum_sq = 0.0;
        total_samples = clipped_samples = nan_count = inf_count = gap_samples = 0;
    }

    double mean() const {
        return total_samples > 0 ? sum / total_samples : 0.0;
    }

    double rms() const {
        return total_samples > 0 ? std::sqrt(sum_sq / total_samples) : 0.0;
    }

    double dc_offset() const { return mean(); }
};

// RAII audio tap writer
class AudioTapWriter {
public:
    // Create a tap writer. If taps are disabled globally, this is a no-op.
    // stage_name: identifies this tap point (e.g., "capture", "post_dsp", "pre_stt")
    // sample_rate: Hz
    // channels: number of interleaved channels
    // fmt: sample format
    AudioTapWriter(const std::string& stage_name,
                   uint32_t sample_rate,
                   uint16_t channels,
                   Format fmt,
                   const std::string& notes = "")
        : stage_name_(stage_name)
        , sample_rate_(sample_rate)
        , channels_(channels)
        , format_(fmt)
        , notes_(notes)
        , enabled_(taps_enabled())
        , raw_file_(nullptr)
    {
        if (!enabled_) return;

        // Record start time
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        start_monotonic_ns_ = ts.tv_sec * 1000000000ULL + ts.tv_nsec;

        // Open raw file for append
        std::string dir = tap_directory();
        std::string raw_path = dir + "/tap_" + stage_name_ + ".raw";
        raw_file_ = std::fopen(raw_path.c_str(), "ab");
        if (!raw_file_) {
            fprintf(stderr, "[AudioTap] ERROR: Failed to open %s\n", raw_path.c_str());
            enabled_ = false;
            return;
        }

        fprintf(stderr, "[AudioTap] Opened tap '%s': %u Hz, %u ch, %s\n",
                stage_name_.c_str(), sample_rate_, channels_, format_to_string(fmt));
    }

    ~AudioTapWriter() {
        if (!enabled_) return;

        if (raw_file_) {
            std::fclose(raw_file_);
        }

        // Write JSON sidecar
        std::string dir = tap_directory();
        std::string json_path = dir + "/tap_" + stage_name_ + ".json";
        FILE* json_file = std::fopen(json_path.c_str(), "w");
        if (json_file) {
            fprintf(json_file,
                "{\n"
                "  \"sample_rate_hz\": %u,\n"
                "  \"channels\": %u,\n"
                "  \"format\": \"%s\",\n"
                "  \"interleaved\": true,\n"
                "  \"start_monotonic_ns\": %llu,\n"
                "  \"total_samples\": %llu,\n"
                "  \"total_frames\": %llu,\n"
                "  \"duration_sec\": %.3f,\n"
                "  \"stats\": {\n"
                "    \"peak\": %.6g,\n"
                "    \"rms\": %.6g,\n"
                "    \"dc_offset\": %.6g,\n"
                "    \"clipped_samples\": %llu,\n"
                "    \"nan_count\": %llu,\n"
                "    \"inf_count\": %llu,\n"
                "    \"gap_samples\": %llu\n"
                "  },\n"
                "  \"notes\": \"%s\"\n"
                "}\n",
                sample_rate_,
                channels_,
                format_to_string(format_),
                (unsigned long long)start_monotonic_ns_,
                (unsigned long long)stats_.total_samples,
                (unsigned long long)(stats_.total_samples / channels_),
                (double)(stats_.total_samples / channels_) / sample_rate_,
                stats_.peak,
                stats_.rms(),
                stats_.dc_offset(),
                (unsigned long long)stats_.clipped_samples,
                (unsigned long long)stats_.nan_count,
                (unsigned long long)stats_.inf_count,
                (unsigned long long)stats_.gap_samples,
                notes_.c_str()
            );
            std::fclose(json_file);

            fprintf(stderr, "[AudioTap] Closed tap '%s': %llu samples, peak=%.4f, rms=%.4f, dc=%.6f\n",
                    stage_name_.c_str(),
                    (unsigned long long)stats_.total_samples,
                    stats_.peak, stats_.rms(), stats_.dc_offset());
        }
    }

    // Non-copyable, movable
    AudioTapWriter(const AudioTapWriter&) = delete;
    AudioTapWriter& operator=(const AudioTapWriter&) = delete;
    AudioTapWriter(AudioTapWriter&& other) noexcept { *this = std::move(other); }
    AudioTapWriter& operator=(AudioTapWriter&& other) noexcept {
        if (this != &other) {
            stage_name_ = std::move(other.stage_name_);
            sample_rate_ = other.sample_rate_;
            channels_ = other.channels_;
            format_ = other.format_;
            notes_ = std::move(other.notes_);
            enabled_ = other.enabled_;
            start_monotonic_ns_ = other.start_monotonic_ns_;
            raw_file_ = other.raw_file_;
            stats_ = other.stats_;
            other.raw_file_ = nullptr;
            other.enabled_ = false;
        }
        return *this;
    }

    bool is_enabled() const { return enabled_; }
    const TapStats& stats() const { return stats_; }

    // Write samples (template for different types)
    // data: pointer to interleaved samples
    // num_samples: total sample count (frames * channels)
    template<typename T>
    void write(const T* data, size_t num_samples) {
        if (!enabled_ || !raw_file_ || num_samples == 0) return;

        std::lock_guard<std::mutex> lock(mutex_);

        // Update stats
        for (size_t i = 0; i < num_samples; ++i) {
            double val = static_cast<double>(data[i]);

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

            // Check clipping for integer formats
            if constexpr (std::is_same_v<T, int16_t>) {
                if (data[i] == INT16_MAX || data[i] == INT16_MIN) {
                    stats_.clipped_samples++;
                }
            } else if constexpr (std::is_same_v<T, int32_t>) {
                if (data[i] == INT32_MAX || data[i] == INT32_MIN) {
                    stats_.clipped_samples++;
                }
            }
        }
        stats_.total_samples += num_samples;

        // Write raw bytes
        size_t bytes = num_samples * format_bytes_per_sample(format_);
        std::fwrite(data, 1, bytes, raw_file_);
    }

    // Convenience: write with explicit format verification
    void write_f32(const float* data, size_t num_samples) {
        if (format_ != Format::F32LE) {
            fprintf(stderr, "[AudioTap] WARNING: write_f32 called but format is %s\n",
                    format_to_string(format_));
        }
        write(data, num_samples);
    }

    void write_s16(const int16_t* data, size_t num_samples) {
        if (format_ != Format::S16LE) {
            fprintf(stderr, "[AudioTap] WARNING: write_s16 called but format is %s\n",
                    format_to_string(format_));
        }
        write(data, num_samples);
    }

    // Record a gap (dropped samples) - writes zeros and tracks in stats
    void record_gap(size_t num_samples) {
        if (!enabled_ || !raw_file_ || num_samples == 0) return;

        std::lock_guard<std::mutex> lock(mutex_);

        // Write zeros
        size_t bytes_per_sample = format_bytes_per_sample(format_);
        std::vector<uint8_t> zeros(num_samples * bytes_per_sample, 0);
        std::fwrite(zeros.data(), 1, zeros.size(), raw_file_);

        stats_.gap_samples += num_samples;
        stats_.total_samples += num_samples;
        // Note: zeros don't affect sum/sum_sq (val=0), but do affect mean calculation
    }

    // Flush to disk (useful for long-running sessions)
    void flush() {
        if (enabled_ && raw_file_) {
            std::fflush(raw_file_);
        }
    }

private:
    std::string stage_name_;
    uint32_t sample_rate_;
    uint16_t channels_;
    Format format_;
    std::string notes_;
    bool enabled_;
    uint64_t start_monotonic_ns_ = 0;
    FILE* raw_file_;
    TapStats stats_;
    std::mutex mutex_;
};

// Convenience: global tap registry for long-lived taps
// Use AUDIO_TAP_<NAME>=1 to enable individual taps
class TapRegistry {
public:
    static TapRegistry& instance() {
        static TapRegistry reg;
        return reg;
    }

    // Check if a specific tap is enabled (AUDIO_TAP_<NAME>=1)
    bool is_tap_enabled(const std::string& name) {
        std::string env_name = "AUDIO_TAP_" + name;
        // Convert to uppercase
        for (char& c : env_name) c = std::toupper(c);
        const char* env = std::getenv(env_name.c_str());
        return env && (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0);
    }

private:
    TapRegistry() = default;
};

} // namespace audio_tap

#endif // AUDIO_TAP_H
