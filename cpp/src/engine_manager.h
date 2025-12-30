#ifndef ENGINE_MANAGER_H
#define ENGINE_MANAGER_H

#include <NvInfer.h>
#include <string>
#include <vector>
#include <memory>
#include <map>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} gLogger;

class EngineManager {
public:
    EngineManager(const std::string& model_dir, bool use_fp16);
    ~EngineManager();

    bool load_engine(const std::string& name);
    
    // Low-level inference interface
    void run_inference(const std::string& name, 
                      const std::map<std::string, void*>& inputs,
                      const std::map<std::string, void*>& outputs,
                      const std::map<std::string, std::vector<int64_t>>& shapes);

private:
    std::string model_dir_;
    bool use_fp16_;
    nvinfer1::IRuntime* runtime_;

    struct EngineInstance {
        nvinfer1::ICudaEngine* engine;
        nvinfer1::IExecutionContext* context;
    };

    std::map<std::string, std::unique_ptr<EngineInstance>> engines_;
};

#endif
