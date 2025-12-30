#include "engine_manager.h"
#include <fstream>
#include <iostream>
#include <cuda_runtime_api.h>

EngineManager::EngineManager(const std::string& model_dir, bool use_fp16) 
    : model_dir_(model_dir), use_fp16_(use_fp16) {
    runtime_ = nvinfer1::createInferRuntime(gLogger);
}

EngineManager::~EngineManager() {
    for (auto& pair : engines_) {
        // TRT 10+ uses smart pointers or unique_ptr-like behavior for some, 
        // but traditionally we delete context, then engine, then runtime.
        if (pair.second->context) pair.second->context->destroy();
        if (pair.second->engine) pair.second->engine->destroy();
    }
    if (runtime_) runtime_->destroy();
}

bool EngineManager::load_engine(const std::string& name) {
    std::string path = model_dir_ + "/" + name + ".engine";
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Engine file not found: " << path << std::endl;
        return false;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> data(size);
    file.read(data.data(), size);

    auto engine = runtime_->deserializeCudaEngine(data.data(), size);
    if (!engine) {
        std::cerr << "Failed to deserialize engine: " << name << std::endl;
        return false;
    }

    auto context = engine->createExecutionContext();
    auto instance = std::make_unique<EngineInstance>();
    instance->engine = engine;
    instance->context = context;

    engines_[name] = std::move(instance);
    return true;
}

void EngineManager::run_inference(const std::string& name, 
                                 const std::map<std::string, void*>& inputs,
                                 const std::map<std::string, void*>& outputs,
                                 const std::map<std::string, std::vector<int64_t>>& shapes) {
    auto it = engines_.find(name);
    if (it == engines_.end()) return;

    auto& instance = it->second;
    
    // Set dynamic shapes
    for (const auto& pair : shapes) {
        int index = instance->engine->getBindingIndex(pair.first.c_str());
        if (index != -1 && instance->engine->bindingIsInput(index)) {
            nvinfer1::Dims dims;
            dims.nbDims = pair.second.size();
            for (size_t i = 0; i < pair.second.size(); ++i) dims.d[i] = pair.second[i];
            instance->context->setBindingDimensions(index, dims);
        }
    }

    // Set pointers
    std::vector<void*> bindings(instance->engine->getNbBindings());
    for (int i = 0; i < instance->engine->getNbBindings(); ++i) {
        const char* b_name = instance->engine->getBindingName(i);
        if (inputs.count(b_name)) bindings[i] = inputs.at(b_name);
        else if (outputs.count(b_name)) bindings[i] = outputs.at(b_name);
    }

    instance->context->executeV2(bindings.data());
}
