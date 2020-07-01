/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvInfer.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

const std::string gSampleName = "Type1a";

/**
 * This example is derived from the TensorRT samples published at
 * https://github.com/NVIDIA/TensorRT. The aim of this example is to test
 * TensorRT networks that have tensors with multiple types.
 */
class TrtExample {
  template <typename T>
  using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
  TrtExample() : mEngine(nullptr) {}

  //!
  //! \brief Function builds the network engine
  //!
  bool build();

  //!
  //! \brief Runs the TensorRT inference engine for this sample
  //!
  bool infer();

private:
  std::shared_ptr<nvinfer1::ICudaEngine>
      mEngine; //!< The TensorRT engine used to run the network

  //!
  //! \brief Uses the TensorRT API to create the Network
  //!
  bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                        SampleUniquePtr<nvinfer1::INetworkDefinition> &network,
                        SampleUniquePtr<nvinfer1::IBuilderConfig> &config);

  //!
  //! \brief Reads the input  and stores the result in a managed buffer
  //!
  bool processInput(const samplesCommon::BufferManager &buffers);

  //!
  //! \brief Classifies digits and verify result
  //!
  bool verifyOutput(const samplesCommon::BufferManager &buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the engine
//!
//! \details This function creates the network by using the API to create a
//!          model and builds the engine that will be used to run the network
//!
//! \return Returns true if the engine was created successfully and false
//! otherwise
//!
bool TrtExample::build() {
  auto builder = SampleUniquePtr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
  if (!builder) {
    return false;
  }
  uint32_t flags =
      1U << static_cast<int>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(flags));
  if (!network) {
    return false;
  }

  auto config =
      SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    return false;
  }

  auto constructed = constructNetwork(builder, network, config);
  if (!constructed) {
    return false;
  }

  return true;
}

//!
//! \brief Uses the API to create the Network
//!
bool TrtExample::constructNetwork(
    SampleUniquePtr<nvinfer1::IBuilder> &builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition> &network,
    SampleUniquePtr<nvinfer1::IBuilderConfig> &config) {
  nvinfer1::Dims dims{4, {1, 2, 3, 4}};

  nvinfer1::ITensor *input =
      network->addInput("input", nvinfer1::DataType::kFLOAT, dims);
  assert(input);

  nvinfer1::IShapeLayer *A = network->addShape(*input);
  assert(A);
  A->setName("A");
  nvinfer1::ITensor *x = A->getOutput(0);

  // nvinfer1::IActivationLayer *B =
  //    network->addActivation(*x, nvinfer1::ActivationType::kRELU);

  auto *B = network->addUnary(*x, nvinfer1::UnaryOperation::kNEG);

  assert(B);
  B->setName("B");

  nvinfer1::ITensor *y = B->getOutput(0);
  y->setName("output");
  network->markOutput(*y);

  config->setMaxWorkspaceSize(16_MiB);
  mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
      builder->buildEngineWithConfig(*network, *config),
      samplesCommon::InferDeleter());

  if (!mEngine) {
    return false;
  }
  gLogInfo << "Engine constructed successfully" << std::endl;
  return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It
//! allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool TrtExample::infer() {
  auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(
      mEngine->createExecutionContext());
  if (!context) {
    return false;
  }
  // Create RAII buffer manager object
  samplesCommon::BufferManager buffers(mEngine, 0, context.get());

  int n_inputs = 0;
  for (int i = 0; i < mEngine->getNbBindings(); i++) {
    if (mEngine->bindingIsInput(i))
      n_inputs++;
  }
  if (n_inputs > 0) {
    auto input_dims = context->getBindingDimensions(0);

    std::vector<float> values(24, 1);

    // Read the input data into the managed buffers
    float *hostShapeBuffer =
        static_cast<float *>(buffers.getHostBuffer("input"));
    for (int i = 0; i < values.size(); i++) {
      std::cout << "Setting input value " << i << ": " << values[i] << "\n";
      hostShapeBuffer[i] = values[i];
    }
    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();
  }

  bool status = context->executeV2(buffers.getDeviceBindings().data());
  if (!status) {
    return false;
  }

  // Memcpy from device output buffers to host output buffers
  buffers.copyOutputToHost();

  // Verify results
  std::vector<float> expected_output{2, 4, 6, 8};

  // float *res = static_cast<float *>(buffers.getHostBuffer("output"));
  int *res = static_cast<int *>(buffers.getHostBuffer("output"));
  std::cout << "\nOutput:\n" << std::endl;
  bool correct = true;
  for (int i = 0; i < expected_output.size(); i++) {
    if (std::abs(res[i] - expected_output[i]) > 0.00001) {
      std::cout << i << ": error incorrect value " << res[i] << " vs "
                << expected_output[i] << "\n";
      correct = false;
    } else {
      std::cout << i << ": " << res[i] << "\n";
    }
  }
  return correct;
}

int main(int argc, char **argv) {
  auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);
  gLogger.reportTestStart(sampleTest);

  TrtExample sample;

  gLogInfo << "Building and running inference engine for shape example"
           << std::endl;

  if (!sample.build()) {
    return gLogger.reportFail(sampleTest);
  }
  if (!sample.infer()) {
    return gLogger.reportFail(sampleTest);
  }
  return gLogger.reportPass(sampleTest);
}
