#include <dlfcn.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

int main(int argc, char** argv) {
  if (argc != 4) return 1;
  for (const char* library : {argv[1], argv[2]}) {
    if (!dlopen(library, RTLD_NOW | RTLD_LOCAL)) {
      std::cerr << dlerror() << '\n';
      return 1;
    }
  }
  tensorflow::GraphDef graph;
  TF_CHECK_OK(
      tensorflow::ReadTextProto(tensorflow::Env::Default(), argv[3], &graph));
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession({}));
  TF_CHECK_OK(session->Create(graph));
  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session->Run({}, {"fused:0", "reference:0"}, {}, &outputs));
  TF_CHECK_OK(session->Close());
  if (outputs.size() != 2 || outputs[0].dtype() != tensorflow::DT_FLOAT ||
      outputs[0].dtype() != outputs[1].dtype() ||
      outputs[0].shape() != outputs[1].shape())
    return 1;
  auto actual = outputs[0].flat<float>();
  auto expected = outputs[1].flat<float>();
  if (!std::equal(expected.data(), expected.data() + expected.size(),
                  actual.data())) {
    std::cerr << "ANNCFused result differs from TensorFlow Add\n";
    return 1;
  }
  return 0;
}
