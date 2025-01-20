#include "annc/service/cpu/kdnn_rewriter.h"
#include "xla/service/hlo_parser.h"

#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace xla;

int main(int argc, char** argv) {
  std::string hlo_text_file = argv[1];
  std::ifstream file(hlo_text_file);
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string hlo_text = buffer.str();
  file.close();

  absl::StatusOr<std::unique_ptr<xla::HloModule>> 
    m_ = ParseAndReturnUnverifiedModule(hlo_text.c_str());

  xla::cpu::KDnnFusionAfterHloLayoutAssign kdnn_fusion;
  kdnn_fusion.Run(m_->get(), {});

  printf("%s\n", m_->get()->ToString().c_str());

  return 0;
}