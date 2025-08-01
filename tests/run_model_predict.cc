#include <iostream>
#include <vector>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/events_writer.h"

using namespace tensorflow;

#define DEFAULT_BATCH_SIZE 16

void create_tensor(const DataType& dtype, Tensor* tensor) {
  const int period = 7;
  if (dtype == DT_FLOAT) {
    auto flat = tensor->flat<float>();
    for (int i = 0; i < flat.size(); ++i)
      flat(i) = static_cast<float>(i % period) / 10.0f;
  } else if (dtype == DT_INT64) {
    auto flat = tensor->flat<int64>();
    for (int i = 0; i < flat.size(); ++i) flat(i) = i % period;
  } else if (dtype == DT_INT32) {
    auto flat = tensor->flat<int32>();
    for (int i = 0; i < flat.size(); ++i) flat(i) = i % period;
  } else if (dtype == DT_STRING) {
    auto flat = tensor->flat<tstring>();
    for (int i = 0; i < flat.size(); ++i) flat(i) = std::to_string(i % period);
  } else if (dtype != DT_RESOURCE && dtype != DT_VARIANT) {
    memset(const_cast<char*>(tensor->tensor_data().data()), 0,
           tensor->tensor_data().size());
  } else {
    LOG(ERROR) << "Invalid dtype: " << dtype;
  }
}

void create_model_io_defs(const std::string& model_path,
                          std::vector<std::pair<std::string, Tensor>>& inputs,
                          std::vector<std::string>& output_defs) {
  SavedModel saved_model;
  const string pb_path = model_path + "/saved_model.pb";
  const string pbtxt_path = model_path + "/saved_model.pbtxt";
  if (Env::Default()->FileExists(pb_path).ok()) {
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), pb_path, &saved_model));
  } else if (Env::Default()->FileExists(pbtxt_path).ok()) {
    TF_CHECK_OK(ReadTextProto(Env::Default(), pbtxt_path, &saved_model));
  } else {
    LOG(ERROR) << "Saved model not exists: " << model_path;
  }
  MetaGraphDef* meta_graph = saved_model.mutable_meta_graphs(0);
  const GraphDef gdef = meta_graph->graph_def();

  // set model input tensors
  for (int i = 0; i < gdef.node_size(); ++i) {
    const NodeDef& ndef = gdef.node(i);
    if (ndef.op() != "Placeholder") continue;
    const std::string name = ndef.name();
    const DataType dtype = ndef.attr().at("dtype").type();
    const TensorShapeProto& shape = ndef.attr().at("shape").shape();
    TensorShape t_shape;
    for (int j = 0; j < shape.dim_size(); ++j) {
      int dim = shape.dim(j).size();
      if (dim == -1) dim = DEFAULT_BATCH_SIZE;
      t_shape.AddDim(dim);
    }
    Tensor input_tensor(dtype, t_shape);
    create_tensor(dtype, &input_tensor);
    inputs.push_back(std::make_pair(name, input_tensor));
  }

  // set model outputs
  if (meta_graph->signature_def().count("serving_default")) {
    const auto& sdef = meta_graph->signature_def().at("serving_default");
    for (auto& output_elem : sdef.outputs())
      output_defs.push_back(output_elem.second.name());
  }
  if (output_defs.empty()) {
    int counter = 0;
    for (int i = gdef.node_size() - 1; i >= 0; --i) {
      const NodeDef& ndef = gdef.node(i);
      if (ndef.op() != "NoOp" && ndef.op() != "Assign" &&
          ndef.op() != "AssignVariableOp" && ndef.op() != "Identity" &&
          ndef.op() != "Const" && ndef.op() != "VarHandleOp" &&
          ndef.op() != "ReadVariableOp") {
        output_defs.push_back(ndef.name() + ":0");
        counter++;
        if (counter > 2) break;
      }
    }
  }
}

int main(int argc, char** argv) {
  const std::string export_dir = argv[1];

  tensorflow::SavedModelBundle bundle;
  tensorflow::SessionOptions session_options;
  session_options.config.set_intra_op_parallelism_threads(4);
  session_options.config.set_inter_op_parallelism_threads(2);
  tensorflow::RunOptions run_options;

  tensorflow::Status status = tensorflow::LoadSavedModel(
      session_options, run_options, export_dir, {"serve"}, &bundle);

  if (!status.ok()) {
    LOG(ERROR) << "Error loading model: " << status.ToString();
    return 1;
  }

  std::vector<std::pair<std::string, Tensor>> inputs;
  std::vector<std::string> output_defs;
  create_model_io_defs(export_dir, inputs, output_defs);
  std::vector<tensorflow::Tensor> outputs;

  run_options.set_trace_level(RunOptions::FULL_TRACE);
  RunMetadata run_metadata;
  status = bundle.session->Run(run_options, inputs, output_defs, {}, &outputs,
                               &run_metadata);

  if (!status.ok()) {
    LOG(ERROR) << "ERROR: " << status.ToString();
    return 1;
  }
  LOG(INFO) << "Inference success";
  for (Tensor& output_tensor : outputs) {
    VLOG(1) << output_tensor.shape().DebugString();
    if (output_tensor.dtype() == DT_FLOAT) {
      auto output_data = output_tensor.flat<float>();
      for (int i = 0; i < 10 && i < output_data.size(); ++i) {
        VLOG(1) << output_data(i);
      }
    }
  }

  for (const auto& dev_stats : run_metadata.step_stats().dev_stats()) {
    VLOG(1) << "Device: " << dev_stats.device();
    for (const auto& node_stats : dev_stats.node_stats()) {
      VLOG(1) << "  Node: " << node_stats.node_name()
              << ", Start(us): " << node_stats.all_start_micros()
              << ", Duration(us): "
              << (node_stats.op_end_rel_micros() -
                  node_stats.op_start_rel_micros());
    }
  }
  return 0;
}