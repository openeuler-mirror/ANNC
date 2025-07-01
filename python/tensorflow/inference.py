import os
import random
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf

"""
To capture profiling data for tensorboard
pip install tensorflow==2.xx
# the same version as tensorflow
pip install tensorboard_plugin_profile=2.xx
pip install etils, importlib_resources

tensorboard --logdir=./log_path --host=0.0.0.0
"""

def generate_random_inputs(func, batch_size: int = 16):
    inputs = {}
    for k, v in func.structured_input_signature[1].items():
        shape = [batch_size if i is None else i for i in v.shape]
        if v.dtype == tf.string:
            inputs[k] = tf.constant(shape=shape, value=str(random.random()), dtype=tf.string)
        elif v.dtype == tf.float32:
            inputs[k] = tf.constant(shape=shape, value=random.random(), dtype=tf.float32)
        elif v.dtype == tf.double:
            inputs[k] = tf.constant(shape=shape, value=random.random(), dtype=tf.double)
        elif v.dtype == tf.float16:
            inputs[k] = tf.constant(shape=shape, value=random.random(), dtype=tf.float16)
        elif v.dtype == tf.int32:
            inputs[k] = tf.constant(shape=shape, value=random.randint(-1000, 1000), dtype=tf.int32)
        elif v.dtype == tf.int64:
            inputs[k] = tf.constant(shape=shape, value=random.randint(-1000, 1000), dtype=tf.int64)
        else:
            raise TypeError(f'Unsupport data type: {v.dtype}')
    return inputs


def inference(func, inputs: dict, run_times: int = 1, profiling_path: str = None):
    if profiling_path:
        # warm-up
        func(**inputs)
    
    if profiling_path:
        options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3,
                                                           python_tracer_level=1,
                                                           device_tracer_level=0)
        tf.profiler.experimental.start(profiling_path, options=options)
    
    for _ in range(run_times):
        result = func(**inputs)
    
    if profiling_path:
        tf.profiler.experimental.stop()
    
    return result


def compare(res1, res2):
    for key in res1:
        absolute_error = np.abs(res1[key].numpy() - res2[key].numpy())
        relative_error = np.abs((res1[key].numpy() - res2[key].numpy()) / (res2[key].numpy() + 1e-12))
        error_metrics = {
            "max_abs_error": np.max(absolute_error),
            "mean_abs_error": np.mean(absolute_error),
            "max_rel_error": np.max(relative_error),
            "mean_rel_error": np.mean(relative_error),
            "error_std": np.std(absolute_error)
        }
        print('###' * 10, key)
        for k, v in error_metrics.items():
            print('--', k, v)


def main():
    import argparse
    parser = argparse.ArgumentParser('tf-inference-tester')
    parser.add_argument('model', help='Model path')
    parser.add_argument('-L', '--lib', help='User defined libraries')
    parser.add_argument('-P', '--profiling_path', help='Output profiling data path')
    parser.add_argument('-C', '--compare', help='Model to compare inference result')
    parser.add_argument('-B', '--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('-R', '--run_times', type=int, default=1, help='Run times')
    args = parser.parse_args()
    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    imported = tf.saved_model.load(args.model)
    concrete_func = imported.signatures['serving_default']
    
    if args.lib:
        for lib in args.lib.split(':'):
            if not lib:
                continue
            tf.load_op_library(lib)
    
    inputs = generate_random_inputs(concrete_func, args.batch_size)
    result = inference(concrete_func, inputs, args.run_times, args.profiling_path)
    if args.compare:
        imported_ = tf.saved_model.load(args.compare)
        concrete_func_ = imported_.signatures['serving_default']
        result_ = inference(concrete_func_, inputs, args.run_times, args.profiling_path)
        compare(result, result_)
    else:
        print(result)


if __name__ == '__main__':
    main()
