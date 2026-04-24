import tensorflow as tf
import numpy as np
import sys
import os
import inspect

def fused_matmul_add_relu_A0732AD9DB33D09F(path):
    # test if .so file exists
    so_path = os.path.join(path, 'outputs/so/fused_matmul_add_relu_A0732AD9DB33D09F.so')
    if not os.path.exists(so_path):
        print(f"Error: {so_path} not found, compilation likely failed")
        return False

    # test if .so file is loadable
    try:
        fused_matmul_add_relu_A0732AD9DB33D09F_module = tf.load_op_library(so_path)
    except Exception as e:
        print(f"Error loading .so: {e}")
        return False

    # test if op is available
    found = False
    # print("printing all available commands")
    for attr in dir(fused_matmul_add_relu_A0732AD9DB33D09F_module):
        print(f"  - {attr}")
        if "fused_matmul_add_relu_a0732ad9db33d09f" in attr:
            found = True
            break
    # print("command printing finished")

    if not found: 
        print("fused_matmul_add_relu_A0732AD9DB33D09F command not found")
        return False

    try:
        op = fused_matmul_add_relu_A0732AD9DB33D09F_module.fused_matmul_add_relu_a0732ad9db33d09f
        print("successfully got op")
    except Exception as e:
        print(f"Error getting fused_matmul_add_relu_A0732AD9DB33D09F op: {e}")
        return False

    all_passed = True

    # test if parameter matches
    try:
        # Create test tensors matching the IR signature
        input_data = np.random.randn(4, 8).astype(np.float32)
        weight_data = np.random.randn(8, 2).astype(np.float32)
        buffer_data = np.zeros((4, 2), dtype=np.float32)  # This is the matmul output buffer
        bias_data = np.random.randn(2).astype(np.float32)
        
        print(f"Creating tensors with shapes:")
        print(f"  Input: {input_data.shape}")
        print(f"  Weight: {weight_data.shape}")
        print(f"  Buffer: {buffer_data.shape}")
        print(f"  Bias: {bias_data.shape}")
        
        tf_input = tf.constant(input_data)
        tf_weight = tf.constant(weight_data)
        tf_buffer = tf.constant(buffer_data)
        tf_bias = tf.constant(bias_data)
        
        # Try calling with 4 parameters
        result = op(tf_input, tf_weight, tf_buffer, tf_bias)
        print(f"Successfully called op with 4 parameters")
        print(f"Result shape: {result.shape}")
    except Exception as e:
        print(f"op requires 4 parameters: input, weight, buffer, bias: {e}")
        return False

    try:
        print("\n" + "="*60)
        print("")
        print("="*60)
        
        # 
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # 
        batch_size = np.random.randint(1, 10)
        in_features = np.random.randint(1, 20)
        out_features = np.random.randint(1, 20)
        
        print(f"\n:")
        print(f"  batch_size: {batch_size}")
        print(f"  in_features: {in_features}")
        print(f"  out_features: {out_features}")
        
        # 
        input_data = np.random.randn(batch_size, in_features).astype(np.float32)
        weight_data = np.random.randn(in_features, out_features).astype(np.float32)
        bias_data = np.random.randn(out_features).astype(np.float32)
        buffer_data = np.zeros((batch_size, out_features), dtype=np.float32)
        
        print(f"\n:")
        print(f"  Input: [{input_data.min():.4f}, {input_data.max():.4f}]")
        print(f"  Weight: [{weight_data.min():.4f}, {weight_data.max():.4f}]")
        print(f"  Bias: [{bias_data.min():.4f}, {bias_data.max():.4f}]")
        
        # 1. TensorFlow
        print(f"\n--- TensorFlow ---")
        tf_input = tf.constant(input_data)
        tf_weight = tf.constant(weight_data)
        tf_bias = tf.constant(bias_data)
        
        # matmul + add + relu
        tf_matmul = tf.matmul(tf_input, tf_weight)
        tf_add = tf_matmul + tf_bias
        tf_expected = tf.nn.relu(tf_add)
        
        # 2. op
        print(f"--- op ---")
        tf_buffer = tf.constant(buffer_data)
        op_result = op(tf_input, tf_weight, tf_buffer, tf_bias)
        
        # numpy
        expected_np = tf_expected.numpy()
        actual_np = op_result.numpy()
        
        print(f"\n:")
        print(f"  TensorFlow: [{expected_np.min():.6f}, {expected_np.max():.6f}]")
        print(f"  op: [{actual_np.min():.6f}, {actual_np.max():.6f}]")
        
        # 
        abs_diff = np.abs(expected_np - actual_np)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        rel_diff = abs_diff / (np.abs(expected_np) + 1e-8)  # 0
        max_rel_diff = np.max(rel_diff)
        
        print(f"\n:")
        print(f"  : {max_abs_diff:.6e}")
        print(f"  : {mean_abs_diff:.6e}")
        print(f"  : {max_rel_diff:.6e}")
        
        # 
        abs_tolerance = 1e-5
        rel_tolerance = 1e-4
        
        # ReLU
        print(f"\n--- ReLU ---")
        negative_indices = np.where(tf_add.numpy() < 0)
        negative_count = len(negative_indices[0])
        print(f"  ReLU: {negative_count}")
        
        if negative_count > 0:
            # 0
            negative_after_relu = actual_np[negative_indices]
            if np.all(negative_after_relu == 0):
                print("   0")
            else:
                print(f"   {np.sum(negative_after_relu != 0)}0")
                all_passed = False
        
        # 
        print(f"\n---  ---")
        if max_abs_diff < abs_tolerance and max_rel_diff < rel_tolerance:
            print(f"    ()")
        else:
            print(f"   ")
            if max_abs_diff >= abs_tolerance:
                print(f"     {max_abs_diff:.6e} >=  {abs_tolerance}")
            if max_rel_diff >= rel_tolerance:
                print(f"     {max_rel_diff:.6e} >=  {rel_tolerance}")
            all_passed = False
        
        # 
        if batch_size * out_features <= 20:  # 
            print(f"\n (10):")
            expected_flat = expected_np.flatten()
            actual_flat = actual_np.flatten()
            for i in range(min(10, len(expected_flat))):
                match = "" if abs(expected_flat[i] - actual_flat[i]) < 1e-5 else ""
                print(f"  [{i}] TF: {expected_flat[i]:.6f}, Custom: {actual_flat[i]:.6f} {match}")
        
    except Exception as e:
        print(f": {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    return all_passed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
        
    path = sys.argv[1]
    success = fused_matmul_add_relu_A0732AD9DB33D09F(path)
    if success:
        print("TEST COMPLETED")
    else:
        print("TEST FAILED")
        sys.exit(1)