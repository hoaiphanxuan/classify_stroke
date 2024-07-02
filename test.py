import tensorflow as tf

def check_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPU is available. GPU details:")
        for gpu in gpus:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"  GPU: {gpu.name}")
            print(f"    Memory: {details.get('memory_size', 'N/A') / 1024**3:.2f} GB")
            print(f"    Compute Capability: {details.get('compute_capability', 'N/A')}")
    else:
        print("GPU is not available on this device.")

check_gpu()