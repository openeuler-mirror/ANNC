import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('annc-application-utils')
    parser.add_argument(
        'path', help='repository path of application (TensorFlow, etc.)')
    parser.add_argument('--force', action='store_true', help='force re-apply')
    return parser.parse_args()


def tf_install():
    args = get_args()
    # apply tensorflow.patch
    # rebuild tf or tfserving
