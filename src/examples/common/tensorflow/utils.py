import tensorflow as tf


def new_session_config(parallel: int = 0):
    # session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config = tf.ConfigProto()
    session_config.intra_op_parallelism_threads = parallel
    session_config.inter_op_parallelism_threads = parallel
    session_config.gpu_options.allow_growth = True
    return session_config
