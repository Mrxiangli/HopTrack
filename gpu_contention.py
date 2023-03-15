import numpy, math, argparse, socket
from numba import cuda

cuda.select_device(0)

# Translation from GPU utiliazation to the internal parameter
GPU_to_num_tx2 = {
    1: 51200,
    10: 269056,
    20: 532608,
    30: 786816,
    40: 1054464,
    50: 1345792,
    60: 1635840,
    70: 1893632,
    80: 2192128,
    90: 2416768,
    99: 2797696}

GPU_to_num_agx_xv = {
    1: 25600,
    10: 112207,
    20: 209134,
    30: 320447,
    40: 421267,
    50: 537432,
    60: 658492,
    70: 721775,
    80: 841952,
    90: 926147,
    99: 2257570}

GPU_to_num_xv_nx = {
    1: 6400,
    10: 69302,
    20: 122771,
    30: 179747,
    40: 263166,
    50: 318430,
    60: 385300,
    70: 466213,
    80: 512834,
    90: 564117,
    99: 682580}

# Kernel function
@cuda.jit
def my_kernel(array):

    # CUDA kernel
    pos = cuda.grid(1)
    tx = cuda.threadIdx.x
    if pos < array.size:
        array[pos] += tx  # element add computation

# GPU contention genarator
def GPU_contention_generator(level):
    ContentionSize = GPU_to_num_agx_xv[level]
    # Covert the GPU utilization level to internal parameter based on device type
    if socket.gethostname() in ['tx2-1', 'tx2-2']:
        ContentionSize = GPU_to_num_tx2[level]
    elif socket.gethostname() in ['xv3']:
        ContentionSize = GPU_to_num_agx_xv[level]

    data = numpy.zeros(ContentionSize)
    multiplier = data.size / 512
    threadsperblock, blockspergrid = 128, 4

    # Copy data to device
    device_data = cuda.to_device(data)
    while True:
        my_kernel[math.ceil(multiplier*blockspergrid), threadsperblock](device_data)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU', type=int, help='GPU contention level')
    args = parser.parse_args()
    GPU_contention_generator(args.GPU)
