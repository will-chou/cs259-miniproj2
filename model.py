import sys

"""Our model for CUDNN

Take the maximum of compute bound time, buffer bandwidth bound time, and the memory bound time
Compute bound time calculated .....
Buffer bandwidth bound time calculated .....
Memory bound time calculated .....

    :param M, N, K: describes the triple (M, N, K) of a matrix multiplication problem
    :return: returns execution time as a float (usec)

"""
def model(M, N, K):
    # compute bound time
    NUM_CORES = 5120
    CORE_CLOCK = 1200

    compute_bound_time = (2*K*M*N/NUM_CORES)/CORE_CLOCK



    # memory bound time
    L2_CACHE_SIZE = 4500000
    MEMORY_BANDWIDTH = 6.53 * (10 ** 11)
    NUM_ELEMENTS_TO_READ = M * K + K * N
    FLOAT_SIZE = sys.getsizeof(float)
    if NUM_ELEMENTS_TO_READ <= L2_CACHE_SIZE: # able to fit in L2 cache
        memory_bound_time = NUM_ELEMENTS_TO_READ * FLOAT_SIZE / MEMORY_BANDWIDTH
    else:
        memory_bound_time = NUM_ELEMENTS_TO_READ * FLOAT_SIZE / MEMORY_BANDWIDTH * NUM_ELEMENTS_TO_READ/L2_CACHE_SIZE

    # Return the maximum of the compute bound time, bandwidth bound time, and memory bound time
    # print(compute_bound_time)
    # print(memory_bound_time)
    return max(compute_bound_time, memory_bound_time)

"""Compares our model's CUDNN with DeepBench's CUDNN

DeepBench results stored in deepbench_benchmarks.txt
They contain 76 results using different inputs M, N, K
DeepBench uses floats and does not use transposed convolution
Accuracy is calculated with mean-squared error
"""
def main():
    file1 = open('deepbench_benchmarks.txt', 'r')
    Lines = file1.readlines()
    mean_squared_error = 0
    print('{:<10s}{:<20s}{:<30s}{:<40s}{:<50s}'.format('M','N','K','CUDNN execution time','Our Model Estimate'))
    for line in Lines[1:]:
        contents = line.split()
        m = int(contents[0])
        n = int(contents[1])
        k = int(contents[2])

        time_cudnn = float(contents[-1])
        our_model_time = model(m, n, k)
        mean_squared_error += ((time_cudnn - our_model_time)**2)
        print('{:<10s}{:<20s}{:<30s}{:<40s}{:<50s}'.format(str(m),str(n),str(k),str(time_cudnn),str(our_model_time)))
    #print(mean_squared_error)
    #print(len(Lines[1:]))
    mean_squared_error = mean_squared_error / len(Lines[1:])
    print("Total mean squared error: " + str(mean_squared_error))


if __name__ == "__main__":
    main()
