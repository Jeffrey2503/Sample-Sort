from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
my_rank = comm.Get_rank()
root = 0

SIZE = 10000  # Define the size for testing

if my_rank == 0:
    # Generating input data
    data = np.random.randint(0, 10000, SIZE)
    #data = np.array([21,17,9,36,32,2,19,31,8,28,35,7,30,11,15,34,4,1,18,10,25,22,12,29,3,26,6,16,20,23,5,37,24,33,14,38,27,13,40,39])
    
    no_of_elements = len(data)

    start = MPI.Wtime()

else:
    data = None
    no_of_elements = None


# Broadcast the number of elements to all processes
no_of_elements = comm.bcast(no_of_elements, root=root)

# Distribute samples to processes
local_size = no_of_elements // num_procs
local_data = np.empty(local_size, dtype=int)
comm.Scatter(data, local_data, root=0)
# print(f"Process {my_rank}: After Scatter - local_data: {local_data}")
# Sort locally
local_data.sort()
print(f"Process {my_rank}: After local sorting - local_data: {local_data}")

# Choose local splitters
splitter = [local_data[no_of_elements // (num_procs * num_procs) * (i + 1)] for i in range(num_procs - 1)]
print(f"Process {my_rank}: Local splitters: {splitter}")
# Gather local splitters at root
all_splitters = comm.gather(splitter, root=root)

# Choose global splitters at root
if my_rank == root:
    all_splitters = np.sort(np.concatenate(all_splitters))
    splitter = [all_splitters[(num_procs - 1) * (i + 1)]
                for i in range(num_procs - 1)]
    print("Sorted Splitters (Root Process):", splitter)
# Broadcast global splitters
splitter = comm.bcast(splitter, root=root)
print("All Splitters:", all_splitters)

# Create buckets locally
buckets = np.zeros((local_size + 1) * num_procs, dtype=int)

j = 0
k = 1
for i in range(local_size):
    if j < (num_procs - 1):
        if local_data[i] < splitter[j]:
            buckets[((local_size + 1) * j) + k] = local_data[i]
            k += 1
        else:
            buckets[((local_size + 1) * j)] = k - 1
            k = 1
            j += 1
    else:
        buckets[((local_size + 1) * j) + k] = local_data[i]


print(f"Process {my_rank}: Bucket content before Alltoall - buckets: {buckets}")


# Sending buckets to respective processors
bucket_buffer = np.zeros((local_size + 1) * num_procs, dtype=int)
comm.Alltoall([buckets, MPI.INT], [bucket_buffer, MPI.INT])

print(f"Process {my_rank}: Bucket buffer after Alltoall - bucket_buffer: {bucket_buffer}")


# Rearrange BucketBuffer
local_bucket = np.zeros(local_size * num_procs, dtype=int)
# Tracks the index for each processor in bucket_buffer
bucket_idx = [0] * num_procs

for i in range(num_procs):
    count = bucket_buffer[(local_size + 1) * i]
    if count > 0:
        start_idx = (local_size + 1) * i + 1
        end_idx = start_idx + count
        local_bucket[bucket_idx[i]: bucket_idx[i] +
                     count] = bucket_buffer[start_idx:end_idx]
        bucket_idx[i] += count

# local_bucket[0] is the count of elements
local_bucket[0] = sum(bucket_idx)

# Gather sorted sub-blocks at root
if my_rank == root:
    output_buffer = np.zeros(num_procs * no_of_elements, dtype=int)
    output = np.zeros(no_of_elements, dtype=int)
else:
    output_buffer = None


print(f"Process {my_rank}: Before Gather - output_buffer: {output_buffer}")
comm.Gather(local_bucket, output_buffer, root=root)
print(f"Process {my_rank}: After Gather - output_buffer: {output_buffer}")


# Rearrange output buffer
'''if my_rank == root:
    count = 0
    for i in range(num_procs):
        k = 1
        for j in range(output_buffer[(2 * no_of_elements // num_procs) * i]):
            output[count] = output_buffer[(
                2 * no_of_elements // num_procs) * i + k]
            count += 1
            print(f"i={i}, j={j}, count={count}, output={output}")'''


if my_rank == root:
    output_buffer.sort()
    output = output_buffer[output_buffer != 0]
    
finish = MPI.Wtime()
total_time = finish - start

# Printing the output
print("Sorted output sequence is:\n", output)
print("Time it took for ", num_procs , "processors" , total_time * 1e6, "micro seconds")
