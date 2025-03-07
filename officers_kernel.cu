#include <algorithm>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cassert>
#include <cuda/std/cstdint>

using namespace cuda::std;

const uint64_t debug_end = (1<<21);

const unsigned BLOCK_SIZE = 128;
const unsigned BLOCK_COUNT = 32;

const unsigned SHARED_BUFFER_SIZE = 32 * 1024; // Must fit in shared memory
const unsigned GLOBAL_BUFFER_SIZE = 64 * 1024; // Must fit in shared memory
const unsigned RARE_VALUE_COUNT = 1584;
// Stored largest position to smallest
__constant__ uint16_t rare_positions[RARE_VALUE_COUNT];
__constant__ uint8_t rare_values[RARE_VALUE_COUNT];

// Masks giving the locations of small rare values
// Has to match BLOCK_SIZE
__constant__ uint32_t low_rare0[4] = { 0x40101013, 0x00004000, 0x00000100, 0x10000004 };
__constant__ uint32_t low_rare1[4] = { 0x92490124, 0x12002004, 0x00400000, 0x00010000 };
__constant__ uint32_t low_rare6[4] = { 0x00000000, 0x00400000, 0x00090000, 0x00002400 };
// If BLOCK_SIZE goes up to 256, we will need low_rare7 too

__host__ bool is_rare(uint16_t x) {
  uint16_t mask = 0b10001;
  return (__builtin_popcount(x & ~mask) % 2) == 0;
}

// Delete bit 1 and shift the rest down
__forceinline__ __host__ __device__ uint8_t compress_officers(uint16_t x) {
  return ((x >> 1) & ~0b1) | (x & 0b1);
}

__device__ uint16_t uncompress_officers_common(uint8_t x) {
  unsigned pop = __popc(x & 0b11110110);
  uint16_t bit1 = (pop%2 == 0) ? 0b10 : 0;
  return (((uint16_t)x & 0b11111110) << 1) | bit1 | ((uint16_t)x & 0b1);
}

__host__ uint16_t uncompress_officers_common_host(uint8_t x) {
  unsigned pop = __builtin_popcount(x & 0b11110110);
  uint16_t bit1 = (pop%2 == 0) ? 0b10 : 0;
  return (((uint16_t)x & 0b11111110) << 1) | bit1 | ((uint16_t)x & 0b1);
}

__device__ uint16_t uncompress_officers_rare(uint8_t x) {
  unsigned pop = __popc(x & 0b11110110);
  uint16_t bit1 = (pop%2 == 0) ? 0 : 0b10;
  return (((uint16_t)x & 0b11111110) << 1) | bit1 | ((uint16_t)x & 0b1);
}


__forceinline__
__device__ void set_bit_8(uint32_t array[8], const unsigned i) {
  __builtin_assume(i < (1u<<8));

  // This definition is bad: the unknown index forces the array to be
  // stored in local memory rather than registers

  // array[i/32] |= 1u << (i%32);

  // This is better: unrolling the loop lets the array go in registers
  
  // #pragma unroll 8
  // for (unsigned j = 0; j < 8; j++) {
  //   if (j == i/32) {
  //     array[j] |= (1u << (i % 32));
  //   }
  // }

  // But this sneaky trick is better still: the clamped funnel shift
  // means that when `j` is too small, the 0 is fully shifted in. And
  // when `j` is too large, the shift distance underflows and again the 0 is
  // fully shifted in.
  #pragma unroll 8
  for (unsigned j = 0; j < 8; j++) {
    array[j] |= __funnelshift_lc(0, 1, i - (32 * j));
  }
}

__forceinline__
__device__ bool get_bit_4(const uint32_t array[4], const unsigned i) {
  __builtin_assume(i < BLOCK_SIZE);
  #pragma unroll 4
  for (unsigned j = 0; j < BLOCK_SIZE/32; j++) {
    if (j == i / 32) {
      return (array[j] & (1u << (i%32))) != 0;
    }
  }
  __builtin_unreachable();
}

__forceinline__
__device__ int lowest_unset(uint32_t array[]) {
  for (unsigned i = 0; i < 256/32; i++) {
    if (array[i] != 0xFFFFFFFF) {
      int bit_pos = __ffs(~array[i]) - 1; // __ffs returns 1-based position
      return i * 32 + bit_pos;
    }
  }
  __builtin_unreachable();
}

__launch_bounds__(BLOCK_SIZE)
__global__ void officers_kernel(unsigned long long start_position, uint8_t *global_buffer, unsigned long long *global_progress) {
  // Buffer stores the 'compressed' values to fit in a byte each
  __shared__ uint8_t buffer[SHARED_BUFFER_SIZE]; // 32kb i.e. indexed by 15 bits

  // The position corresponding to `threadIdx.x = 0` in the block
  // (Same in all threads)
  uint64_t block_offset = start_position + blockIdx.x * BLOCK_SIZE;
  // The position before which the buffer in shared memory is correct
  // (Same in all threads)
  uint64_t buffer_correct = start_position;
  // The position corresponding to the current thread.
  uint64_t position = block_offset + threadIdx.x;

  for(unsigned i = 0; i < SHARED_BUFFER_SIZE; i += blockDim.x) {
    buffer[i + threadIdx.x] = global_buffer[i + threadIdx.x];
  }
  __syncthreads();

  while(*global_progress < debug_end ) { // A kernel has to exit for the profiler to work
  // while (true) {
    // A 256-bit array per thread for tracking the mex common value
    uint32_t mex_array[256 / 32];
    for (unsigned i = 0; i < 256 / 32; i += 1) {
      mex_array[i] = 0;
    }

    for(unsigned rare_idx = 0; rare_idx < RARE_VALUE_COUNT; rare_idx++)
    {
      uint16_t rare_pos = rare_positions[rare_idx];
      uint8_t rare_value = rare_values[rare_idx];

      // This iteration is going to be reading the values of the buffer in the
      // interval [block_offset - 1 - rare_pos, block_offset + BLOCK_SIZE - 1 -
      // rare_pos)

      if (rare_pos < BLOCK_SIZE * BLOCK_COUNT && rare_pos > BLOCK_SIZE) {
        // We are in the danger zone:

        // if (rare_pos < BLOCK_SIZE * BLOCK_COUNT), other blocks may
        // not have filled in the relevant values yet.
        // but later when (rare_pos <= BLOCK_SIZE), we are safe again,
        // because such values are handled within the current block.

        // Determine how much we need to copy from global memory,
        // and wait for it to be available.
        __shared__ uint64_t buffer_correct_shared;
        if (threadIdx.x == 0) {
          uint64_t buffer_correct_target = buffer_correct;

          while (buffer_correct_target <=
                 block_offset + (BLOCK_SIZE - 1) - 1 - rare_pos) {
            // This should be safe because all of these values are
            // updated `BLOCK_SIZE` at a time.
            while (*global_progress == buffer_correct_target) {
              __nanosleep(100); // TODO: necessary?
              __threadfence();  // TODO: necessary?
            }

            buffer_correct_target += BLOCK_SIZE;
          }
          buffer_correct_shared = buffer_correct_target;
        }
        __syncthreads();

        // Now copy that section in:
        while (buffer_correct < buffer_correct_shared) {
          buffer[(buffer_correct + threadIdx.x) % SHARED_BUFFER_SIZE] =
              global_buffer[(buffer_correct + threadIdx.x) %
                            GLOBAL_BUFFER_SIZE];
          buffer_correct += BLOCK_SIZE;
        }
        __syncthreads();
      }

      if(threadIdx.x < rare_pos+1) [[likely]] {
        // TODO: This is often an unaligned load
        // TODO: Can the compiler identify that this load can be
        // coalesced or is the expression too complicated?
        uint8_t prev = buffer[(position - 1 - rare_pos) % SHARED_BUFFER_SIZE];
        uint8_t option = prev ^ rare_value;
        set_bit_8(mex_array, option);
      }
    }

    for (unsigned done_idx = 0; done_idx < BLOCK_SIZE; done_idx++) {
      __shared__ uint8_t final_value;
      if (threadIdx.x == done_idx) {
        final_value = lowest_unset(mex_array);
        buffer[position % SHARED_BUFFER_SIZE] = final_value;
      }
      __syncthreads(); // Make sure value is written

      // Do the final, intra-block mex_array setting for positions later than this one
      const uint8_t local_final_value = final_value;
      if (threadIdx.x > done_idx) { // TODO: Could be omitted, no harm in setting bits of lower positions?
        uint8_t v = 0xFF;

        if (get_bit_4(low_rare0, threadIdx.x - done_idx - 1))
          v = compress_officers(0);
        else if (get_bit_4(low_rare1, threadIdx.x - done_idx - 1))
          v = compress_officers(1);
        else if (get_bit_4(low_rare6, threadIdx.x - done_idx - 1))
          v = compress_officers(6);

        if (v != 0xFF)
          set_bit_8(mex_array, local_final_value ^ v);
      }
      // __syncthreads(); // TODO: This shouldn't be necessary, but a previous version was giving incorrect results without it
    }
    
    // Copy results out
    global_buffer[position % GLOBAL_BUFFER_SIZE] = buffer[position % SHARED_BUFFER_SIZE];
    __threadfence();

    // Increment progress counter
    if (threadIdx.x == 0) {
      atomicAdd(global_progress, BLOCK_SIZE);
      // __threadfence_system();
    }

    // The values we just wrote are correct
    buffer_correct += BLOCK_SIZE;
    // Increment progress counter
    position += BLOCK_SIZE * BLOCK_COUNT;
    block_offset += BLOCK_SIZE * BLOCK_COUNT;
  }
}

std::vector<uint16_t> naive_officers(unsigned length) {
  std::vector<uint16_t> values;
  values.push_back(0); // G(0) = 0, special case because you can't remove a coin
  values.push_back(0); // G(1) = 0, special case because you can't split into 0 and 0

  for (uint64_t i = 2; i < length; i++) {
    std::vector<bool> seen(512, false);

    uint64_t midway = (i-1) >> 1;
    for (uint64_t j = 0; j <= midway; j++) {
      uint16_t value = values[j] ^ values[i - 1 - j];

      if(value >= seen.size()) // Won't ever happen in practice
        seen.resize(value + 1, false);

      seen[value] = true;
    }

    uint16_t mex = 0;
    while (mex < seen.size() && seen[mex]) {
      mex++;
    }
    values.push_back(mex);
  }
  return values;
}

void investigate(const std::vector<uint16_t> &values, uint64_t to_compute) {
  std::map<uint16_t, std::vector<std::pair<uint64_t, uint16_t>>> elims;

  uint64_t midway = (to_compute-1) >> 1;
  for (uint64_t j = 0; j <= midway; j++) {
    if(!is_rare(values[j])) continue;

    uint16_t value = values[j] ^ values[to_compute - 1 - j];

    elims[value].push_back({static_cast<uint64_t>(to_compute - 1 - j), values[to_compute - 1 - j]});
  }

  for (const auto& [key, values] : elims) {
    std::cout << key << " eliminated by ";
    for (const auto &[pos, val] : values) {
      std::cout << "(" << pos << ", " << val << ") ";
    }
    std::cout << std::endl;
  }
}

std::vector<uint16_t> cached_naive_officers(unsigned length, const std::string& filename) {
  // Try to read from file
  std::ifstream cache(filename);
  if (cache.good()) {
    size_t size;
    cache >> size;

    if (size >= length) {
      std::vector<uint16_t> result(size);
      for (size_t i = 0; i < size; ++i) {
        cache >> result[i];
      }

      if (cache.good()) {
        return result;
      }
    }
  }

  // Otherwise recompute
  std::vector<uint16_t> result = naive_officers(length);

  std::ofstream out(filename);
  out << result.size() << "\n";
  for (const auto& val : result) {
    out << val << " ";
  }

  return result;
}

void run_officers() {
  std::cout << "Sanity checking compression." << std::endl;
  for (uint16_t i = 0; i < (1 << 8); i++) {
    if(is_rare(i)) continue;
    uint8_t c = compress_officers(i);
    uint16_t u = uncompress_officers_common_host(c);
    if (i != u) {
      std::cout << "Failed on " << i << std::endl;
      exit(-1);
    }
  }

  std::cout << "Sanity checking ordering." << std::endl;
  for (uint16_t i = 0; i < 32; i++) {
    if (is_rare(i)) continue;
    for (uint16_t j = 0; j < 32; j++) {
      if (is_rare(j)) continue;
      uint8_t ci = compress_officers(i);
      uint8_t cj = compress_officers(j);
      if ((i < j) != (ci < cj)) {
        std::cout << "Failed on " << i << std::endl;
        std::cout << "Failed on " << j << std::endl;
        exit(-1);
      }
    }
  }

  // We start at 1<<16, because less than 1<<15 contains rare values
  // and so use the fast algorithm would fail. So we precompute 1<<15
  // to 1<<16 to fill the buffer, and then start the kernel at 1<<16
  
  // Progress counter
  unsigned long long *d_progress;
  unsigned long long *h_progress;
  cudaMallocHost(&h_progress, sizeof(unsigned long long));
  cudaHostGetDevicePointer(&d_progress, h_progress, 0);

  unsigned initial_length = (1<<16) + (1<<15);
  std::cout << "Generating/loading initial " << initial_length << " values." << std::endl;
  std::vector<uint16_t> initial = cached_naive_officers(initial_length, "officers_initial.txt");

  std::cout << "Filling rare value constants." << std::endl;
  {
    uint16_t h_rare_positions[RARE_VALUE_COUNT];
    uint8_t h_rare_values[RARE_VALUE_COUNT];

    uint64_t pos = 0;
    uint64_t idx = 0;
    for (const uint16_t val : initial) {
      if (is_rare(val)) {
        h_rare_positions[RARE_VALUE_COUNT - 1 - idx] = pos;
        h_rare_values[RARE_VALUE_COUNT - 1 - idx] = compress_officers(val);
        idx++;
      }
      pos++;
    }
    assert(idx == RARE_VALUE_COUNT);
    cudaMemcpyToSymbol(rare_positions, h_rare_positions, sizeof(h_rare_positions));
    cudaMemcpyToSymbol(rare_values, h_rare_values, sizeof(h_rare_values));
  }

  std::vector<bool> seen_values(512, false);
  {
    uint64_t pos = 0;
    for (const uint16_t val : initial) {
      if (!seen_values[val]) {
        // std::cout << "New value! G(" << pos << ") = " << val << std::endl;
        seen_values[val] = true;
      }
      pos++;
    }
  }

  std::cout << "Compressing the range " << (1 << 15) << " to " << (1<<16) << "." << std::endl;
  std::vector<uint16_t> shorter_vector(initial.begin() + (1 << 15), initial.begin() + (1 << 16));
  std::vector<uint8_t> compressed_vector;
  compressed_vector.reserve(shorter_vector.size());
  for (const uint16_t value : shorter_vector) {
    compressed_vector.push_back(compress_officers(value));
  }

  uint8_t *d_global_buffer;
  uint8_t *h_global_buffer;
  cudaMallocHost(&h_global_buffer, GLOBAL_BUFFER_SIZE * sizeof(uint8_t));
  cudaHostGetDevicePointer(&d_global_buffer, h_global_buffer, 0);
  cudaMemcpy(h_global_buffer, compressed_vector.data(),
             compressed_vector.size() * sizeof(uint8_t),
             cudaMemcpyHostToHost);

  *h_progress = 1u << 16;

  std::cout << "Launching." << std::endl;
  officers_kernel<<<BLOCK_COUNT, BLOCK_SIZE>>>(1u << 16, d_global_buffer, d_progress);

  unsigned long long reported_progress = *h_progress;

  while (*h_progress < debug_end) {
  // while (true) {
    while (reported_progress == *h_progress) {
      std::this_thread::sleep_for(std::chrono::microseconds(10000));

      cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(-1);
      }
    }
    while (reported_progress < *h_progress) {
      uint8_t news[BLOCK_SIZE] = {0};
      auto loc = reported_progress % GLOBAL_BUFFER_SIZE;
      memcpy(&news, h_global_buffer+loc, BLOCK_SIZE * sizeof(uint8_t));
      for (unsigned i = 0; i < BLOCK_SIZE; i++) {
        uint64_t pos = reported_progress + i;
        uint16_t val = uncompress_officers_common_host(news[i]);
        if (!seen_values[val]) {
          std::cout << "New value! G(" << pos << ") = " << val << std::endl;
          seen_values[val] = true;
        }
        // if(uncompress_officers_common_host(news[i]) != initial[pos]) {
        //   std::cout << "At: ";
        //   std::cout << pos;
        //   std::cout << " = (1<<16) + ";
        //   std::cout << pos - (1<<16);
        //   std::cout << " Got: ";
        //   std::cout << uncompress_officers_common_host(news[i]);
        //   std::cout << " Expected: ";
        //   std::cout << initial[pos];
        //   std::cout << std::endl;
        //   cudaDeviceSynchronize();
        //   exit(-1);
        // }
      }
      reported_progress += BLOCK_SIZE;
    }
  }
  cudaDeviceSynchronize();

  cudaFreeHost(h_global_buffer);
  cudaFreeHost(h_progress);
  exit(0);
}

int main() {
  run_officers();
}
