#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <assert.h>
#include <omp.h>

#define BLOCK_SIZE 256

// measure cost of additions without atomics
template <typename T>
void woAtomicOnGlobalMem(T* result, int size, int n)
{
  #pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
  for (unsigned int tid = 0; tid < n; tid++) {
    for ( unsigned int i = tid * size; i < (tid + 1) * size; i++) {
      result[tid] += i % 2;
    }
  }
}

// measure cost of additions with atomics
template <typename T>
void wiAtomicOnGlobalMem(T* result, int size, int n)
{
  #pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
  for (unsigned int tid = 0; tid < n; tid++) {
    #pragma omp atomic update
    result[tid/size] += tid % 2;
  }
}

template <typename T>
void memset(T* result, int n)
{
  #pragma omp target teams distribute parallel for
  for (unsigned int tid = 0; tid < n; tid++) {
    result[tid] = 0;
  }
}

template <typename T>
void atomicCost (int t, int repeat)
{
  for (int size = 1; size <= 16; size++) {

    printf("\n\n");
    printf("Each thread sums up %d elements\n", size);

    assert(t % size == 0);
    assert(t / size % BLOCK_SIZE == 0);

    size_t result_size = sizeof(T) * t / size;

    T* result_wi = (T*) malloc (result_size);
    T* result_wo = (T*) malloc (result_size);

    #pragma omp target data map(alloc: result_wi[0:t/size], result_wo[0:t/size])
    {
      auto start = std::chrono::steady_clock::now();
      for(int i=0; i<repeat; i++)
      {
        memset(result_wi, 0, t/size);
        wiAtomicOnGlobalMem<T>(result_wi, size, t);
      }
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of WithAtomicOnGlobalMem: %f (us)\n",
              time * 1e-3f / repeat);
      #pragma omp target update from (result_wi[0:t/size])

      start = std::chrono::steady_clock::now();
      for(int i=0; i<repeat; i++)
      {
        memset(result_wo, 0, t/size);
        woAtomicOnGlobalMem<T>(result_wo, size, t/size);
      }
      end = std::chrono::steady_clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of WithoutAtomicOnGlobalMem: %f (us)\n",
              time * 1e-3f / repeat);
      #pragma omp target update from (result_wo[0:t/size])

      int diff = memcmp(result_wi, result_wo, result_size);
      printf("%s\n", diff ? "FAIL" : "PASS"); 
    }

    free(result_wi);
    free(result_wo);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int t = 922521600;
  assert(t % BLOCK_SIZE == 0);
  
  printf("\nFP64 atomic add\n");
  atomicCost<double>(t, repeat); 

  printf("\nINT32 atomic add\n");
  atomicCost<int>(t, repeat); 

  printf("\nFP32 atomic add\n");
  atomicCost<float>(t, repeat); 

  return 0;
}
