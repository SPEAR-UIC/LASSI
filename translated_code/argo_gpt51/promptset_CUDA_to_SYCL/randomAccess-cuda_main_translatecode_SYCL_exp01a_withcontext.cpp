
#include <CL/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace sycl_compat {
  namespace sycl = cl::sycl;
}

using namespace sycl_compat::sycl;

typedef unsigned long long int u64Int;
typedef long long int s64Int;

/* Random number generator */
#define POLY 0x0000000000000007UL
#define PERIOD 1317624576693539401L

#define K1_BLOCKSIZE 256
#define K2_BLOCKSIZE 128
#define K3_BLOCKSIZE 128

inline u64Int HPCC_starts_host(s64Int n) {
  int i, j;
  u64Int m2[64];
  u64Int temp, ran;

  while (n < 0) n += PERIOD;
  while (n > PERIOD) n -= PERIOD;
  if (n == 0) return 0x1;

  temp = 0x1;
  for (i = 0; i < 64; i++) {
    m2[i] = temp;
    temp = (temp << 1) ^ ((s64Int)temp < 0 ? POLY : 0);
    temp = (temp << 1) ^ ((s64Int)temp < 0 ? POLY : 0);
  }

  for (i = 62; i >= 0; i--)
    if ((n >> i) & 1)
      break;

  ran = 0x2;
  while (i > 0) {
    temp = 0;
    for (j = 0; j < 64; j++)
      if ((ran >> j) & 1)
        temp ^= m2[j];
    ran = temp;
    i -= 1;
    if ((n >> i) & 1)
      ran = (ran << 1) ^ ((s64Int)ran < 0 ? POLY : 0);
  }

  return ran;
}

// Device variant of HPCC_starts
inline u64Int HPCC_starts_dev(s64Int n) {
  int i, j;
  u64Int m2[64];
  u64Int temp, ran;

  while (n < 0) n += PERIOD;
  while (n > PERIOD) n -= PERIOD;
  if (n == 0) return 0x1;

  temp = 0x1;
  for (i = 0; i < 64; i++) {
    m2[i] = temp;
    temp = (temp << 1) ^ ((s64Int)temp < 0 ? POLY : 0);
    temp = (temp << 1) ^ ((s64Int)temp < 0 ? POLY : 0);
  }

  for (i = 62; i >= 0; i--)
    if ((n >> i) & 1)
      break;

  ran = 0x2;
  while (i > 0) {
    temp = 0;
    for (j = 0; j < 64; j++)
      if ((ran >> j) & 1)
        temp ^= m2[j];
    ran = temp;
    i -= 1;
    if ((n >> i) & 1)
      ran = (ran << 1) ^ ((s64Int)ran < 0 ? POLY : 0);
  }

  return ran;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = std::atoi(argv[1]);
  int failure;
  u64Int i;
  u64Int temp;
  double totalMem;
  u64Int *Table = nullptr;
  u64Int logTableSize, TableSize;

  /* calculate local memory per node for the update table */
  totalMem = 1024.0 * 1024.0 * 512.0;
  totalMem /= sizeof(u64Int);

  /* calculate the size of update array (must be a power of 2) */
  for (totalMem *= 0.5, logTableSize = 0, TableSize = 1;
       totalMem >= 1.0;
       totalMem *= 0.5, logTableSize++, TableSize <<= 1)
    ;

  std::printf("Table size = %llu\n", (unsigned long long)TableSize);

  if (posix_memalign((void **)&Table, 1024, TableSize * sizeof(u64Int)) != 0) {
    std::fprintf(stderr,
                 "Failed to allocate memory for the update table %llu\n",
                 (unsigned long long)TableSize);
    return 1;
  }

  /* Print parameters for run */
  std::fprintf(stdout, "Main table size = 2^%llu = %llu words\n",
               (unsigned long long)logTableSize,
               (unsigned long long)TableSize);

  const u64Int NUPDATE = 4 * TableSize;
  std::fprintf(stdout, "Number of updates = %llu\n",
               (unsigned long long)NUPDATE);

  // SYCL queue (in-order to resemble CUDA stream behavior)
  cl::sycl::queue q{cl::sycl::property::queue::in_order{}};

  // Device allocations
  u64Int *d_Table = cl::sycl::malloc_device<u64Int>(TableSize, q);
  u64Int *d_ran   = cl::sycl::malloc_device<u64Int>(128, q);

  if (!d_Table || !d_ran) {
    std::fprintf(stderr, "Failed to allocate device memory\n");
    free(Table);
    if (d_Table) cl::sycl::free(d_Table, q);
    if (d_ran) cl::sycl::free(d_ran, q);
    return 1;
  }

  auto start = std::chrono::steady_clock::now();

  for (int rep = 0; rep < repeat; rep++) {
    // initTable
    {
      u64Int gridSize =
          (TableSize + K1_BLOCKSIZE - 1) / K1_BLOCKSIZE;

      cl::sycl::range<1> global(gridSize * K1_BLOCKSIZE);
      cl::sycl::range<1> local(K1_BLOCKSIZE);

      q.submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(
          cl::sycl::nd_range<1>(global, local),
          [=](cl::sycl::nd_item<1> item) {
            u64Int idx = item.get_global_id(0);
            if (idx < TableSize) {
              d_Table[idx] = idx;
            }
          });
      });
    }

    // initRan
    {
      cl::sycl::range<1> global(K2_BLOCKSIZE);
      cl::sycl::range<1> local(K2_BLOCKSIZE);

      q.submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(
          cl::sycl::nd_range<1>(global, local),
          [=](cl::sycl::nd_item<1> item) {
            int j = (int)item.get_local_id(0);
            if (j < 128) {
              // NUPDATE/128 is integral
              u64Int perThread = NUPDATE / 128;
              d_ran[j] = HPCC_starts_dev((s64Int)(perThread * j));
            }
          });
      });
    }

    // update
    {
      cl::sycl::range<1> global(K3_BLOCKSIZE);
      cl::sycl::range<1> local(K3_BLOCKSIZE);

      q.submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(
          cl::sycl::nd_range<1>(global, local),
          [=](cl::sycl::nd_item<1> item) {
            int j = (int)item.get_local_id(0);
            if (j >= 128) return;

            u64Int perThread = NUPDATE / 128;
            u64Int ran = d_ran[j];

            for (u64Int k = 0; k < perThread; k++) {
              ran = (ran << 1) ^ ((s64Int)ran < 0 ? POLY : 0);
              u64Int index = ran & (TableSize - 1);

              auto aref =
                  cl::sycl::atomic_ref<
                      u64Int,
                      cl::sycl::memory_order::relaxed,
                      cl::sycl::memory_scope::device,
                      cl::sycl::access::address_space::global_space>(
                          d_Table[index]);
              aref.fetch_xor(ran);
            }
            d_ran[j] = ran;
          });
      });
    }
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
          .count();

  std::printf("Average kernel execution time: %f (s)\n",
              (time * 1e-9f) / repeat);

  // Copy back result
  q.memcpy(Table, d_Table, TableSize * sizeof(u64Int)).wait();

  /* validation */
  temp = 0x1;
  for (i = 0; i < NUPDATE; i++) {
    temp = (temp << 1) ^ (((s64Int)temp < 0) ? POLY : 0);
    Table[temp & (TableSize - 1)] ^= temp;
  }

  temp = 0;
  for (i = 0; i < TableSize; i++)
    if (Table[i] != i) {
      temp++;
    }

  std::fprintf(stdout,
               "Found %llu errors in %llu locations (%s).\n",
               (unsigned long long)temp,
               (unsigned long long)TableSize,
               (temp <= (u64Int)(0.01 * (double)TableSize)) ? "PASS"
                                                            : "FAIL");

  if (temp <= (u64Int)(0.01 * (double)TableSize))
    failure = 0;
  else
    failure = 1;

  free(Table);
  cl::sycl::free(d_Table, q);
  cl::sycl::free(d_ran, q);

  return failure;
}
