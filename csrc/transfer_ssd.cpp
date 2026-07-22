#include <errno.h>
#include <fcntl.h>
#include <torch/extension.h>
#include <unistd.h>
#include <vector>

#include <future>
#include <mutex>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

#include "transfer_ssd.h"
#include "monitoring/metrics_manager.h"

namespace flexkv {

static void partition_and_remap_blocks_by_device(
    const int64_t *cpu_block_ids, const int64_t *ssd_block_ids, int num_blocks,
    int num_devices, int round_robin,
    std::vector<std::vector<int>> &cpu_blocks_partition,
    std::vector<std::vector<int>> &ssd_blocks_partition) {
  for (int i = 0; i < num_blocks; i++) {
    int64_t ssd_block_id = ssd_block_ids[i];
    int64_t cpu_block_id = cpu_block_ids[i];
    int device_id = (ssd_block_id / round_robin) % num_devices;
    int block_id_in_device =
        ((ssd_block_id / round_robin) / num_devices) * round_robin +
        (ssd_block_id % round_robin);
    ssd_blocks_partition[device_id].push_back(block_id_in_device);
    cpu_blocks_partition[device_id].push_back(cpu_block_id);
  }
}

// ---- Unified transfer: handles both block-first and layer-first I/O ----
// ---- The actual I/O operation (pread/pwrite vs io_uring) is abstracted  ----
// ---- via an IOCallable so the iteration logic is shared by both paths. ----
using IOCallable = std::function<void(int fd, void *cpu_ptr, int64_t ssd_offset,
                                      int64_t size, bool is_read)>;

static void transfer_blocks_impl(
    const std::vector<int> &fd_list, const std::vector<int> &cpu_block_ids,
    const std::vector<int> &ssd_block_ids_in_device, int start_layer,
    int end_layer, int start_block, int end_block, int64_t cpu_tensor_ptr,
    int64_t cpu_layer_stride_in_bytes, int64_t ssd_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes, int64_t ssd_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes, int64_t block_stride_in_bytes,
    int num_files_per_device, bool is_read, bool is_mla,
    bool enable_block_first_transfer, IOCallable &do_io) {
  if (end_block <= start_block) return;

  // Block-first: single large I/O per block (all layers contiguous)
  if (enable_block_first_transfer) {
    int64_t layers_size = cpu_layer_stride_in_bytes * (end_layer - start_layer);
    for (int bid = start_block; bid < end_block; bid++) {
      int cpu_block_id = cpu_block_ids[bid];
      int ssd_block_id = ssd_block_ids_in_device[bid];
      int fd = fd_list[ssd_block_id % num_files_per_device];
      ssd_block_id /= num_files_per_device;
      void *cpu_ptr = reinterpret_cast<char *>(cpu_tensor_ptr) +
                      block_stride_in_bytes * cpu_block_id +
                      start_layer * cpu_layer_stride_in_bytes;
      int64_t ssd_off = ssd_block_id * block_stride_in_bytes +
                        start_layer * ssd_layer_stride_in_bytes;
      do_io(fd, cpu_ptr, ssd_off, layers_size, is_read);
      FLEXKV_CPU_SSD_TRANSFER(is_read, layers_size);
    }
    return;
  }

  // Layer-first: check if blocks are contiguous for layer-major batch I/O
  bool blocks_contiguous = true;
  for (int bid = start_block; bid < end_block - 1; bid++) {
    if (cpu_block_ids[bid + 1] != cpu_block_ids[bid] + 1 ||
        ssd_block_ids_in_device[bid + 1] != ssd_block_ids_in_device[bid] + 1) {
      blocks_contiguous = false;
      break;
    }
  }

  if (blocks_contiguous) {
    // Layer-major: one large I/O per layer (all blocks contiguous)
    int num_blocks = end_block - start_block;
    int64_t batch_size = block_stride_in_bytes * num_blocks;
    int cpu_bid = cpu_block_ids[start_block];
    int ssd_bid = ssd_block_ids_in_device[start_block];
    int fd = fd_list[ssd_bid % num_files_per_device];
    ssd_bid /= num_files_per_device;
    for (int lid = start_layer; lid < end_layer; lid++) {
      void *cpu_ptr = reinterpret_cast<char *>(cpu_tensor_ptr) +
                      block_stride_in_bytes * cpu_bid +
                      lid * cpu_layer_stride_in_bytes;
      int64_t ssd_off = ssd_bid * block_stride_in_bytes +
                        lid * ssd_layer_stride_in_bytes;
      do_io(fd, cpu_ptr, ssd_off, batch_size, is_read);
      FLEXKV_CPU_SSD_TRANSFER(is_read, batch_size);
    }
    return;
  }

  // Layer-first fragmented: per-layer, per-K/V I/O for non-contiguous blocks
  for (int bid = start_block; bid < end_block; bid++) {
    int cpu_block_id = cpu_block_ids[bid];
    int ssd_block_id = ssd_block_ids_in_device[bid];
    int fd = fd_list[ssd_block_id % num_files_per_device];
    ssd_block_id /= num_files_per_device;
    for (int lid = start_layer; lid < end_layer; lid++) {
      void *cpu_k_ptr = reinterpret_cast<char *>(cpu_tensor_ptr) +
                        block_stride_in_bytes * cpu_block_id +
                        lid * cpu_layer_stride_in_bytes;
      int64_t ssd_k_off = ssd_block_id * block_stride_in_bytes +
                          lid * ssd_layer_stride_in_bytes;
      do_io(fd, cpu_k_ptr, ssd_k_off, chunk_size_in_bytes, is_read);
      FLEXKV_CPU_SSD_TRANSFER(is_read, chunk_size_in_bytes);
      if (is_mla) continue;
      void *cpu_v_ptr = cpu_k_ptr + cpu_kv_stride_in_bytes;
      int64_t ssd_v_off = ssd_k_off + ssd_kv_stride_in_bytes;
      do_io(fd, cpu_v_ptr, ssd_v_off, chunk_size_in_bytes, is_read);
      FLEXKV_CPU_SSD_TRANSFER(is_read, chunk_size_in_bytes);
    }
  }
}

// NOTE that we may also use other techniques such as
// AIO, O_DIRECT, and etc to improve the performance
void transfer_kv_blocks_ssd(
    SSDIOCTX &ioctx, const torch::Tensor &cpu_layer_id_list,
    int64_t cpu_tensor_ptr, const torch::Tensor &ssd_block_ids,
    const torch::Tensor &cpu_block_ids, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes,
    int64_t ssd_layer_stride_in_bytes, // in single file
    int64_t ssd_kv_stride_in_bytes,    // in single file
    int64_t chunk_size_in_bytes, int64_t block_stride_in_bytes, bool is_read,
    int num_blocks_per_file, int round_robin, int num_threads_per_device,
    bool is_mla) {
  const int num_devices = ioctx.get_num_devices();
  const int num_files_per_device = ioctx.get_num_files_per_device();

  const int64_t *ssd_block_id_ptr = ssd_block_ids.data_ptr<int64_t>();
  const int64_t *cpu_block_id_ptr = cpu_block_ids.data_ptr<int64_t>();

  const int num_blocks = ssd_block_ids.size(0);
  const int num_layers = cpu_layer_id_list.size(0);
  const int32_t *cpu_layer_id_list_ptr = cpu_layer_id_list.data_ptr<int32_t>();
  bool is_direct = chunk_size_in_bytes % 4096 == 0;

  IOUring &iouring = ioctx.get_iouring();
  std::vector<std::vector<int>> &fds = ioctx.get_fds(is_read, is_direct);

  std::vector<std::vector<int>> cpu_blocks_partition(num_devices,
                                                     std::vector<int>());
  std::vector<std::vector<int>> ssd_blocks_partition(num_devices,
                                                     std::vector<int>());
  partition_and_remap_blocks_by_device(
      cpu_block_id_ptr, ssd_block_id_ptr, num_blocks, num_devices, round_robin,
      cpu_blocks_partition, ssd_blocks_partition);

  const bool cpu_is_block_first =
      block_stride_in_bytes > cpu_layer_stride_in_bytes;
  const bool ssd_is_block_first =
      block_stride_in_bytes > ssd_layer_stride_in_bytes;
  const bool enable_block_first_transfer =
      cpu_is_block_first && ssd_is_block_first;

  std::vector<std::thread> threads;
  std::vector<std::future<std::exception_ptr>> futures;
  for (int t = 0; t < num_threads_per_device; t++) {
    for (int d = 0; d < num_devices; d++) {
      int start_layer = cpu_layer_id_list_ptr[0];
      int end_layer = cpu_layer_id_list_ptr[0] + num_layers;
      int num_transfer_blocks = cpu_blocks_partition[d].size();
      int num_blocks_per_thread =
          (num_transfer_blocks + num_threads_per_device - 1) /
          num_threads_per_device;
      int start_block = t * num_blocks_per_thread;
      int end_block =
          std::min(start_block + num_blocks_per_thread, num_transfer_blocks);
      if (start_block < end_block) {
        if (iouring.enabled()) {
          IOCallable do_io = [&](int fd, void *cpu_ptr, int64_t ssd_offset,
                                 int64_t size, bool is_read) {
            int rc;
            if (is_read) {
              rc = iouring.prep_read(fd, cpu_ptr, size, ssd_offset);
              if (rc < 0) {
                ssize_t bytes_transfer = pread(fd, cpu_ptr, size, ssd_offset);
                if (bytes_transfer != size) {
                  throw std::runtime_error("Failed to transfer block");
                }
              }
            } else {
              rc = iouring.prep_write(fd, cpu_ptr, size, ssd_offset);
              if (rc < 0) {
                ssize_t bytes_transfer = pwrite(fd, cpu_ptr, size, ssd_offset);
                if (bytes_transfer != size) {
                  throw std::runtime_error("Failed to transfer block");
                }
              }
            }
          };
          transfer_blocks_impl(
              fds[d], cpu_blocks_partition[d], ssd_blocks_partition[d],
              start_layer, end_layer, start_block, end_block, cpu_tensor_ptr,
              cpu_layer_stride_in_bytes, ssd_layer_stride_in_bytes,
              cpu_kv_stride_in_bytes, ssd_kv_stride_in_bytes,
              chunk_size_in_bytes, block_stride_in_bytes, num_files_per_device,
              is_read, is_mla, enable_block_first_transfer, do_io);
          continue;
        }

        std::promise<std::exception_ptr> prom;
        futures.push_back(prom.get_future());
        threads.emplace_back(
            [d, &fds, &cpu_blocks_partition, &ssd_blocks_partition, start_layer,
             end_layer, start_block, end_block, cpu_tensor_ptr,
             cpu_layer_stride_in_bytes, ssd_layer_stride_in_bytes,
             cpu_kv_stride_in_bytes, ssd_kv_stride_in_bytes,
             chunk_size_in_bytes, block_stride_in_bytes, num_files_per_device,
             is_read, is_mla, enable_block_first_transfer,
             prom = std::move(prom)]() mutable {
              try {
                IOCallable do_io = [&](int fd, void *cpu_ptr, int64_t ssd_offset,
                                       int64_t size, bool is_read) {
                  ssize_t bytes_transfer;
                  if (is_read) {
                    bytes_transfer =
                        pread(fd, cpu_ptr, size, ssd_offset);
                  } else {
                    bytes_transfer = pwrite(fd, cpu_ptr, size, ssd_offset);
                  }
                  if (bytes_transfer != size) {
                    throw std::runtime_error("Failed to transfer block");
                  }
                };
                transfer_blocks_impl(
                    fds[d], cpu_blocks_partition[d], ssd_blocks_partition[d],
                    start_layer, end_layer, start_block, end_block,
                    cpu_tensor_ptr, cpu_layer_stride_in_bytes,
                    ssd_layer_stride_in_bytes, cpu_kv_stride_in_bytes,
                    ssd_kv_stride_in_bytes, chunk_size_in_bytes,
                    block_stride_in_bytes, num_files_per_device, is_read,
                    is_mla, enable_block_first_transfer, do_io);
                prom.set_value(nullptr);
              } catch (...) {
                prom.set_value(std::current_exception());
              }
            });
      }
    } // end device loop
  } // end thread loop

  if (iouring.enabled()) {
    if (iouring.wait_completion()) {
      throw std::runtime_error("Failed to transfer data");
    }
  } else {
    // wait for all threads to finish
    for (auto &thread : threads) {
      thread.join();
    }

    // check if any error occurs
    for (auto &fut : futures) {
      if (auto eptr = fut.get()) {
        std::rethrow_exception(eptr);
      }
    }
  }
}

} // namespace flexkv