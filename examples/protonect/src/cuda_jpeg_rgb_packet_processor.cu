/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2014 individual OpenKinect contributors. See the CONTRIB file
 * for details.
 *
 * This code is licensed to you under the terms of the Apache License, version
 * 2.0, or, at your option, the terms of the GNU General Public License,
 * version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
 * or the following URLs:
 * http://www.apache.org/licenses/LICENSE-2.0
 * http://www.gnu.org/licenses/gpl-2.0.txt
 *
 * If you redistribute this file in source form, modified or unmodified, you
 * may:
 *   1) Leave this header intact and distribute it under the same terms,
 *      accompanying it with the APACHE20 and GPL20 files, or
 *   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
 *   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
 * In all cases you must keep the copyright notice intact and include a copy
 * of the CONTRIB file.
 *
 * Binary distributions must follow the binary distribution requirements of
 * either License.
 */

#include <libfreenect2/rgb_packet_processor.h>

#include <opencv2/opencv.hpp>
#include <npp.h>
#include <cuda_runtime.h>
#include <jpeglib.h>
#include <stdexcept>

static inline NppStatus nppSafeCall(NppStatus err)
{
  if (err != NPP_SUCCESS) {
    std::cerr << "NPP error " << err << std::endl;
    throw std::runtime_error("NPP error");
  }
  return err;
}

static inline cudaError_t cudaSafeCall(cudaError_t err)
{
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
    abort();//throw std::runtime_error(cudaGetErrorString(err));
  }
  return err;
}

static const int WIDTH = 1920;
static const int HEIGHT = 1080;
static const int COMPS = 3;
static const int RESTART_INTERVAL = 50;
static const int MCU_WIDTH = 16;
//static const int MCU_HEIGHT = 8;
static const int SEGMENTS = 324;//(WIDTH/MCU_WIDTH * HEIGHT/MCU_HEIGHT)/RESTART_INTERVAL
static const int THREADS_PER_TBLOCK = 192;

/* Original code from GPUJPEG
  - src/gpujpeg_huffman_gpu_decoder.cu
  - src/gpujpeg_table.cpp
 */

/**
 * Copyright (c) 2011, CESNET z.s.p.o
 * Copyright (c) 2011, Silicon Genome, LLC.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * 4 pre-built tables for faster Huffman decoding (codewords up-to 16 bit length):
 *   0x00000 to 0x0ffff: luminance DC table
 *   0x10000 to 0x1ffff: luminance AC table
 *   0x20000 to 0x2ffff: chrominance DC table
 *   0x30000 to 0x3ffff: chrominance AC table
 *
 * Each entry consists of:
 *   - Number of bits of code corresponding to this entry (0 - 16, both inclusive) - bits 4 to 8
 *   - Number of run-length coded zeros before currently decoded coefficient + 1 (1 - 64, both inclusive) - bits 9 to 15
 *   - Number of bits representing the value of currently decoded coefficient (0 - 15, both inclusive) - bits 0 to 3
 * bit #:    15                      9   8               4   3           0
 *         +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
 * value:  |      RLE zero count       |   code bit size   | value bit size|
 *         +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
 */
__device__ uint16_t gpujpeg_huffman_gpu_decoder_tables_full[4 * (1 << 16)];

/** Number of code bits to be checked first (with high chance for the code to fit into this number of bits). */
#define QUICK_CHECK_BITS 4
#define QUICK_TABLE_ITEMS (4 * (1 << QUICK_CHECK_BITS))
// TODO: try to tweak QUICK table size and memory space

/** Same table as above, but copied into constant memory */
__constant__ uint16_t gpujpeg_huffman_gpu_decoder_tables_quick_const[QUICK_TABLE_ITEMS];

/**
 * Loads at least specified number of bits into the register
 */
__device__ inline void
gpujpeg_huffman_gpu_decoder_load_bits(
                const unsigned int required_bit_count, unsigned int & r_bit,
                unsigned int & r_bit_count, uint4 * const s_byte, unsigned int & s_byte_idx
) {
    // Add bytes until have enough
    while(r_bit_count < required_bit_count) {
        // Load byte value and posibly skip next stuffed byte if loaded byte's value is 0xFF
        const uint8_t byte_value = ((const uint8_t*)s_byte)[s_byte_idx++];
        if((uint8_t)0xFF == byte_value) {
            s_byte_idx++;
        }

        // Add newly loaded byte to the buffer, updating bit count
        r_bit = (r_bit << 8) + byte_value;
        r_bit_count += 8;
    }
}


/**
 * Get bits
 *
 * @param nbits  Number of bits to get
 * @param get_bits
 * @param get_buff
 * @param data
 * @param data_size
 * @return bits
 */
__device__ inline unsigned int
gpujpeg_huffman_gpu_decoder_get_bits(
                const unsigned int nbits, unsigned int & r_bit, unsigned int & r_bit_count,
                uint4 * const s_byte, unsigned int & s_byte_idx)
{
    // load bits into the register if haven't got enough
    gpujpeg_huffman_gpu_decoder_load_bits(nbits, r_bit, r_bit_count, s_byte, s_byte_idx);

    // update remaining bit count
    r_bit_count -= nbits;

    // return bits
    return (r_bit >> r_bit_count) & ((1 << nbits) - 1);
}

/**
 * Gets bits without removing them from the buffer.
 */
__device__ inline unsigned int
gpujpeg_huffman_gpu_decoder_peek_bits(
                const unsigned int nbits, unsigned int & r_bit, unsigned int & r_bit_count,
                uint4 * const s_byte, unsigned int & s_byte_idx)
{
    // load bits into the register if haven't got enough
    gpujpeg_huffman_gpu_decoder_load_bits(nbits, r_bit, r_bit_count, s_byte, s_byte_idx);

    // return bits
    return (r_bit >> (r_bit_count - nbits)) & ((1 << nbits) - 1);
}

/**
 * Removes some bits from the buffer (assumes that they are there)
 */
__device__ inline void
gpujpeg_huffman_gpu_decoder_discard_bits(const unsigned int nb, unsigned int, unsigned int & r_bit_count) {
    r_bit_count -= nb;
}


/**
 * To find dc or ac value according to code and its bit length s
 */
__device__ inline int
gpujpeg_huffman_gpu_decoder_value_from_category(int nbits, int code)
{
    // TODO: try to replace with __constant__ table lookup
    return code < ((1 << nbits) >> 1) ? (code + ((-1) << nbits) + 1) : code;
}

/**
 * Decodes next coefficient, updating its output index
 *
 * @param table
 * @param get_bits
 * @param get_buff
 * @param data
 * @param data_size
 * @return int
 */
__device__ inline int
gpujpeg_huffman_gpu_decoder_get_coefficient(
                unsigned int & r_bit, unsigned int & r_bit_count, uint4* const s_byte,
                unsigned int & s_byte_idx, const unsigned int table_offset, unsigned int & coefficient_idx)
{
    // Peek next 16 bits and use them as an index into decoder table to find all the info.
    const unsigned int table_idx = table_offset + gpujpeg_huffman_gpu_decoder_peek_bits(16, r_bit, r_bit_count, s_byte, s_byte_idx);

    // Try the quick table first (use the full table only if not succeded with the quick table)
    unsigned int packed_info = gpujpeg_huffman_gpu_decoder_tables_quick_const[table_idx >> (16 - QUICK_CHECK_BITS)];
    if(0 == packed_info) {
        packed_info = gpujpeg_huffman_gpu_decoder_tables_full[table_idx];
    }

    // remove the right number of bits from the bit buffer
    gpujpeg_huffman_gpu_decoder_discard_bits((packed_info >> 4) & 0x1F, r_bit, r_bit_count);

    // update coefficient index by skipping run-length encoded zeros
    coefficient_idx += packed_info >> 9;

    // read coefficient bits and decode the coefficient from them
    const unsigned int value_nbits = packed_info & 0xF;
    const unsigned int value_code = gpujpeg_huffman_gpu_decoder_get_bits(value_nbits, r_bit, r_bit_count, s_byte, s_byte_idx);

    // return deocded coefficient
    return gpujpeg_huffman_gpu_decoder_value_from_category(value_nbits, value_code);
}

/**
 * Decode one 8x8 block
 *
 * @return 0 if succeeds, otherwise nonzero
 */
__device__ inline int
gpujpeg_huffman_gpu_decoder_decode_block(
    int & dc, int16_t* const data_output, const unsigned int table_offset,
    unsigned int & r_bit, unsigned int & r_bit_count, uint4* const s_byte,
    unsigned int & s_byte_idx, const uint4* & d_byte, unsigned int & d_byte_chunk_count)
{
    // TODO: try unified decoding of DC/AC coefficients

    // Index of next coefficient to be decoded (in ZIG-ZAG order)
    unsigned int coefficient_idx = 0;

    // Section F.2.2.1: decode the DC coefficient difference
    // Get the coefficient value (using DC coding table)
    int dc_coefficient_value = gpujpeg_huffman_gpu_decoder_get_coefficient(r_bit, r_bit_count, s_byte, s_byte_idx, table_offset, coefficient_idx);

    // Convert DC difference to actual value, update last_dc_val
    dc = dc_coefficient_value += dc;

    // Output the DC coefficient (assumes gpujpeg_natural_order[0] = 0)
    // TODO: try to skip saving of zero coefficients
    data_output[0] = dc_coefficient_value;

    // TODO: error check: coefficient_idx must still be 0 in valid codestreams
    coefficient_idx = 1;

    // Section F.2.2.2: decode the AC coefficients
    // Since zeroes are skipped, output area must be cleared beforehand
    do {
        // Possibly load more bytes into shared buffer from global memory
        if(s_byte_idx >= 16) {
            // Move remaining bytes to begin and update index of next byte
            s_byte[0] = s_byte[1];
            s_byte_idx -= 16;

            // Load another byte chunk from global memory only if there is one
            if(d_byte_chunk_count) {
                s_byte[1] = *(d_byte++);
                d_byte_chunk_count--;
            }
        }

        // decode next coefficient, updating its destination index
        const int coefficient_value = gpujpeg_huffman_gpu_decoder_get_coefficient(r_bit, r_bit_count, s_byte, s_byte_idx, table_offset + 0x10000, coefficient_idx);

        // stop with this block if have all coefficients
        if(coefficient_idx > 64) {
            break;
        }

        // save the coefficient   TODO: try to ommit saving 0 coefficients
        //NPP iDCT expects zig-zag order
        data_output[coefficient_idx - 1] = coefficient_value;
    } while(coefficient_idx < 64);

    return 0;
}


/**
 * Huffman decoder kernel
 *
 * @return void
 */
__global__ void
gpujpeg_huffman_decoder_decode_kernel(
  const uint2 *d_segment,
  const uint8_t* d_data_compressed,
  int16_t *d_dct_y, int16_t *d_dct_cb, int16_t *d_dct_cr,
  const size_t dct_step_y, const size_t dct_step_cb, const size_t dct_step_cr
) {
  int segment_index = blockIdx.x * THREADS_PER_TBLOCK + threadIdx.x;
  if (segment_index >= SEGMENTS)
    return;

  const unsigned int dct_step[3] = {dct_step_y, dct_step_cb, dct_step_cr};
  int16_t *const d_dct[3] = {d_dct_y, d_dct_cb, d_dct_cr};

  const uint2 *segment = &d_segment[segment_index];

  // Byte buffers in shared memory
  __shared__ uint4 s_byte_all[2 * THREADS_PER_TBLOCK]; // 32 bytes per thread
  uint4 * const s_byte = s_byte_all + 2 * threadIdx.x;

    // Last DC coefficient values   TODO: try to move into shared memory
  int dc[COMPS] = {0};

  // Get aligned compressed data chunk pointer and load first 2 chunks of the data
  const unsigned int d_byte_begin_idx = segment->x;
  const unsigned int d_byte_begin_idx_aligned = d_byte_begin_idx & ~15; // loading 16byte chunks
  const uint4* d_byte = (uint4*)(d_data_compressed + d_byte_begin_idx_aligned);

  // Get number of remaining global memory byte chunks (not to read bytes out of buffer)
  const unsigned int d_byte_end_idx_aligned = (segment->y + 15) & ~15;
  unsigned int d_byte_chunk_count = (d_byte_end_idx_aligned - d_byte_begin_idx_aligned) / 16;

  // Load first 2 chunks of compressed data into the shared memory buffer and remember index of first code byte (skipping bytes read due to alignment)
  s_byte[0] = d_byte[0];
  s_byte[1] = d_byte[1];
  d_byte += 2;
  d_byte_chunk_count = max(d_byte_chunk_count, 2) - 2;
  unsigned int s_byte_idx = d_byte_begin_idx - d_byte_begin_idx_aligned;

  // bits loaded into the register and their count
  unsigned int r_bit_count = 0;
  unsigned int r_bit = 0; // LSB-aligned

  for (int u = 0; u < RESTART_INTERVAL; u++) {
    const int mcu_index = segment_index * RESTART_INTERVAL + u;
    const int mcu_y = mcu_index / (WIDTH / MCU_WIDTH);
    const int mcu_x = mcu_index % (WIDTH / MCU_WIDTH);

    const int comp[4] = {0, 0, 1, 2};
    const int samp[4] = {2, 2, 1, 1};
    const int xoffset[4] = {0, 1, 0, 0};
    for (int b = 0; b < 4; b++) {
      const int c = comp[b];
      int16_t *block = d_dct[c] + dct_step[c] / sizeof(int16_t) * mcu_y + DCTSIZE2 * (samp[b] * mcu_x + xoffset[b]);
      const unsigned int huff_offset = (c ? 0x20000 : 0);
      gpujpeg_huffman_gpu_decoder_decode_block(dc[c], block, huff_offset, r_bit, r_bit_count, s_byte, s_byte_idx, d_byte, d_byte_chunk_count);
    }
  }
}

namespace libfreenect2
{

class CudaJpegRgbPacketProcessorImpl
{
public:
  struct jpeg_decompress_struct dinfo;
  struct jpeg_error_mgr jerr;

  NppiDCTState *dct_state;
  Npp8u *d_quant_tables;
  NppiSize src_size[COMPS];
  Npp16s *d_dct[COMPS];
  Npp32s dct_step[COMPS];
  Npp8u *src_image[COMPS];
  Npp32s src_image_step[COMPS];
  Npp8u *packed_image;
  Npp32s packed_image_step;

  uint16_t *huffman_tables_full;
  uint16_t *huffman_tables_quick;
  uint8_t *d_jpeg_buf;
  static const int d_jpeg_buf_size = 2*1024*1024;

  uint2 *mcu_segments;
  uint2 *d_mcu_segments;
  unsigned char *quant_tables;
  bool huffman_tables_loaded;

  Frame *frame;

  double timing_acc;
  double timing_acc_n;

  double timing_current_start;

  CudaJpegRgbPacketProcessorImpl()
  {
    dinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&dinfo);

    initializeNpp();

    newFrame();

    timing_acc = 0.0;
    timing_acc_n = 0.0;
    timing_current_start = 0.0;
  }

  ~CudaJpegRgbPacketProcessorImpl()
  {
    delete[] quant_tables;
    delete[] mcu_segments;
    delete[] huffman_tables_quick;
    delete[] huffman_tables_full;
    cudaFree(packed_image);
    for (int i = 0; i < COMPS; i++) {
      cudaFree(src_image[i]);
      cudaFree(d_dct[i]);
    }
    cudaFree(d_mcu_segments);
    cudaFree(d_jpeg_buf);
    cudaFree(d_quant_tables);
    nppiDCTFree(dct_state);

    jpeg_destroy_decompress(&dinfo);
  }

  void newFrame()
  {
    frame = new Frame(1920, 1080, 3);
  }

  void startTiming()
  {
    timing_current_start = cv::getTickCount();
  }

  void stopTiming()
  {
    timing_acc += (cv::getTickCount() - timing_current_start) / cv::getTickFrequency();
    timing_acc_n += 1.0;

    if(timing_acc_n >= 100.0)
    {
      double avg = (timing_acc / timing_acc_n);
      std::cout << "[CudaJpegRgbPacketProcessor] avg. time: " << (avg * 1000) << "ms -> ~" << (1.0/avg) << "Hz" << std::endl;
      timing_acc = 0.0;
      timing_acc_n = 0.0;
    }
  }

  void initializeNpp()
  {
    const NppLibraryVersion *ver = nppGetLibVersion();
    const char *gpuName = nppGetGpuName();
    std::cout << "[CudaJpegRgbPacketProcessor::initializeNpp] NPP " <<
		ver->major << "." << ver->minor << "." << ver->build << " on GPU " << gpuName << std::endl;

    nppSafeCall(nppiDCTInitAlloc(&dct_state));
    cudaSafeCall(cudaMalloc(&d_quant_tables, COMPS * DCTSIZE2));
    cudaSafeCall(cudaMalloc(&d_jpeg_buf, d_jpeg_buf_size));
    cudaSafeCall(cudaMalloc(&d_mcu_segments, SEGMENTS*sizeof(uint2)));

    size_t pitch;
    for (int i = 0; i < COMPS; i++) {
      src_size[i].width = WIDTH;
      /* Assuming YUV422 */
      if (i > 0)
        src_size[i].width /= 2;
      src_size[i].height = HEIGHT;

      cudaSafeCall(cudaMallocPitch(&d_dct[i], &pitch, src_size[i].width * DCTSIZE * sizeof(Npp16s), src_size[i].height / DCTSIZE));
      dct_step[i] = static_cast<Npp32s>(pitch);
      cudaSafeCall(cudaMallocPitch(&src_image[i], &pitch, src_size[i].width, src_size[i].height));
      src_image_step[i] = static_cast<Npp32s>(pitch);
    }

    cudaSafeCall(cudaMallocPitch(&packed_image, &pitch, WIDTH*3, HEIGHT));
    packed_image_step = static_cast<Npp32s>(pitch);

    huffman_tables_full = new uint16_t[4 * (1 << 16)];
    huffman_tables_quick = new uint16_t[QUICK_TABLE_ITEMS];
    mcu_segments = new uint2[SEGMENTS];
    quant_tables = new unsigned char[COMPS * DCTSIZE2];

    cudaFuncSetCacheConfig(gpujpeg_huffman_decoder_decode_kernel, cudaFuncCachePreferL1);

    huffman_tables_loaded = false;
  }

  struct derived_huffman_table {
    int maxcode[18];
    int valoffset[18];
    unsigned char huffval[256];
  };

  struct derived_huffman_table derived_tables[4];

  void deriveHuffmanTable(const struct jpeg_decompress_struct &dinfo, bool dc, int tblno, struct derived_huffman_table *derived)
  {
    const JHUFF_TBL *huff;
    if (dc)
      huff = dinfo.dc_huff_tbl_ptrs[tblno];
    else
      huff = dinfo.ac_huff_tbl_ptrs[tblno];

    // Figure C.1: make table of Huffman code length for each symbol
    char huffsize[257];
    int p = 0;
    for (int l = 1; l <= 16; l++)
      for (int i = 0; i < huff->bits[l]; i++)
        huffsize[p++] = (char)l;
    huffsize[p] = 0;

    // Figure C.2: generate the codes themselves
    unsigned int huffcode[257];
    unsigned int code = 0;
    int si = huffsize[0];
    p = 0;
    while (huffsize[p]) {
      while (huffsize[p] == si) {
        huffcode[p++] = code;
        code++;
      }
      code <<= 1;
      si++;
    }

    // Figure F.15: generate decoding tables for bit-sequential decoding
    p = 0;
    for (int l = 1; l <= 16; l++) {
      if (huff->bits[l]) {
        derived->valoffset[l] = p - (int)huffcode[p];
        p += huff->bits[l];
        derived->maxcode[l] = huffcode[p-1];
      } else {
        derived->maxcode[l] = -1;
      }
    }
    derived->valoffset[17] = 0;
    derived->maxcode[17] = 0xFFFFFL;

    for (int i = 0; i < sizeof(derived->huffval); i++)
      derived->huffval[i] = (unsigned char)huff->huffval[i];
  }

  void gpujpeg_huffman_gpu_decoder_table_setup(const int bits, const struct derived_huffman_table* d_table_src, const int table_idx)
  {
    // Decode one codeword from given bits to get following:
    //  - minimal number of bits actually needed to decode the codeword (up to 16 bits, 0 for invalid ones)
    //  - category ID represented by the codeword, consisting from:
    //    - number of run-length-coded preceding zeros (up to 16, or 63 for both special end-of block symbol or invalid codewords)
    //    - bit-size of the actual value of coefficient (up to 16, 0 for invalid ones)
    int code_nbits = 1, category_id = 0;

    // First, decode codeword length (This is per Figure F.16 in the JPEG spec.)
    int code_value = bits >> 15; // only single bit initially
    while ( code_value > d_table_src->maxcode[code_nbits] ) {
      code_value = bits >> (16 - ++code_nbits); // not enough to decide => try more bits
    }

    // With garbage input we may reach the sentinel value l = 17.
    if ( code_nbits > 16 ) {
      code_nbits = 0;
      // category ID remains 0 for invalid symbols from garbage input
    } else {
      category_id = d_table_src->huffval[d_table_src->valoffset[code_nbits] + code_value];
    }

    // decompose category number into 1 + number of run-length coded zeros and length of the value
    // (special category #0 contains all invalid codes and special end-of-block code -- all of those codes
    // should terminate block decoding => use 64 run-length zeros and 0 value bits for such symbols)
    const int value_nbits = 0xF & category_id;
    const int rle_zero_count = category_id ? min(1 + (category_id >> 4), 64) : 64;

    // save all the info into the right place in the destination table
    const int packed_info = (rle_zero_count << 9) + (code_nbits << 4) + value_nbits;
    huffman_tables_full[(table_idx << 16) + bits] = packed_info;

    // some threads also save entries into the quick table
    const int dest_idx_quick = bits >> (16 - QUICK_CHECK_BITS);
    if(bits == (dest_idx_quick << (16 - QUICK_CHECK_BITS))) {
      // save info also into the quick table if number of required bits is less than quick
      // check bit count, otherwise put 0 there to indicate that full table lookup consultation is needed
      huffman_tables_quick[(table_idx << QUICK_CHECK_BITS) + dest_idx_quick] = code_nbits <= QUICK_CHECK_BITS ? packed_info : 0;
    }
  }

  void gpujpeg_huffman_decoder_table_kernel()
  {
    for (int idx = 0; idx < 65536; idx++)
      for (int t = 0; t < 4; t++)
        gpujpeg_huffman_gpu_decoder_table_setup(idx, &derived_tables[t], t);

    cudaSafeCall(cudaMemcpyToSymbol(gpujpeg_huffman_gpu_decoder_tables_quick_const, huffman_tables_quick, sizeof(uint16_t)*QUICK_TABLE_ITEMS));
    cudaSafeCall(cudaMemcpyToSymbol(gpujpeg_huffman_gpu_decoder_tables_full, huffman_tables_full, sizeof(uint16_t)*(4 * (1 << 16))));
  }

  int extractSegments(const unsigned char *buf, size_t len)
  {
    int seg = 0;
    mcu_segments[0].x = 0;

    const unsigned char *p = buf;
    for (;;) {
      // Use rawmemchr because the end is always ff d9
      p = (const unsigned char *)rawmemchr(p, '\xff');
      p++;
      if (p[0] >= '\xd0' && p[0] <= '\xd7') {
        mcu_segments[seg].y = p - buf - 1;
        seg++;
        if (seg >= SEGMENTS)
          return -1;
        mcu_segments[seg].x = p - buf + 1;
      } else if (p[0] == '\xd9') {
        mcu_segments[seg].y = p - buf - 1;
        seg++;
        break;
      }
    }

/*
    int expect = 0xd0;
    for (int i = 0; i < len - 1; i++) {
      if (buf[i] != 0xff)
        continue;
      i++;
      if (buf[i] == 0)
        continue;

      if (buf[i] == expect) {
        mcu_segments[seg].y = i - 1;
        seg++;
        if (seg >= SEGMENTS)
          return -1;
        mcu_segments[seg].x = i + 1;
        expect = ((expect < 0xd7) ? (expect + 1) : 0xd0);
      } else if (buf[i] == 0xd9) {
        mcu_segments[seg].y = i - 1;
        seg++;
        break;
      }
    }
*/
    return seg;
  }

  void decompress(unsigned char *buf, size_t len)
  {
    jpeg_mem_src(&dinfo, buf, len);
    int header_status = jpeg_read_header(&dinfo, true);
    if (header_status != JPEG_HEADER_OK)
      throw std::runtime_error("header is not ok");

    if (dinfo.image_width != WIDTH || dinfo.image_height != HEIGHT ||
      dinfo.num_components != COMPS || dinfo.restart_interval != RESTART_INTERVAL)
      throw std::runtime_error("image parameters do not match preset");

    cudaMemcpyAsync(d_jpeg_buf, dinfo.src->next_input_byte, dinfo.src->bytes_in_buffer, cudaMemcpyHostToDevice);

    int segs = extractSegments(dinfo.src->next_input_byte, dinfo.src->bytes_in_buffer);
    if (segs != SEGMENTS)
      throw std::runtime_error("missing segments");
    cudaMemcpyAsync(d_mcu_segments, mcu_segments, sizeof(uint2)*SEGMENTS, cudaMemcpyHostToDevice);

    /* Quant Table */
    for (int i = 0; i < COMPS; i++) {
      const int natural_order[DCTSIZE2] = {
         0,  1,  8, 16,  9,  2,  3, 10,
        17, 24, 32, 25, 18, 11,  4,  5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13,  6,  7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63,
      };
      int n = dinfo.comp_info[i].quant_tbl_no;
      for (int j = 0; j < DCTSIZE2; j++)
        quant_tables[i * DCTSIZE2 + j] = dinfo.quant_tbl_ptrs[n]->quantval[natural_order[j]];
    }
    cudaMemcpyAsync(d_quant_tables, quant_tables, COMPS * DCTSIZE2, cudaMemcpyHostToDevice);

    /* Huffman table */
    if (!huffman_tables_loaded) {
      for (int k = 0; k < 2; k++)
        for (int j = 0; j < 2; j++)
          deriveHuffmanTable(dinfo, !j, k, &derived_tables[k*2+j]);
      gpujpeg_huffman_decoder_table_kernel();

      huffman_tables_loaded = true;
    }

    gpujpeg_huffman_decoder_decode_kernel<<<2, THREADS_PER_TBLOCK>>>(d_mcu_segments, d_jpeg_buf,
      d_dct[0], d_dct[1], d_dct[2], dct_step[0], dct_step[1], dct_step[2]);

    /* Apply inverse DCT */
    cudaDeviceSynchronize();
    for (int i = 0; i < COMPS; i++)  {
      nppSafeCall(nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R_NEW(d_dct[i], dct_step[i], src_image[i], src_image_step[i],
        d_quant_tables + i * DCTSIZE2, src_size[i], dct_state));
    }

    nppSafeCall(nppiYUV422ToRGB_8u_P3C3R(src_image, src_image_step, packed_image, packed_image_step, src_size[0]));

    //cudaSafeCall(cudaMemcpy2D(frame->data, WIDTH*3, packed_image, packed_image_step, WIDTH*3, HEIGHT, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    cudaSafeCall(cudaGetLastError());
    jpeg_abort_decompress(&dinfo);
  }

};

CudaJpegRgbPacketProcessor::CudaJpegRgbPacketProcessor() :
    impl_(new CudaJpegRgbPacketProcessorImpl())
{
}

CudaJpegRgbPacketProcessor::~CudaJpegRgbPacketProcessor()
{
  delete impl_;
}

void CudaJpegRgbPacketProcessor::process(const RgbPacket &packet)
{
  if (listener_ == 0)
    return;

	impl_->startTiming();

  try
  {
    impl_->decompress(packet.jpeg_buffer, packet.jpeg_buffer_length);
		if(listener_->onNewFrame(Frame::Color, impl_->frame))
		{
			impl_->newFrame();
		}
	}
	catch (const std::runtime_error &err)
	{
		std::cerr << "[CudaJpegRgbPacketProcessor::doProcess] Failed to decompress: '" << err.what() << "'" << std::endl;
	}

	impl_->stopTiming();
}

} /* namespace libfreenect2 */
