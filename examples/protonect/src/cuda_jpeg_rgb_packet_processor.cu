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
  if (err != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(err));
  return err;
}

namespace libfreenect2
{

class CudaJpegRgbPacketProcessorImpl
{
public:
  static const int WIDTH = 1920;
  static const int HEIGHT = 1080;
  static const int COMPS = 3;

  struct jpeg_decompress_struct dinfo;
  struct jpeg_error_mgr jerr;

  NppiDCTState *dct_state;
  Npp8u *d_quant_tables;
  NppiSize src_size[COMPS];
  Npp16s *d_dct[COMPS];
  Npp32s dct_step[COMPS];
  Npp16s *h_dct[COMPS];
  Npp8u *src_image[COMPS];
  Npp32s src_image_step[COMPS];
  Npp8u *packed_image;
  Npp32s packed_image_step;
  NppiDecodeHuffmanSpec *huff_dc_table[COMPS];
  NppiDecodeHuffmanSpec *huff_ac_table[COMPS];

  bool huffman_tables_loaded;

  Frame *frame;

  double timing_acc;
  double timing_acc_n;

  double timing_current_start;

  size_t packet_buffer_size;
  unsigned char *packet_buffer;

  CudaJpegRgbPacketProcessorImpl()
  {
    dinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&dinfo);

    initializeNpp();

    newFrame();

    timing_acc = 0.0;
    timing_acc_n = 0.0;
    timing_current_start = 0.0;
    packet_buffer_size = 0;
    packet_buffer = NULL;
  }

  ~CudaJpegRgbPacketProcessorImpl()
  {
    cudaFree(packed_image);
    for (int i = 0; i < COMPS; i++) {
      cudaFree(src_image[i]);
      cudaFree(d_dct[i]);
    }
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

    size_t pitch;
    for (int i = 0; i < COMPS; i++) {
      src_size[i].width = WIDTH;
      /* Assuming YUV422 */
      if (i > 0)
        src_size[i].width /= 2;
      src_size[i].height = HEIGHT;

      cudaSafeCall(cudaMallocPitch(&d_dct[i], &pitch, src_size[i].width * DCTSIZE * sizeof(Npp16s), src_size[i].height / DCTSIZE));
      dct_step[i] = static_cast<Npp32s>(pitch);
      cudaSafeCall(cudaHostAlloc(&h_dct[i], dct_step[i] * src_size[i].height / DCTSIZE, cudaHostAllocDefault));
      cudaSafeCall(cudaMallocPitch(&src_image[i], &pitch, src_size[i].width, src_size[i].height));
      src_image_step[i] = static_cast<Npp32s>(pitch);
    }

    cudaSafeCall(cudaMallocPitch(&packed_image, &pitch, WIDTH*3, HEIGHT));
    packed_image_step = static_cast<Npp32s>(pitch);

	int size;
	nppiDecodeHuffmanSpecGetBufSize_JPEG(&size);
    for (int i = 0; i < COMPS; i++) {
      huff_dc_table[i] = reinterpret_cast<NppiDecodeHuffmanSpec *>(new unsigned char[size]);
      huff_ac_table[i] = reinterpret_cast<NppiDecodeHuffmanSpec *>(new unsigned char[size]);
    }

    huffman_tables_loaded = false;
  }

  void decompress(unsigned char *buf, size_t len)
  {
    jpeg_mem_src(&dinfo, buf, len);
    int header_status = jpeg_read_header(&dinfo, true);
    if (header_status != JPEG_HEADER_OK)
      throw std::runtime_error("header is not ok");

    if (dinfo.image_width != WIDTH || dinfo.image_height != HEIGHT || dinfo.num_components != COMPS)
      throw std::runtime_error("image parameters do not match preset");

    /* Quant Table */
    unsigned char quant_tables[COMPS * DCTSIZE2];
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
    for (int i = 0; i < COMPS; i++) {
      nppiDecodeHuffmanSpecInitHost_JPEG(&dinfo.dc_huff_tbl_ptrs[dinfo.comp_info[i].dc_tbl_no]->bits[1], nppiDCTable, huff_dc_table[i]);
      nppiDecodeHuffmanSpecInitHost_JPEG(&dinfo.ac_huff_tbl_ptrs[dinfo.comp_info[i].ac_tbl_no]->bits[1], nppiACTable, huff_ac_table[i]);
    }
    nppiDecodeHuffmanScanHost_JPEG_8u16s_P3R(
      dinfo.src->next_input_byte, dinfo.src->bytes_in_buffer - 2,
      dinfo.restart_interval, dinfo.Ss, dinfo.Se, dinfo.Ah, dinfo.Al,
      h_dct, dct_step, huff_dc_table, huff_ac_table, src_size);
    for (int i = 0; i < COMPS; i++)
      cudaMemcpy(d_dct[i], h_dct[i], dct_step[i] * src_size[i].height / DCTSIZE, cudaMemcpyHostToDevice);

    /* Apply inverse DCT */
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
