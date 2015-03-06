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
#include <nv_headers/jpeglib.h>
#include <iostream>
#include <stdexcept>

void abort_jpeg_error(j_decompress_ptr info, const char *msg)
{
  jpeg_abort_decompress(info);
  throw std::runtime_error(msg);
}

void my_error_exit(j_common_ptr info)
{
  char buffer[JMSG_LENGTH_MAX];
  info->err->format_message(info, buffer);
  abort_jpeg_error((j_decompress_ptr)info, buffer);
}

namespace libfreenect2
{

class TegraJpegRgbPacketProcessorImpl
{
public:

  struct jpeg_decompress_struct dinfo;
  struct jpeg_error_mgr jerr;

  Frame *frame;

  double timing_acc;
  double timing_acc_n;

  double timing_current_start;

  static const size_t WIDTH = 1920;
  static const size_t HEIGHT = 1080;
  static const size_t PITCH = 7680;

  TegraJpegRgbPacketProcessorImpl()
  {
    dinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = my_error_exit;

    jpeg_create_decompress(&dinfo);

    newFrame();

    timing_acc = 0.0;
    timing_acc_n = 0.0;
    timing_current_start = 0.0;
  }

  ~TegraJpegRgbPacketProcessorImpl()
  {
    jpeg_destroy_decompress(&dinfo);
  }

  void newFrame()
  {
    //XXX before we have better way to avoid unnecessary memcpy
    //hack this to pass a zero-copy pointer
    frame = new Frame(1, 1, sizeof(void *));
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
      std::cout << "[TegraJpegRgbPacketProcessor] avg. time: " << (avg * 1000) << "ms -> ~" << (1.0/avg) << "Hz" << std::endl;
      timing_acc = 0.0;
      timing_acc_n = 0.0;
    }
  }


  void decompress(unsigned char *buf, size_t len)
  {
    jpeg_mem_src(&dinfo, buf, len);
    jpeg_read_header(&dinfo, TRUE);

    if (dinfo.progressive_mode)
      abort_jpeg_error(&dinfo, "Tegra HW doesn't support progressive JPEG; use TurboJPEG");

    if (!dinfo.tegra_acceleration)
      abort_jpeg_error(&dinfo, "Tegra HW acceleration is disabled unexpectedly");

    if (dinfo.image_width != WIDTH || dinfo.image_height != HEIGHT)
      abort_jpeg_error(&dinfo, "image dimensions does not match preset");

    dinfo.out_color_space = JCS_RGBA_8888;

    jpeg_start_decompress(&dinfo);

    // Hardware acceleration returns the entire surface in one go.
    // The normal way with software decoding uses jpeg_read_scanlines with loop.
    if (jpeg_read_scanlines(&dinfo, NULL, 0) != dinfo.output_height)
      abort_jpeg_error(&dinfo, "Incomplete decoding result");

    /* Empirically: 1 surface for RGBA; 3 surfaces for YUV */
    size_t pitch = dinfo.jpegTegraMgr->pitch[0];
    unsigned char *surface = dinfo.jpegTegraMgr->buff[0];
    if (pitch == 0 || surface == NULL)
      abort_jpeg_error(&dinfo, "Empty result buffer");

    //XXX this won't work for grayscale output where itch != width*bpp for grayscale output, 
    if (pitch != PITCH || dinfo.output_height != HEIGHT)
      abort_jpeg_error(&dinfo, "buffer size mismatch");

    //XXX before we have better way to avoid unnecessary memcpy
    //hack this to pass a zero-copy pointer
    void **pp = reinterpret_cast<void **>(frame->data);
    pp[0] = surface;

    jpeg_finish_decompress(&dinfo);
  }
};

TegraJpegRgbPacketProcessor::TegraJpegRgbPacketProcessor() :
    impl_(new TegraJpegRgbPacketProcessorImpl())
{
}

TegraJpegRgbPacketProcessor::~TegraJpegRgbPacketProcessor()
{
  delete impl_;
}

void TegraJpegRgbPacketProcessor::process(const RgbPacket &packet)
{
  if (listener_ == NULL)
    return;

  impl_->startTiming();

  impl_->frame->timestamp = packet.timestamp;

  try
  {
    impl_->decompress(packet.jpeg_buffer, packet.jpeg_buffer_length);
    if (listener_->onNewFrame(Frame::Color, impl_->frame))
    {
      impl_->newFrame();
    }
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << "[TegraJpegRgbPacketProcessor::process] jpeg error " << err.what() << std::endl;
  }

  impl_->stopTiming();
}

} /* namespace libfreenect2 */
