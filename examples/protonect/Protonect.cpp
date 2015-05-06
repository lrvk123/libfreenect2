/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2011 individual OpenKinect contributors. See the CONTRIB file
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


#include <iostream>
#include <signal.h>

#include <opencv2/opencv.hpp>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/registration.h>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/threading.h>

bool protonect_shutdown = false;

void sigint_handler(int s)
{
  protonect_shutdown = true;
}

int main(int argc, char *argv[])
{
  std::string program_path(argv[0]);
  size_t executable_name_idx = program_path.rfind("Protonect");

  std::string binpath = "/";

  if(executable_name_idx != std::string::npos)
  {
    binpath = program_path.substr(0, executable_name_idx);
  }


  libfreenect2::Freenect2 freenect2;
  libfreenect2::Freenect2Device *dev = freenect2.openDefaultDevice();

  if(dev == 0)
  {
    std::cout << "no device connected or failure opening the default one!" << std::endl;
    return -1;
  }

  signal(SIGINT,sigint_handler);
  protonect_shutdown = false;

  libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
  libfreenect2::FrameMap frames;

  dev->setColorFrameListener(&listener);
  dev->setIrAndDepthFrameListener(&listener);
  dev->start();

  libfreenect2::Freenect2Device::IrCameraParams irparam = dev->getIrCameraParams();
  libfreenect2::Freenect2Device::ColorCameraParams ccp = dev->getColorCameraParams();
  libfreenect2::Registration* reg = new libfreenect2::Registration(&irparam, &ccp);

  std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
  std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;

  while(!protonect_shutdown)
  {
    listener.waitForNewFrame(frames);
    libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
    libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
    libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];


    cv::Mat rgbMat = cv::Mat(rgb->height, rgb->width, CV_8UC3, rgb->data);
    cv::Mat depthMat = cv::Mat(depth->height, depth->width, CV_32FC1, depth->data);

    cv::Mat colorRegistered = cv::Mat::zeros(depth->height, depth->width, CV_8UC3);
    cv::Mat depthRegistered = cv::Mat::ones(rgb->height, rgb->width, CV_32FC1)*65536.0f;
    const float DEPTH_FILTER_TOLERANCE = 0.01f;//allowed depth noise percentage

    for(int x = 0; x < 512; x++)
    {
      for(int y = 0; y < 424; y++)
      {
        const float &z = depthMat.at<float>(y, x);
        if(z == 0)
          continue;
        float cx, cy;
        reg->apply(x, y, z, cx, cy);
        if(cx < 0 || cy < 0 || cx >= 1920 || cy >= 1080)
          continue;
        float &min = depthRegistered.at<float>(cy, cx);
        if(min > z)
          min = z;
      }
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,3));
    cv::erode(depthRegistered, depthRegistered, kernel);

    for(int x = 0; x < 512; x++)
    {
      for(int y = 0; y < 424; y++)
      {
        const float &z = depthMat.at<float>(y, x);
        if(z == 0)
          continue;
        float cx, cy;
        reg->apply(x, y, z, cx, cy);
        if(cx < 0 || cy < 0 || cx >= 1920 || cy >= 1080)
          continue;
        const float &min = depthRegistered.at<float>(cy, cx);
        if ((z - min)/z > DEPTH_FILTER_TOLERANCE)
          continue;
        colorRegistered.at<cv::Vec3b>(y, x) = rgbMat.at<cv::Vec3b>(cy, cx);
      }
    }

    cv::imshow("depth", depthMat/4500.0f);
    cv::imshow("color_registered", colorRegistered);

    int key = cv::waitKey(1);
    protonect_shutdown = protonect_shutdown || (key > 0 && ((key & 0xFF) == 27)); // shutdown on escape

    listener.release(frames);
    //libfreenect2::this_thread::sleep_for(libfreenect2::chrono::milliseconds(100));
  }

  // TODO: restarting ir stream doesn't work!
  // TODO: bad things will happen, if frame listeners are freed before dev->stop() :(
  dev->stop();
  dev->close();

  return 0;
}
