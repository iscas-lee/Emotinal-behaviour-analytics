// This file contains the headers,
// which will be part of the precompiled headers

#pragma once

////////////////////////////////////////////////////////////////////
// Generic / Windows

#include <SDKDDKVer.h>
#include <stdio.h>
#include <tchar.h>
#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>
#include <math.h>

////////////////////////////////////////////////////////////////////
// OpenCV

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

////////////////////////////////////////////////////////////////////
// OpenFace

#include <FaceActionUnit.h>
#include <FaceUtil.h>
#include <FaceDetector.h>
#include <FaceAnalyser.h>
#include <FaceModel.h>
#include <FaceSystem.h>

////////////////////////////////////////////////////////////////////
// OSCPack

#include <oscpack/OscOutboundPacketStream.h>
#include <oscpack/UdpSocket.h>

////////////////////////////////////////////////////////////////////
// FaceSensor

#include "OscSender.h"

