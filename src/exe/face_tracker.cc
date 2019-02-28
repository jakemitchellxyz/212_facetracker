///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2010, Jason Mora Saragih, all rights reserved.
//
// This file is part of FaceTracker.
//
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
//
//     * The software is provided under the terms of this licence stricly for
//       academic, non-commercial, not-for-profit purposes.
//     * Redistributions of source code must retain the above copyright notice, 
//       this list of conditions (licence) and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright 
//       notice, this list of conditions (licence) and the following disclaimer 
//       in the documentation and/or other materials provided with the 
//       distribution.
//     * The name of the author may not be used to endorse or promote products 
//       derived from this software without specific prior written permission.
//     * As this software depends on other libraries, the user must adhere to 
//       and keep in place any licencing terms of those libraries.
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite the following work:
//
//       J. M. Saragih, S. Lucey, and J. F. Cohn. Face Alignment through 
//       Subspace Constrained Mean-Shifts. International Conference of Computer 
//       Vision (ICCV), September, 2009.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
// EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF 
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////

// Hugely Modified by Md. Iftekhar Tanveer for Blind Emotion Project

#include <Tracker.h>
//#include <highgui.h>
#include <iostream>
#include <math.h>
//=============================================================================
void Draw(cv::Mat &image,cv::Mat &shape,cv::Mat &con,cv::Mat &tri,cv::Mat &visi)
{
  int i,n = shape.rows/2; 
  cv::Point p1,p2; 
  cv::Scalar c;

  //draw triangulation
  c = CV_RGB(0,0,0);
  for(i = 0; i < tri.rows; i++){
    if(visi.at<int>(tri.at<int>(i,0),0) == 0 ||
       visi.at<int>(tri.at<int>(i,1),0) == 0 ||
       visi.at<int>(tri.at<int>(i,2),0) == 0)continue;
    p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
		   shape.at<double>(tri.at<int>(i,0)+n,0));
    p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
		   shape.at<double>(tri.at<int>(i,1)+n,0));
    cv::line(image,p1,p2,c);
    p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
		   shape.at<double>(tri.at<int>(i,0)+n,0));
    p2 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
		   shape.at<double>(tri.at<int>(i,2)+n,0));
    cv::line(image,p1,p2,c);
    p1 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
		   shape.at<double>(tri.at<int>(i,2)+n,0));
    p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
		   shape.at<double>(tri.at<int>(i,1)+n,0));
    cv::line(image,p1,p2,c);
  }
  //draw connections
  c = CV_RGB(0,0,255);
  for(i = 0; i < con.cols; i++){
    if(visi.at<int>(con.at<int>(0,i),0) == 0 ||
       visi.at<int>(con.at<int>(1,i),0) == 0)continue;
    p1 = cv::Point(shape.at<double>(con.at<int>(0,i),0),
		   shape.at<double>(con.at<int>(0,i)+n,0));
    p2 = cv::Point(shape.at<double>(con.at<int>(1,i),0),
		   shape.at<double>(con.at<int>(1,i)+n,0));
    cv::line(image,p1,p2,c,1);
  }
  //draw points
  for(i = 0; i < n; i++){    
    if(visi.at<int>(i,0) == 0)continue;
    p1 = cv::Point(shape.at<double>(i,0),shape.at<double>(i+n,0));
    c = CV_RGB(255,0,0); 
	
	// color brows green
	if (i == 57) {
		c = CV_RGB(0, 255, 0); 
	}
	
	cv::circle(image, p1, 2, c);
  }return;
}
//=============================================================================
int parse_cmd(int argc, const char** argv,
	      char* ftFile,char* conFile,char* triFile,
	      bool &fcheck,double &scale,int &fpd)
{
  int i; fcheck = false; scale = 1; fpd = -1;
  for(i = 1; i < argc; i++){
    if((std::strcmp(argv[i],"-?") == 0) ||
       (std::strcmp(argv[i],"--help") == 0)){
      std::cout << "track_face:- Written by Jason Saragih 2010" << std::endl
	   << "Performs automatic face tracking" << std::endl << std::endl
	   << "#" << std::endl 
	   << "# usage: ./face_tracker [options]" << std::endl
	   << "#" << std::endl << std::endl
	   << "Arguments:" << std::endl
	   << "-m <string> -> Tracker model (default: ../model/face2.tracker)"
	   << std::endl
	   << "-c <string> -> Connectivity (default: ../model/face.con)"
	   << std::endl
	   << "-t <string> -> Triangulation (default: ../model/face.tri)"
	   << std::endl
	   << "-s <double> -> Image scaling (default: 1)" << std::endl
	   << "-d <int>    -> Frames/detections (default: -1)" << std::endl
	   << "--check     -> Check for failure" << std::endl;
      return -1;
    }
  }
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"--check") == 0){fcheck = true; break;}
  }
  if(i >= argc)fcheck = false;
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"-s") == 0){
      if(argc > i+1)scale = std::atof(argv[i+1]); else scale = 1;
      break;
    }
  }
  if(i >= argc)scale = 1;
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"-d") == 0){
      if(argc > i+1)fpd = std::atoi(argv[i+1]); else fpd = -1;
      break;
    }
  }
  if(i >= argc)fpd = -1;
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"-m") == 0){
      if(argc > i+1)std::strcpy(ftFile,argv[i+1]);
      else strcpy(ftFile,"../model/face2.tracker");
      break;
    }
  }
  if(i >= argc)std::strcpy(ftFile,"../model/face2.tracker");
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"-c") == 0){
      if(argc > i+1)std::strcpy(conFile,argv[i+1]);
      else strcpy(conFile,"../model/face.con");
      break;
    }
  }
  if(i >= argc)std::strcpy(conFile,"../model/face.con");
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"-t") == 0){
      if(argc > i+1)std::strcpy(triFile,argv[i+1]);
      else strcpy(triFile,"../model/face.tri");
      break;
    }
  }
  if(i >= argc)std::strcpy(triFile,"../model/face.tri");
  return 0;
}
float getSystemTime(){
	return cv::getTickCount()/cv::getTickFrequency()*1000;
}
//=============================================================================
int main(int argc, const char** argv)
{
  //parse command line arguments
  char ftFile[256],conFile[256],triFile[256];
  bool fcheck = false; double scale = 1; int fpd = -1; bool show = true;
  if(parse_cmd(argc,argv,ftFile,conFile,triFile,fcheck,scale,fpd)<0)return 0;

  //set other tracking parameters
  std::vector<int> wSize1(1); wSize1[0] = 7;
  std::vector<int> wSize2(3); wSize2[0] = 11; wSize2[1] = 9; wSize2[2] = 7;
  int nIter = 5; double clamp=3,fTol=0.01; 
  FACETRACKER::Tracker model(ftFile);
  cv::Mat tri=FACETRACKER::IO::LoadTri(triFile);
  cv::Mat con=FACETRACKER::IO::LoadCon(conFile);
  
  //initialize camera and display window
  cv::Mat frame,gray,im; double fps=0; char sss[256]; std::string text; 
  CvCapture* camera = cvCreateCameraCapture(CV_CAP_ANY); if(!camera)return -1;
  int64 t1,t0 = cvGetTickCount(); int fnum=0;
  cvNamedWindow("Face Tracker",1);
  std::cout << "Hot keys: "        << std::endl
	    << "\t ESC - quit"     << std::endl
	    << "\t d   - Redetect" << std::endl;

  //loop until quit (i.e user presses ESC)
  bool failed = true;
  double pitch = 0, yaw = 0, roll = 0;

  // Main loop
  int center_reset = 2000;
  double center_pitch = 0.0;
  double center_yaw = 0.0;
  double center_roll = 0.0;

  float gesture_timer = 3000;
  float gesture_threshold = 0.2;
  
  bool move = false;
  float yes_gesture_started = 0.0;
  float yes_threshold = gesture_threshold;
  float yes_direction = 0.0;

  bool turn = false;
  float no_gesture_started = 0.0;
  float no_threshold = gesture_threshold;
  float no_direction = 0;

  bool tilt = false;
  float inod_gesture_started = 0.0;
  float inod_threshold = gesture_threshold;
  float inod_direction = 0;

  double calibrated_right_mouth_corner = 0.0;
  double calibrated_left_mouth_corner = 0.0;

  while(1){ 
			//grab image, resize and flip
			IplImage* I = cvQueryFrame(camera); if(!I)continue; frame = I;
			if(scale == 1)im = frame; 
			else cv::resize(frame,im,cv::Size(scale*frame.cols,scale*frame.rows));
			cv::flip(im,im,1); cv::cvtColor(im,gray,CV_BGR2GRAY);

			//track this image
			std::vector<int> wSize; if(failed)wSize = wSize2; else wSize = wSize1; 
			if(model.Track(gray,wSize,fpd,nIter,clamp,fTol,fcheck) == 0){
			  int idx = model._clm.GetViewIdx(); failed = false;

			  // Extract pitch, yaw and roll movements 
			  pitch = model._clm._pglobl.at<double>(1);
			  yaw = model._clm._pglobl.at<double>(2);
			  roll = model._clm._pglobl.at<double>(3);

			  // =================================================================
			  // =============== Homework: Your code will go here >>
			  float this_time = getSystemTime();
			  //printf("time: %f, Pitch = %0.2f  Yaw = %0.2f  Roll = %0.2f\n",this_time,pitch,yaw,roll);
			 
			  // CENTERING ======================================================
			  int int_time = static_cast<int>(this_time);
			 		  
			  if ((int_time % center_reset < 50) && !(yes_gesture_started || no_gesture_started || inod_gesture_started))
			  {
				  //printf("\n\n %d \n\n", int_time % center_reset);
				  printf("\n\nReset Center\n\n");
				  center_pitch = pitch;
				  center_yaw = yaw;
				  center_roll = roll;
			  }

			  // YES ========================================================
			  double adjusted_pitch = center_pitch - pitch;
			  
			  if ((abs(adjusted_pitch) > yes_threshold) && (move == false))
			  {
				  yes_direction = adjusted_pitch;
				  move = true;
				  yes_gesture_started = this_time;
			  }
			  if ((move && (abs(adjusted_pitch) < yes_threshold)) && ((adjusted_pitch > 0) != (yes_direction > 0)))
			  {
				  if (yes_gesture_started && (this_time - yes_gesture_started < gesture_timer))
				  {
					  printf("\n\n one 'yes' detected!!\n\n");
					  yes_gesture_started = 0.0;
				  }
				  move = false;
			  }

			  // NO =========================================================
			  double adjusted_yaw = center_yaw - yaw;
			  
			  if ((abs(adjusted_yaw) > no_threshold) && (turn == false))
			  {
				  no_direction = adjusted_yaw;
				  turn = true;
				  no_gesture_started = this_time;
			  }
			  if ((turn && (abs(adjusted_yaw) < no_threshold)) && ((adjusted_yaw > 0) != (no_direction > 0)))
			  {
				  if (this_time - no_gesture_started < gesture_timer)
				  {
					  printf("\n\n one 'no' detected!!\n\n");
					  no_gesture_started = 0.0;
				  }
				  turn = false;
			  }

			  // INDIAN NOD =================================================
			  double adjusted_roll = center_roll - roll;
			  
			  if ((abs(adjusted_roll) > inod_threshold) && (tilt == false))
			  {
				  inod_direction = adjusted_roll;
				  tilt = true;
				  inod_gesture_started = this_time;
			  }
			  if ((tilt && (abs(adjusted_roll) < inod_threshold)) && ((adjusted_roll > 0) != (inod_direction > 0)))
			  {
				  if (this_time - inod_gesture_started < gesture_timer)
				  {
					  printf("\n\n one 'indian nod' detected!!\n\n");
					  inod_gesture_started = 0.0;
				  }
				  tilt = false;
			  }
        
        // SMILE ======================================================
			  int i, n = model._shape.rows / 2;
			  int left_corner = 48;
			  int right_corner = 54;

			  double x_right = model._shape.at<double>(right_corner, 0);
			  double x_left = model._shape.at<double>(left_corner, 0);

			  if (abs(x_right - calibrated_right_mouth_corner) > 150)
			  {
				  calibrated_right_mouth_corner = x_right;
			  }
			  if (abs(x_left - calibrated_left_mouth_corner) > 150)
			  {
				  calibrated_left_mouth_corner = x_left;
			  }

			  calibrated_left_mouth_corner += 0.05 * (x_left - calibrated_left_mouth_corner);
			  calibrated_right_mouth_corner += 0.05 * (x_right - calibrated_right_mouth_corner);

			  double real_width = x_right - x_left;
			  double calibrated_width = calibrated_right_mouth_corner - calibrated_left_mouth_corner;

			  double growth = real_width - calibrated_width;

			  if (growth > 8)
			  {
				  printf("\n\n Stop smiling, cutie!\n\n");
			  }
			  
			  cv::Point left_corner_point = cv::Point(model._shape.at<double>(left_corner, 0), model._shape.at<double>(left_corner+n, 0));
			  cv::Point right_corner_point = cv::Point(model._shape.at<double>(right_corner, 0), model._shape.at<double>(right_corner + n, 0));
			  
			  cv::Scalar c = CV_RGB(0, 255, 0);
			
			  //cv::line(im, cv::Point(calibrated_left_mouth_corner, 0), cv::Point(calibrated_left_mouth_corner, 10000), c,1);
			  //cv::line(im, cv::Point(calibrated_right_mouth_corner, 0), cv::Point(calibrated_right_mouth_corner, 10000), c, 1);
			  //cv::circle(im, left_corner_point, 3, c); //image, center, radius, color
			  //cv::circle(im, right_corner_point, 3, c); //image, center, radius, color

			  // SURPRISE ========================================================
			  int n = model._shape.rows / 2;
			  cv::Scalar c = CV_RGB(0, 255, 0);
				
			  double eye_height = model._shape.at<double>(27 + n, 0);
			  double avg_brow_height = 0.0;
			  for (int i = 17; i < 27; i++)
			  {
				  avg_brow_height += model._shape.at<double>(i + n, 0);
			  }
			  avg_brow_height = avg_brow_height / 10;
			  
			  //avg height line
			  cv::Point brow_center_point = cv::Point(10, avg_brow_height);
			  cv::line(im, cv::Point(10, avg_brow_height), cv::Point(600, avg_brow_height), c);

			  double mouth_open_distance = (model._shape.at<double>(57 + n, 0) - (model._shape.at<double>(51 + n, 0)));
			  printf("\n%f\n", mouth_open_distance);
			  // =================================================================
	  
			  Draw(im,model._shape,con,tri,model._clm._visi[idx]); 
			}else{
			  if(show){cv::Mat R(im,cvRect(0,0,150,50)); R = cv::Scalar(0,0,255);}
			  model.FrameReset(); failed = true;
			}     
			//draw framerate on display image 
			if(fnum >= 9){      
			  t1 = cvGetTickCount();
			  fps = 10.0/((double(t1-t0)/cvGetTickFrequency())/1e+6); 
			  t0 = t1; fnum = 0;
			}else fnum += 1;
			if(show){
			  sprintf(sss,"%d frames/sec",(int)ceil(fps)); text = sss;
			  cv::putText(im,text,cv::Point(10,20),
				  CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(255,255,255));
			}
			//show image and check for user input
			imshow("Face Tracker",im); 
			int c = cvWaitKey(5);
			if(c == 27)
				break; 
			else 
				if(char(c) == 'd')model.FrameReset();
  } // End of main loop

  return 0;
}
//=============================================================================
