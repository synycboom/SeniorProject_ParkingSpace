//
//  main.cpp
//  ParkingSpace
//
//  Created by synycboom on 9/11/2558 BE.
//  Copyright © 2558 HOME. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <array>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/features2d.hpp>
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "SVMTest.cpp"
#include "HistogramTool.cpp"
using namespace cv;
using namespace std;
using namespace ml;

int main(int argc, const char * argv[]) {
    
    Mat img = imread(DataManager::getInstance().FULL_PATH_PHOTO + "Marking_day3/3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    threshold(img, img, 150, 255, 3);
    threshold(img, img, 100, 255, 0);
    imshow("test", img);
    
    waitKey(0);
    
    
    
}