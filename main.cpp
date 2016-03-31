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
    
    Mat rect1 = imread(DataManager::getInstance().FULL_PATH_PHOTO + "Marking_day3/1.JPG", CV_LOAD_IMAGE_GRAYSCALE);
    Mat rect2 = imread(DataManager::getInstance().FULL_PATH_PHOTO + "Marking_day3/template.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//    imshow("rect1", rect1);
//    imshow("rect2", rect2);

    int x1 = 0;
    int y1 = 0;
    int x2 = rect2.cols;
    int y2 = rect2.rows;

    while(x2 < rect1.cols){
        while(y2 < rect1.rows){
            Mat i(rect1, Rect(x1,y1, x2,y2));
            imshow("asd", i);
            waitKey(0);
            y1++;
            y2++;
        }
        x1++;
        x2++;
    }

    
    
    Mat pa = Mat::zeros(rect1.size(), CV_8UC1);
    Mat pb = Mat::zeros(rect2.size(), CV_8UC1);
    
    logPolar(rect1, pa, Point2f(rect1.cols >> 1, rect1.rows >> 1), 50, INTER_CUBIC);
    logPolar(rect2, pb, Point2f(rect2.cols >> 1, rect2.rows >> 1), 20, INTER_CUBIC);
    
//    imshow("log-rect1",pa);
//    imshow("log-rect2",pb);
    
    cv::Mat pa_64f, pb_64f;
    pa.convertTo(pa_64f, CV_64FC1);
    pb.convertTo(pb_64f, CV_64FC1);

    
    Point2d pt = phaseCorrelate(pa_64f, pb_64f);
    
    std::cout << "Shift = " << pt
    << "Rotation = " << cv::format("%.2f", pt.y*180/(rect1.cols >> 1))
    << std::endl;
    cout << exp(pt.x) << endl;
//    resize(rect1, rect1, Size(), exp(pt.x), exp(pt.x), INTER_CUBIC);
    imshow("out", rect1);
    
    waitKey(0);
    
    
    
}