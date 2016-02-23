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

#include<sys/types.h>
#include<dirent.h>

#include "DataManager.cpp"

#define numPath 1

using namespace cv;
using namespace std;
using namespace ml;

const string PHOTO_PATH(DataManager::FULL_PATH_PHOTO);

void scanDir(string path, vector<string>* fileList){
    DIR *dp;
    dirent *d;
    
    const char* _path = path.c_str();
    if((dp = opendir(_path)) != NULL)
        perror("opendir");
    
    while((d = readdir(dp)) != NULL)
    {
        if(!strcmp(d->d_name,".") || !strcmp(d->d_name,".."))
            continue;
        fileList->push_back(d->d_name);
    }
}

bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

void cropImage(string inputFile, string outputFile){
    
    Mat img = imread(inputFile);
    Mat origin;
    Mat output;
    
    //    cvtColor(img, img, CV_BGR2HSV);
    Mat yellowMat, greenMat, pinkMat, orangeMat;
    
    inRange(img, Scalar(110,180,150), Scalar(140,255,255), greenMat);
    inRange(img, Scalar(90,150,215), Scalar(100,255,255), orangeMat);
    inRange(img, Scalar(240,170,170), Scalar(250,200,255), pinkMat);
    inRange(img, Scalar(50,170,170), Scalar(70,255,255), yellowMat);
    
    medianBlur(yellowMat, yellowMat, 3);
    medianBlur(orangeMat, orangeMat, 3);
    medianBlur(pinkMat, pinkMat, 3);
    medianBlur(greenMat, greenMat, 3);
    output = greenMat + orangeMat + pinkMat + yellowMat;
    
    Mat canny_yellow,canny_pink,canny_green,canny_orange;
    vector<vector<Point> > contour_yellow,contour_pink,contour_green,contour_orange;
    vector<Vec4i> hierarchy;
    
    
    Canny( yellowMat, canny_yellow, 100, 200, 3 );
    Canny( orangeMat, canny_orange, 100, 200, 3 );
    Canny( pinkMat, canny_pink, 100, 200, 3 );
    Canny( greenMat, canny_green, 100, 200, 3 );
    
    findContours( canny_yellow, contour_yellow, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    findContours( canny_pink, contour_pink, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    findContours( canny_green, contour_green, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    findContours( canny_orange, contour_orange, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    
    cv::Point centoid_yellow,centoid_red,centoid_green,centoid_blue;
    
    cv::Moments momYellow= cv::moments(cv::Mat(contour_yellow[0]));
    cv::Moments momRed= cv::moments(cv::Mat(contour_pink[0]));
    cv::Moments momGreen= cv::moments(cv::Mat(contour_green[0]));
    cv::Moments momBlue= cv::moments(cv::Mat(contour_orange[0]));
    
    centoid_yellow =  Point(momYellow.m10/momYellow.m00,momYellow.m01/momYellow.m00);
    centoid_red = Point(momRed.m10/momRed.m00,momRed.m01/momRed.m00);
    centoid_green = Point(momGreen.m10/momGreen.m00,momGreen.m01/momGreen.m00);
    centoid_blue = Point(momBlue.m10/momBlue.m00,momBlue.m01/momBlue.m00);
    
    // compute the width of the new image, which will be the
    // maximum distance between bottom-right and bottom-left
    // x-coordiates or the top-right and top-left x-coordinates
    double widthA = sqrt(
                         pow(centoid_red.x - centoid_yellow.x, 2) +
                         pow(centoid_red.y - centoid_yellow.y, 2)
                         );
    double widthB = sqrt(
                         pow(centoid_green.x - centoid_blue.x, 2) +
                         pow(centoid_green.y - centoid_blue.y, 2)
                         );
    double maxWidth = max(int(widthA), int(widthB));
    
    // compute the height of the new image, which will be the
    // maximum distance between the top-right and bottom-right
    // y-coordinates or the top-left and bottom-left y-coordinates
    double heightA = sqrt(
                          pow((centoid_green.x - centoid_red.x),2) +
                          pow((centoid_green.y - centoid_red.y),2)
                          );
    double heightB = sqrt(
                          pow((centoid_blue.x - centoid_yellow.x),2) +
                          pow((centoid_blue.y - centoid_yellow.y),2)
                          );
    double maxHeight = max(int(heightA), int(heightB));
    
    cv::Point2f source_points[4];
    cv::Point2f dest_points[4];
    
    source_points[0] = centoid_blue;
    source_points[1] = centoid_red;
    source_points[2] = centoid_green;
    source_points[3] = centoid_yellow;
    
    dest_points[0] = Point(0,0);
    dest_points[1] = Point(maxWidth - 1,0);
    dest_points[2] = Point(maxWidth - 1, maxHeight - 1);
    dest_points[3] = Point(0, maxHeight);
    
    Mat m = getPerspectiveTransform(source_points, dest_points);
    warpPerspective(img, output, m, Size(maxWidth, maxHeight) );
    imwrite(outputFile, output);
    
}



int main(int argc, const char * argv[]) {

    if(argc != 3){
        cout << "usage: (params) Input_File Output_Path/FileName" << endl;
        return 1;
    }
    string inputFile = argv[1];
    string outputPath = argv[2];
    
    cropImage(inputFile, outputPath);
    
}
