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

void surf(Mat img1, Mat img2){
    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
    Mat des1, des2;
    
    int minHessian = 600;
    
    Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create( minHessian );
    
    detector->detect( img1, keypoints1 );
    detector->detect( img2, keypoints2 );
    
    detector->detectAndCompute(img1, Mat(), keypoints1, des1);
    detector->detectAndCompute(img2, Mat(), keypoints2, des2);
    
//    drawKeypoints(img1, keypoints1, img1);
//    drawKeypoints(img2, keypoints2, img2);
//    
//    imshow("img1", img1);
//    imshow("img2", img2);
//    waitKey(0);
    
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( des1, des2, matches );
    
    double max_dist = 0; double min_dist = 100;
    
    for( int i = 0; i < des1.rows; i++ )
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    
    std::vector< DMatch > good_matches;
    
    for( int i = 0; i < des1.rows; i++ )
    { if( matches[i].distance <= max(2*min_dist, 0.02) )
    { good_matches.push_back( matches[i]); }
    }
    
    //-- Draw only "good" matches
    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2,
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    
    for( int i = 0; i < good_matches.size(); i++ )
    {
        printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx );
        
        cout << keypoints1[good_matches[i].queryIdx].pt << endl;
    }
    
    //-- Show detected matches
    imshow( "Good Matches", img_matches );
    waitKey(0);
}


int main(int argc, const char * argv[]) {
    
    Mat img1 = imread(DataManager::getInstance().FULL_PATH_PHOTO + "Marking_day3/15.JPG");
    Mat img2 = imread(DataManager::getInstance().FULL_PATH_PHOTO + "Marking_day3/t3.png");
    
//    Mat img1 = imread(DataManager::getInstance().FULL_PATH_PHOTO + "Marking_day3/15.JPG",1);
//    Mat img2 = imread(DataManager::getInstance().FULL_PATH_PHOTO + "Marking_day3/template1.png", 1);
    Mat img3 = imread(DataManager::getInstance().FULL_PATH_PHOTO + "Marking_day3/template2.png", 1);
    Mat img4 = imread(DataManager::getInstance().FULL_PATH_PHOTO + "Marking_day3/template3.png", 1);
    Mat img5 = imread(DataManager::getInstance().FULL_PATH_PHOTO + "Marking_day3/template4.png", 1);

    
    freak(img1, img2);

    
    
    
    
    vector<Point> correlationVect;
    
    int x1 = 0;
    int y1 = 0;
    int x2 = 24;
    int y2 = 24;
    int step = 6;
    
    
//    Mat subImg(img1, Rect(790,400, 24,24));
//    imshow("asd", subImg);
//    waitKey(0);
//    imwrite(DataManager::getInstance().FULL_PATH_PHOTO + "template4.png",subImg);
    
    
    while(x1 + x2 < img1.cols){
        while(y1 + y2 < img1.rows){
            Mat subImg(img1, Rect(x1,y1, x2,y2));
    
            Mat pa = Mat::zeros(subImg.size(), CV_8UC1);
            Mat pb = Mat::zeros(img2.size(), CV_8UC1);
            Mat pc = Mat::zeros(img2.size(), CV_8UC1);
            Mat pd = Mat::zeros(img2.size(), CV_8UC1);
            Mat pe = Mat::zeros(img2.size(), CV_8UC1);
            
            logPolar(subImg, pa, Point2f(subImg.cols >> 1, subImg.rows >> 1), 5, WARP_FILL_OUTLIERS);
            logPolar(img2, pb, Point2f(img2.cols >> 1, img2.rows >> 1), 5, WARP_FILL_OUTLIERS);
            logPolar(img3, pc, Point2f(img3.cols >> 1, img3.rows >> 1), 5, WARP_FILL_OUTLIERS);
            logPolar(img4, pd, Point2f(img4.cols >> 1, img4.rows >> 1), 5, WARP_FILL_OUTLIERS);
            logPolar(img5, pe, Point2f(img5.cols >> 1, img5.rows >> 1), 5, WARP_FILL_OUTLIERS);
            
//            imshow("origin", subImg);
//            imshow("template", img2);
//            imshow("pa", pa);
//            imshow("pb", pb);
//            cout << x1 << " " << y1 << " " << x2 << " " << y2 << endl;
            
            cv::Mat match1, match2, match3, match4;
            cv::matchTemplate(pa, pb, match1, CV_TM_CCORR_NORMED);
            cv::matchTemplate(pa, pc, match2, CV_TM_CCORR_NORMED);
            cv::matchTemplate(pa, pd, match3, CV_TM_CCORR_NORMED);
            cv::matchTemplate(pa, pe, match4, CV_TM_CCORR_NORMED);
//            double minVal, maxVal;
//            cv::Point minLoc, maxLoc;
//            cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
//            cv::Point screwCenter = maxLoc + cv::Point(pb.cols/2, pb.rows/2);
            Point loc(x1,y1);
            
            if(match1.at<float>(0,0) >= 0.97f)
                correlationVect.push_back(loc);
            
            if(match2.at<float>(0,0) >= 0.97f)
                correlationVect.push_back(loc);
            
            if(match3.at<float>(0,0) >= 0.97f)
                correlationVect.push_back(loc);
            
            if(match4.at<float>(0,0) >= 0.97f)
                correlationVect.push_back(loc);
            
//            cout << result.at<float>(0,0) << endl;
//            cout << screwCenter << endl;
//            waitKey(0);
            
            
            
            
//            cv::Mat pa_64f, pb_64f;
//            pa.convertTo(pa_64f, CV_64FC1);
//            pb.convertTo(pb_64f, CV_64FC1);
//           z 
//            Point2d pt = phaseCorrelate(pa_64f, pb_64f);
//            correlationVect.push_back(pt);
            
            y1+=step;
        }
        y1 = 0;
        x1+=step;
    }


    cout << correlationVect.size() << endl;

    
    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };
    
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    
    const float* ranges[] = { h_ranges, s_ranges };
    
    int channels[] = { 0, 1 };
    
    MatND hist_base1, hist_base2, hist_base3, hist_base4;
    
    cvtColor(img2, img2, COLOR_BGR2HSV);
    cvtColor(img3, img3, COLOR_BGR2HSV);
    cvtColor(img4, img4, COLOR_BGR2HSV);
    cvtColor(img5, img5, COLOR_BGR2HSV);
    
    calcHist( &img2, 1, channels, Mat(), hist_base1, 2, histSize, ranges, true, false );
    normalize( hist_base1, hist_base1, 0, 1, NORM_MINMAX, -1, Mat() );
    
    calcHist( &img3, 1, channels, Mat(), hist_base2, 2, histSize, ranges, true, false );
    normalize( hist_base2, hist_base2, 0, 1, NORM_MINMAX, -1, Mat() );
    
    calcHist( &img4, 1, channels, Mat(), hist_base3, 2, histSize, ranges, true, false );
    normalize( hist_base3, hist_base3, 0, 1, NORM_MINMAX, -1, Mat() );
    
    calcHist( &img5, 1, channels, Mat(), hist_base4, 2, histSize, ranges, true, false );
    normalize( hist_base4, hist_base4, 0, 1, NORM_MINMAX, -1, Mat() );

    int compare_method = 1;
    
    int miny = 100, mino = 100, ming = 100, minp = 100;
    Point yellow, orange, green, pink;
    
    for(int i = 0; i < correlationVect.size(); i++){
        
        MatND hist_target;
        Mat temp(img1, Rect(correlationVect[i].x,correlationVect[i].y, 24,24));
        imshow("d", temp);
        Mat tmp;
        temp.copyTo(tmp);
//        cout << i << endl;
        
        cvtColor( tmp, tmp, COLOR_BGR2HSV );
        
        calcHist( &tmp, 1, channels, Mat(), hist_target, 2, histSize, ranges, true, false );
        normalize( hist_target, hist_target, 0, 1, NORM_MINMAX, -1, Mat() );
        
        double comparey = compareHist( hist_base1, hist_target, compare_method );
        double compareo = compareHist( hist_base2, hist_target, compare_method );
        double compareg = compareHist( hist_base3, hist_target, compare_method );
        double comparep = compareHist( hist_base4, hist_target, compare_method );
        
        if(comparey < miny){
            miny = comparey;
            yellow = correlationVect[i];
        }
        
        if(compareo < mino){
            mino = compareo;
            orange = correlationVect[i];;
        }
        
        if(compareg < ming){
            ming = compareg;
            green = correlationVect[i];
        }
        
        if(comparep < minp){
            minp = comparep;
            pink = correlationVect[i];
        }
        cout << "yellow: " << comparey << endl;
        cout << "orange: " << compareo << endl;
        cout << "green: " << compareg << endl;
        cout << "pink: " << comparep << endl;
        
//        waitKey(0);
    }
    
    Mat yelloImg(img1, Rect(yellow.x,yellow.y, 24,24));
    Mat orangeImg(img1, Rect(orange.x,orange.y, 24,24));
    Mat greenImg(img1, Rect(green.x,green.y, 24,24));
    Mat pinkImg(img1, Rect(pink.x,pink.y, 24,24));
    
    imshow("yellow", yelloImg);
    imshow("orange", orangeImg);
    imshow("green", greenImg);
    imshow("pink", pinkImg);
    
    waitKey(0);


    
    
    
    
    
    
    
//    Mat pa = Mat::zeros(img1.size(), CV_8UC1);
//    Mat pb = Mat::zeros(img2.size(), CV_8UC1);
//
//    logPolar(img1, pa, Point2f(img1.cols >> 1, img1.rows >> 1), 10, WARP_FILL_OUTLIERS);
//    logPolar(img2, pb, Point2f(img2.cols >> 1, img2.rows >> 1), 10, WARP_FILL_OUTLIERS);
//
//    imshow("pa", pa);
//    imshow("pb", pb);
//    
//    cv::Mat pa_64f, pb_64f;
//    pa.convertTo(pa_64f, CV_64FC1);
//    pb.convertTo(pb_64f, CV_64FC1);
//
//    Point2d pt = phaseCorrelate(pa_64f, pb_64f);
//    float base = exp( log(img1.cols * 0.5f) / img1.cols * 0.5f);
//    float scale = pow(base, pt.x);
//    float rotate = pt.y*180/(img1.cols >> 1);
//    
//    cout << "Shift = " << pt << endl
//    << "Rotation = " << cv::format("%.2f", rotate ) << endl
//    << "Scale = " << cv::format("%.2f", scale )
//    << endl;
//
//    Point2f src_center(img1.cols/2.0F, img1.rows/2.0F);
//    Mat rot_mat = getRotationMatrix2D(src_center, rotate, 1.0);
//    Mat dst;
//    warpAffine(img1, dst, rot_mat, img1.size());
//    
//    
//    resize(dst, dst, Size(), scale, scale, INTER_CUBIC);
//    imshow("out", dst);
//    imshow("img1", img1);
//    imshow("img2", img2);
    
//    waitKey(0);
    
    
    
}