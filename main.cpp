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
#include <map>

using namespace cv;
using namespace std;
using namespace ml;

int h_min_slider;
int s_min_slider;
int v_min_slider;

int h_max_slider;
int s_max_slider;
int v_max_slider;

const int YELLOW = 0;
const int BLUE = 1;
const int GREEN = 2;
const int PINK = 3;

Mat img1;
Point yellowMarker, blueMarker, greenMarker, pinkMarker;
Mat yellowMat, greenMat, pinkMat, blueMat;
bool lostCount[] = {false, false, false, false};

void on_change () {
    Mat HSVImage;
    Mat processedImage;
    
    cvtColor(img1, HSVImage, CV_BGR2HSV); //convert image to HSV and save into HSVImage
    inRange(HSVImage, Scalar(h_min_slider,s_min_slider,v_min_slider),
            Scalar(h_max_slider,s_max_slider,v_max_slider), processedImage);
    
    imshow("Processed Image", processedImage);
    
    cout << "Min :: " << h_min_slider << " :: " << s_min_slider << " :: " << v_min_slider << endl;
    cout << "Max :: " << h_max_slider << " :: " << s_max_slider << " :: " << v_max_slider << endl;
    waitKey(30);
}

static void on_Hchange (int value,void *userData) {
    h_min_slider = value;
    on_change();
}

static void on_Schange (int value, void *userData) {
    s_min_slider = value;
    on_change();
}

static void on_Vchange (int value, void *userData) {
    v_min_slider = value;
    on_change();
}

static void on_max_Hchange (int value,void *userData) {
    h_max_slider = value;
    on_change();
}

static void on_max_Schange (int value, void *userData) {
    s_max_slider = value;
    on_change();
}

static void on_max_Vchange (int value, void *userData) {
    v_max_slider = value;
    on_change();
}

void showTrackbarHSV(){
    namedWindow("Trackbars",1);
    createTrackbar("max_H", "Trackbars", 0, 255,on_max_Hchange,&h_max_slider);
    createTrackbar("max_S", "Trackbars", 0, 255,on_max_Schange,&s_max_slider);
    createTrackbar("max_V", "Trackbars", 0, 255,on_max_Vchange,&v_max_slider);
    createTrackbar("min_H", "Trackbars", 0, 255,on_Hchange,&h_min_slider);
    createTrackbar("min_S", "Trackbars", 0, 255,on_Schange,&s_min_slider);
    createTrackbar("min_V", "Trackbars", 0, 255,on_Vchange,&v_min_slider);
    waitKey(0);
}

vector<string> split(const std::string &text, char sep) {
    vector<string> tokens;
    size_t start = 0, end = 0;
    while ((end = text.find(sep, start)) != std::string::npos) {
        string temp = text.substr(start, end - start);
        if (temp != "") tokens.push_back(temp);
        start = end + 1;
    }
    string temp = text.substr(start);
    if (temp != "") tokens.push_back(temp);
    return tokens;
}

void getMarkerPos(Mat img1, Mat img2, vector<Point> &marker){
    
    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
    Mat des1, des2;
    
    int minHessian = 600;
    
    Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create( minHessian );
//    Ptr<AKAZE> detector = AKAZE::create();
    detector->detect( img1, keypoints1 );
    detector->detect( img2, keypoints2 );
    
    detector->detectAndCompute(img1, Mat(), keypoints1, des1);
    detector->detectAndCompute(img2, Mat(), keypoints2, des2);
    
//    Mat keypointImg1, keypointImg2;
//    drawKeypoints(img1, keypoints1, keypointImg1, Scalar(0,0,255));
//    drawKeypoints(img2, keypoints2, keypointImg2);
//    imshow("asd", keypointImg1);
//    imshow("Keypoint", keypointImg2);
//    waitKey(0);
    
    FlannBasedMatcher matcher;
    vector< DMatch > matches;
    matcher.match( des1, des2, matches );
    
    double max_dist = 0; double min_dist = 100;
    
    for( size_t i = 0; i < des1.rows; i++ ){
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    
    vector< DMatch > good_matches;
    
    for( size_t i = 0; i < des1.rows; i++ ){
        if( matches[i].distance <= max(2*min_dist, 0.02) ){
            good_matches.push_back( matches[i]);
        }
    }
    

    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2,
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::DEFAULT );
    
    
    for( int i = 0; i < good_matches.size(); i++ ){
//        printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx );
        Point tmp((int) keypoints1[good_matches[i].queryIdx].pt.x,
                  (int) keypoints1[good_matches[i].queryIdx].pt.y);
        marker.push_back(tmp);
    }
    
//    imshow( "Good Matches", img_matches );
//    waitKey(0);
    
}

struct MARKERINFO{
    Point u;
    Point v;
    double distance;
};

vector<Point> getAllPointInColor(Mat color){
    Mat canny;
    vector<Point> points;
    vector<vector<Point> > contour;
    vector<Vec4i> hierarchy;
    
    Canny( color, canny, 100, 200, 3 );
    findContours( canny, contour, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    
    for(size_t i = 0; i < contour.size(); i++){
        Moments mom = moments(Mat(contour[i]));
        points.push_back(Point(mom.m10/mom.m00,mom.m01/mom.m00));
    }
    
    return points;
}

Point getNearestPoint(Point base, vector<Point> target, int maximumDistance){
    double minDistance = 99999;
    Point output;
    for(size_t i = 0; i < target.size(); i++){
        double distance = norm(base - target[i]);
        if(distance < minDistance && distance < maximumDistance){
            minDistance = distance;
            output = target[i];
        }
    }
    
    if(minDistance == 99999)
        return base;
    else
        return output;
}

Point findFourthMarker(vector<Point> _marker, Mat missingColor,int maximumDistance = 100){
    
    Point I;
    Point J;
    Point K;
    
    MARKERINFO A;
    MARKERINFO B;
    MARKERINFO C;
    
    MARKERINFO farSet;
    MARKERINFO normalSet;
    MARKERINFO nearSet;
    
    A.u = _marker[0];
    A.v = _marker[1];
    A.distance = norm(A.u - A.v);
    
    B.u = _marker[1];
    B.v = _marker[2];
    B.distance = norm(B.u - B.v);

    C.u = _marker[0];
    C.v = _marker[2];
    C.distance = norm(C.u - C.v);
    
    vector<double> distanceVect;
    
    distanceVect.push_back(A.distance);
    distanceVect.push_back(B.distance);
    distanceVect.push_back(C.distance);
    
    sort(distanceVect.begin(), distanceVect.end());
    reverse(distanceVect.begin(), distanceVect.end());
    
    for(size_t i = 0; i < distanceVect.size(); i++){
        double _distance = distanceVect[i];
        
        if(i == 0){
            if(_distance == A.distance){
                farSet.u = A.u;
                farSet.v = A.v;
            }
            if(_distance == B.distance){
                farSet.u = B.u;
                farSet.v = B.v;
            }
            if(_distance == C.distance){
                farSet.u = C.u;
                farSet.v = C.v;
            }
        }
        if(i == 1){
            if(_distance == A.distance){
                normalSet.u = A.u;
                normalSet.v = A.v;
            }
            if(_distance == B.distance){
                normalSet.u = B.u;
                normalSet.v = B.v;
            }
            if(_distance == C.distance){
                normalSet.u = C.u;
                normalSet.v = C.v;
            }
        }
        if(i == 2){
            if(_distance == A.distance){
                nearSet.u = A.u;
                nearSet.v = A.v;
            }
            if(_distance == B.distance){
                nearSet.u = B.u;
                nearSet.v = B.v;
            }
            if(_distance == C.distance){
                nearSet.u = C.u;
                nearSet.v = C.v;
            }
        }
    }
    
    if(nearSet.u == normalSet.u || nearSet.u == normalSet.v){
        I = (nearSet.u == normalSet.u) ? normalSet.v : normalSet.u;
        J = nearSet.u;
        K = nearSet.v;
    }
    if(nearSet.v == normalSet.u || nearSet.v == normalSet.v){
        I = (nearSet.v == normalSet.u) ? normalSet.v : normalSet.u;
        J = nearSet.v;
        K = nearSet.u;
    }
    Point fourth = I + (K - J);
    
    vector<Point> points = getAllPointInColor(missingColor);

    if(points.size() > 0){
       fourth = getNearestPoint(fourth, points, maximumDistance);
    }
    return fourth;
}


vector<Point> findAccurateRectPoint(vector<Point> markerPos, int offset){
    
    vector<Rect> rectPoint;
    vector<Point> tmp;
    
    while(!markerPos.empty()){
        
        Point iPoint = markerPos.back(); markerPos.pop_back();
        int x = iPoint.x, y = iPoint.y;
        
        Rect iArea(iPoint.x - offset, iPoint.y - offset, offset * 2, offset * 2);
        rectPoint.push_back(iArea);
        tmp.push_back(iPoint);
        
        if(markerPos.empty())
            continue;
        
        rectPoint.pop_back();
        tmp.pop_back();
        vector<Point> jPoint;
        
        //Select the points that are in the iArea
        //and delete it all
        for(size_t i = 0; i < markerPos.size(); i++){
            if( iArea.contains(markerPos[i]) ){
//                cout << iPoint << " contains: " << markerPos[i] << endl;
                jPoint.push_back(markerPos[i]);
                markerPos.erase(markerPos.begin() + i);
                
            }
        }
        
        //Delete duplicate Point
        for(size_t i = 0; i < markerPos.size(); i++){
            for(size_t j = 0; j < markerPos.size(); j++){
                if(i == j) continue;
                if(markerPos[i].x == markerPos[j].x && markerPos[i].y == markerPos[j].y){
                    markerPos.erase(markerPos.begin() + i);
                }
            }
            
        }
        
        for(size_t i = 0; i < jPoint.size(); i++){
            x += jPoint[i].x;
            y += jPoint[i].y;
        }
        
        x /= (jPoint.size() + 1);
        y /= (jPoint.size() + 1);
        
        Rect jArea(x - offset, y - offset, offset * 2, offset * 2);
        rectPoint.push_back(jArea);
        tmp.push_back(Point(x,y));
        
//        cout << "Next " << endl;
    }
    
//        for(size_t i = 0; i < rectPoint.size(); i++){
//            rectangle(img1, rectPoint[i], Scalar(0,0,255));
//        }
//        imshow("result", img1);
//        waitKey(0);
    
    return tmp;
}

void getScore(vector<Point> markerPos, vector<Point> score[], int minArea = 2000, int offset = 10){
    
    for (size_t i = 0; i < markerPos.size(); i++){
        //check green point
        Mat green_(greenMat, Rect(markerPos[i].x - offset, markerPos[i].y - offset, offset * 2 , offset * 2));
        Mat blue_(blueMat, Rect(markerPos[i].x - offset, markerPos[i].y - offset, offset * 2 , offset * 2));
        Mat yellow_(yellowMat, Rect(markerPos[i].x - offset, markerPos[i].y - offset, offset * 2 , offset * 2));
        Mat pink_(pinkMat, Rect(markerPos[i].x - offset, markerPos[i].y - offset, offset * 2 , offset * 2));
        
        if(cv::sum( yellow_ )[0] > minArea ){
            score[YELLOW].push_back(markerPos[i]);
        }
        
        if(cv::sum( blue_ )[0] > minArea){
            score[BLUE].push_back(markerPos[i]);
        }
        
        if(cv::sum( green_ )[0] > minArea){
            score[GREEN].push_back(markerPos[i]);
        }
        
        if(cv::sum( pink_ )[0] > minArea ){
            score[PINK].push_back(markerPos[i]);
        }
    }
}

Point getHighestPointScore(map<string, int> pointScore, int color){
    //if all points are not equal this function will return the point that has highest score
    //otherwise this function will return the point with highest area in the color
    int max = 0, round = 1, prevScore = 0;
    bool isAllEqual = true;
    string maxString = "";
    typedef map<string, int>::iterator it_type;
    for(it_type iterator = pointScore.begin(); iterator != pointScore.end(); iterator++, round++) {
        // iterator->first = key
        // iterator->second = value
        if(iterator->second > max){
            max = iterator->second;
            maxString = iterator->first;
        }
        
        cout << iterator->first << " " << iterator->second << endl;
        
        //to check whether all scores are equal
        if(round == 1){
            prevScore = iterator->second;
            continue;
        }
        
        if(prevScore != iterator->second)
            isAllEqual = false;
        
    }
    
    if(isAllEqual){
        Mat colorMat;
        switch (color) {
            case 0:
                colorMat = yellowMat;
                break;
            case 1:
                colorMat = blueMat;
                break;
            case 2:
                colorMat = greenMat;
                break;
            case 3:
                colorMat = pinkMat;
                break;
            default: break;
        }
        
        int maxArea = 0;
        Point maxPoint;

        //Find the point that has maximum area
        for(it_type iterator = pointScore.begin(); iterator != pointScore.end(); iterator++) {
            
            vector<string> tmp =  split(iterator->first, '|');
            Point p(stoi(tmp[0]), stoi(tmp[1]));
            
            Mat colorArea_(colorMat, Rect(p.x - 10, p.y - 10,20,20));
            if(sum( colorArea_ )[0] > maxArea){
                maxArea = sum( colorArea_ )[0];
                maxPoint = p;
            }
        }

        return maxPoint;
    }
    
        
    vector<string> tmp =  split(maxString, '|');
    return Point(stoi(tmp[0]), stoi(tmp[1]));
}


string getPosString(Point p){
    return to_string(p.x) + "|" + to_string(p.y);
}

void setMarkerByScore(const vector<Point> score[]){
    Point marker[4];
    for(int i = 0; i < 4; i++){
        map<string,int> pointScore;
        
        for(size_t j = 0; j < score[i].size(); j++){
            pointScore[getPosString(score[i][j])] += 1;
        }
        
        //if the color has no score then ignored
        if(score[i].size() == 0)
            continue;
        
        marker[i] = getHighestPointScore(pointScore, i);
    }
    yellowMarker = marker[YELLOW];
    blueMarker = marker[BLUE];
    greenMarker = marker[GREEN];
    pinkMarker = marker[PINK];
}

void setMarkerByMaximumArea(vector<Point> markerPos,int minArea = 1000,int offset = 10){
    double yellowSum = 0, blueSum = 0, greenSum = 0, pinkSum = 0;
    
    for (size_t i = 0; i < markerPos.size(); i++){
        //check green point
        Mat green_(greenMat, Rect(markerPos[i].x - offset, markerPos[i].y - offset, offset * 2 , offset * 2));
        Mat blue_(blueMat, Rect(markerPos[i].x - offset, markerPos[i].y - offset, offset * 2 , offset * 2));
        Mat yellow_(yellowMat, Rect(markerPos[i].x - offset, markerPos[i].y - offset, offset * 2 , offset * 2));
        Mat pink_(pinkMat, Rect(markerPos[i].x - offset, markerPos[i].y - offset, offset * 2 , offset * 2));
        
        if(cv::sum( yellow_ )[0] > minArea && cv::sum( yellow_ )[0] > yellowSum){
            yellowSum = cv::sum( yellow_ )[0];
            yellowMarker = markerPos[i];
            cout << "yellow: " << yellowMarker << endl;
        }
        
        if(cv::sum( blue_ )[0] > minArea && cv::sum( blue_ )[0] > blueSum){
            blueSum = cv::sum( blue_ )[0];
            blueMarker = markerPos[i];
            cout << "blue: " << blueMarker << endl;
        }
        
        if(cv::sum( green_ )[0] > minArea && cv::sum( green_ )[0] > greenSum){
            greenSum = cv::sum( green_ )[0];
            greenMarker = markerPos[i];
            cout << "green: " << greenMarker << endl;
        }
        
        if(cv::sum( pink_ )[0] > minArea && cv::sum( pink_ )[0] > pinkSum){
            pinkSum = cv::sum( pink_ )[0];
            pinkMarker = markerPos[i];
            cout << "pink: " << pinkMarker << endl;
        }
    }
}

void showScore(vector<Point> score[]){
    for(size_t i = 0; i < 4; i++){
        cout << score[i].size() << " ";
    }
    cout << endl;
}

int getSumArea(Mat m){
    int sum = 0;
    for(int i = 0; i < m.rows; i++)
    for(int j = 0; j < m.cols; i++)
        sum += (int) m.at<uchar>(j, i);
    
    return sum;
}

vector<Point> getMatchedMarker(vector<Point> marker){
    vector<Point> output;
    for(int i = 0; i < 4; i++){
        if(marker[i].x != 0 && marker[i].y != 0)
            output.push_back(marker[i]);
    }
    return output;
}

vector<Mat> getUnmatchedColor(vector<Point> marker, map<int,Mat> color){
    vector<Mat> output;
    map<int, bool> checkColor;
    checkColor[YELLOW] = false;
    checkColor[BLUE] = false;
    checkColor[GREEN] = false;
    checkColor[PINK] = false;
    
    for(int i = 0; i < marker.size(); i++){
        if(marker[i].x == 0 && marker[i].y == 0)
            continue;
        
        if(sum(Mat(color[YELLOW], Rect(marker[i].x - 5, marker[i].y - 5, 10,10)))[0] > 0)
            checkColor[YELLOW] = true;
        if(sum(Mat(color[BLUE], Rect(marker[i].x - 5, marker[i].y - 5, 10,10)))[0] > 0)
            checkColor[BLUE] = true;
        if(sum(Mat(color[GREEN], Rect(marker[i].x - 5, marker[i].y - 5, 10,10)))[0] > 0)
            checkColor[GREEN] = true;
        if(sum(Mat(color[PINK], Rect(marker[i].x - 5, marker[i].y - 5, 10,10)))[0] > 0)
            checkColor[PINK] = true;
    }
    
    if(!checkColor[YELLOW])
        output.push_back(color[YELLOW]);
    if(!checkColor[BLUE])
        output.push_back(color[BLUE]);
    if(!checkColor[GREEN])
        output.push_back(color[GREEN]);
    if(!checkColor[PINK])
        output.push_back(color[PINK]);
    
    cout << output.size() << endl;
    return output;
}

int main(int argc, const char * argv[]) {
    
    bool setByScore = true;
    bool isRectangle = true;

    for(int k = 1; k < 30; k++){
        
    //    img1 = imread(DataManager::getInstance().FULL_PATH_PHOTO + "Marking_day3/3s.JPG");
    img1 = imread(DataManager::getInstance().FULL_PATH_PHOTO + "Marking4/" + to_string(k) +".png");
    
    Mat marker1 = imread(DataManager::getInstance().FULL_PATH_PHOTO + "Marker/marker1.png");
    Mat marker2 = imread(DataManager::getInstance().FULL_PATH_PHOTO + "Marker/marker2.png");
    Mat marker3 = imread(DataManager::getInstance().FULL_PATH_PHOTO + "Marker/marker3.png");
    Mat marker4 = imread(DataManager::getInstance().FULL_PATH_PHOTO + "Marker/marker4.png");
    
    vector<Point> markerPos;
    vector<Point> posToDel;
    
    
    Mat HSVImage;
    cvtColor(img1, HSVImage, CV_BGR2HSV); //convert image to HSV and save into HSVImage
    
    
    
//    showTrackbarHSV();
    
    //using 2 ranges for green marker
    Mat _greenMat;
    
    inRange(HSVImage, Scalar(46,52,197), Scalar(81,255,255), _greenMat);
    inRange(HSVImage, Scalar(46,131,177), Scalar(81,255,255), greenMat);
    
    inRange(HSVImage, Scalar(94,127,228), Scalar(116,255,255), blueMat);
    
//    inRange(HSVImage, Scalar(59,71,240), Scalar(116,255,255), blueMat);
    inRange(HSVImage, Scalar(114,45,229), Scalar(153,255,255), pinkMat);
    inRange(HSVImage, Scalar(26,34,204), Scalar(40,255,255), yellowMat);
    
    greenMat += _greenMat;
    
    medianBlur(greenMat, greenMat, 3);
    medianBlur(blueMat, blueMat, 3);
    medianBlur(pinkMat, pinkMat, 3);
    medianBlur(yellowMat, yellowMat, 3);
    
    Mat output = greenMat + blueMat + pinkMat + yellowMat;
//    Mat output = pinkMat;

    
    if(setByScore){
        vector<Point> score[4];
        
        getMarkerPos(img1, marker1,markerPos);
        markerPos = findAccurateRectPoint(markerPos, 10);
        getScore(markerPos, score,2000);
        showScore(score);
        
        getMarkerPos(img1, marker2,markerPos);
        markerPos = findAccurateRectPoint(markerPos, 10);
        getScore(markerPos, score,2000);
        showScore(score);
        
        getMarkerPos(img1, marker3,markerPos);
        markerPos = findAccurateRectPoint(markerPos, 10);
        getScore(markerPos, score,2000);
        showScore(score);
        
        getMarkerPos(img1, marker4,markerPos);
        markerPos = findAccurateRectPoint(markerPos, 10);
        getScore(markerPos, score,2000);
        showScore(score);
        
        setMarkerByScore(score);
    }
    else{
        getMarkerPos(img1, marker1,markerPos);
        getMarkerPos(img1, marker2,markerPos);
        getMarkerPos(img1, marker3,markerPos);
        getMarkerPos(img1, marker4,markerPos);
        
        markerPos = findAccurateRectPoint(markerPos, 10);
        setMarkerByMaximumArea(markerPos);
    }
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
//    for(size_t i = 0; i < markerPos.size(); i++){
//        circle(greenMat, markerPos[i], 10, Scalar(0,0,255));
//    }
    
    
    
    
//    findFourthMarker()
    
//    Scalar(255,255,0)
   
    
        
    vector<Point> matchedMarker_vect;
    matchedMarker_vect.push_back(yellowMarker);
    matchedMarker_vect.push_back(blueMarker);
    matchedMarker_vect.push_back(greenMarker);
    matchedMarker_vect.push_back(pinkMarker);
        
    map<int,Mat> colorMat_vect;
    colorMat_vect[YELLOW] = yellowMat;
    colorMat_vect[BLUE] = blueMat;
    colorMat_vect[GREEN] = greenMat;
    colorMat_vect[PINK] = pinkMat;
        
        
    matchedMarker_vect = getMatchedMarker(matchedMarker_vect);
    vector<Mat> missingColor = getUnmatchedColor(matchedMarker_vect, colorMat_vect);
        
    cvtColor(greenMat, greenMat, CV_GRAY2BGR);
    cvtColor(blueMat, blueMat, CV_GRAY2BGR);
    cvtColor(pinkMat, pinkMat, CV_GRAY2BGR);
    cvtColor(yellowMat, yellowMat, CV_GRAY2BGR);
    cvtColor(output, output, CV_GRAY2BGR);
        
    //show matched marker from surf
    for(size_t i = 0; i < matchedMarker_vect.size(); i++){
        circle(output, matchedMarker_vect[i], 10, Scalar(0,0,255));
        circle(img1, matchedMarker_vect[i], 10, Scalar(0,0,255));
    }
        
    if(matchedMarker_vect.size() == 3){
        int maximumDistance = 200;
        Point fourthMaker = findFourthMarker(matchedMarker_vect,missingColor[0], maximumDistance);
        matchedMarker_vect.push_back(fourthMaker);
        
        //show matched marker from the forth point approximation
        circle(output, fourthMaker, maximumDistance, Scalar(255,255,0));
        circle(output, fourthMaker, 2, Scalar(255,255,0));
        circle(img1, fourthMaker, maximumDistance, Scalar(255,255,0));
    }
        
//        vector<Point> points = getAllPointInColor(yellowMat);
//        
//        for(size_t i = 0; i < points.size(); i++){
//            circle(output, points[i], 10, Scalar(0,0,255));
//            circle(img1, points[i], 10, Scalar(0,0,255));
//        }

        
    imshow("res", img1);
    imshow("binary", output);
    waitKey(0);
    
    }
    
}