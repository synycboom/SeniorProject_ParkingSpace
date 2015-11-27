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

#define DICTIONARY_BUILD 0
#define MODE 2
#define MAX_CAR 114
#define MAX_NOT_CAR 107
using namespace cv;
using namespace std;
using namespace ml;

const float WIDTH = 380;
const float HEIGHT = 204;


////////////////////////////////////// Image Registration //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
void imageRegistrator(){
    VideoCapture cap(DataManager::getInstance().FULL_PATH_VIDEO + "point.mp4");
    Mat img;
    Mat origin;
    Mat output;
    namedWindow("input");
    namedWindow("output");
    while(1){
        cap.read(img);
        img.copyTo(origin);
        cvtColor(img, img, CV_BGR2HSV);
        Mat yellowMat, greenMat, redMat, blueMat;
        
        //yellow
        inRange(img, Scalar(22,100,100), Scalar(38,255,255), yellowMat);
        //blue have 2 point problem
        inRange(img, Scalar(100,25,25), Scalar(130,255,255), blueMat);
        //red
        inRange(img, Scalar(0,50,50), Scalar(10,255,255), redMat);
        //green
        inRange(img, Scalar(38,50,50), Scalar(75,255,255), greenMat);
        
        medianBlur(yellowMat, yellowMat, 11);
        medianBlur(blueMat, blueMat, 11);
        medianBlur(redMat, redMat, 11);
        medianBlur(greenMat, greenMat, 11);
        
        Mat canny_yellow,canny_red,canny_green,canny_blue;
        vector<vector<Point> > contour_yellow,contour_red,contour_green,contour_blue;
        vector<Vec4i> hierarchy;
        
        /// Detect edges using canny
        Canny( yellowMat, canny_yellow, 100, 200, 3 );
        Canny( blueMat, canny_red, 100, 200, 3 );
        Canny( redMat, canny_green, 100, 200, 3 );
        Canny( greenMat, canny_blue, 100, 200, 3 );
        /// Find contours
        findContours( canny_yellow, contour_yellow, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        findContours( canny_red, contour_red, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        findContours( canny_green, contour_green, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        findContours( canny_blue, contour_blue, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        
        cv::Point centoid_yellow,centoid_red,centoid_green,centoid_blue;
        
        cv::Moments momYellow= cv::moments(cv::Mat(contour_yellow[0]));
        cv::Moments momRed= cv::moments(cv::Mat(contour_red[0]));
        cv::Moments momGreen= cv::moments(cv::Mat(contour_green[0]));
        cv::Moments momBlue= cv::moments(cv::Mat(contour_blue[0]));
        
        //        cv::circle(origin,
        //                   Point(momYellow.m10/momYellow.m00,momYellow.m01/momYellow.m00),
        //                   2,cv::Scalar(255),2);
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
        
        source_points[0] = centoid_red; // bottom-left
        source_points[1] = centoid_blue; //top-left
        source_points[2] = centoid_green; //bottom-right
        source_points[3] = centoid_yellow; //top-right
        
        dest_points[0] = Point(0,0);
        dest_points[1] = Point(maxWidth - 1,0);
        dest_points[2] = Point(maxWidth - 1, maxHeight - 1);
        dest_points[3] = Point(0, maxHeight);
        
        Mat m = getPerspectiveTransform(source_points, dest_points);
        warpPerspective(origin, output, m, Size(maxWidth, maxHeight) );
        imshow("output", output);
        imshow("input" , origin);
        waitKey(10);
    }

}

//////////////////////////////////////// Main Feature /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
void mainFeature(){
    /// Using 50 bins for hue and 60 for saturation
    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };
    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    // Use the o-th and 1-st channels
    int channels[] = { 0, 1 };
    MatND hist_base;
    
    Mat mask6(HEIGHT, WIDTH, CV_8UC1, Scalar(0));
    rectangle(mask6,
              cv::Point(100, 100),
              cv::Point(170, 150),
              cv::Scalar(255, 255, 255),
              CV_FILLED
              );
    
    Rect cropRect(100,100,70,50);
    
    namedWindow("CropFrame");
    namedWindow("input");
    namedWindow("output");
    
    Mat cropFrame(HEIGHT, WIDTH, CV_8UC3, Scalar(0));
    Mat copyInput(HEIGHT, WIDTH, CV_8UC3, Scalar(0));
    Mat output(204, 380, CV_8UC3, Scalar(0));
    
    string vidPath = DataManager::getInstance().FULL_PATH_VIDEO + "dumb1.mp4";
    VideoCapture cap(vidPath);
    
    double comparedResult = 1;
    bool isFirstFrame = true;
    bool isCarOccupied = false;
    Scalar colorGreen(0,200,0);
    Scalar colorRed(0,0,200);
    
    while(1){
        Mat frame;
        cap.read(frame);
        frame.copyTo(copyInput);
        
        frame(cropRect).copyTo(cropFrame);
        imshow("CropFrame", cropFrame);
        
        
        if(isFirstFrame){
            cvtColor( frame, frame, COLOR_BGR2HSV );
            /// Calculate the histograms for the HSV images
            calcHist( &frame, 1, channels, mask6, hist_base, 2, histSize, ranges, true, false );
            normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );
            isFirstFrame = false;
            continue;
        }
        
        MatND hist_target;
        
        cvtColor( frame, frame, COLOR_BGR2HSV );
        /// Calculate the histograms for the HSV images
        calcHist( &frame, 1, channels, mask6, hist_target, 2, histSize, ranges, true, false );
        normalize( hist_target, hist_target, 0, 1, NORM_MINMAX, -1, Mat() );
        
        //initial output string
        stringstream ss[4];
        rectangle(output, cv::Point(0, 0), cv::Point(380,204),
                  cv::Scalar(255,255,255), -1);
        
        
        
        /// Apply the histogram comparison methods
        for( int compare_method = 0; compare_method < DataManager::getInstance().compareMethod.size(); compare_method++ )
        {
            comparedResult = compareHist( hist_base, hist_target, compare_method );
            
            ss[compare_method] << DataManager::getInstance().compareMethod[compare_method] << ": "
            << to_string(comparedResult);
        }
        
        comparedResult = compareHist( hist_base, hist_target, 0 );
        //smooth jitter
        //        comparedResult = comparedResult * (0.2) + (0.8) * compareHist( hist_base, hist_target, 0 );
        
        string frameNumberString1 = ss[0].str();
        string frameNumberString2 = ss[1].str();
        string frameNumberString3 = ss[2].str();
        string frameNumberString4 = ss[3].str();
        
        putText(output, frameNumberString1.c_str(), cv::Point(15, 20),
                FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
        putText(output, frameNumberString2.c_str(), cv::Point(15, 40),
                FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
        putText(output, frameNumberString3.c_str(), cv::Point(15, 60),
                FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
        putText(output, frameNumberString4.c_str(), cv::Point(15, 80),
                FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
        
        if(comparedResult <= 0.45)
        isCarOccupied = true;
        else
        isCarOccupied = false;
        
        if(isCarOccupied)
        rectangle(copyInput,
                  cv::Point(100, 100),
                  cv::Point(170, 150),
                  colorRed,
                  CV_FILLED
                  );
        else
        rectangle(copyInput,
                  cv::Point(100, 100),
                  cv::Point(170, 150),
                  colorGreen,
                  CV_FILLED
                  );
        
        imshow("input", copyInput);
        imshow("output", output);
        
        waitKey(30);
    }
}


void testFeature(){
    string imgPath1 = DataManager::getInstance().FULL_PATH_PHOTO + "boom.jpg";
    string imgPath2 = DataManager::getInstance().FULL_PATH_PHOTO + "boom_bg.jpg";

    namedWindow("ORB");
    namedWindow("test");
    namedWindow("first");
    namedWindow("output");
    
//    Mat img_1 = imread(imgPath1);
    Mat img_1;
//    VideoCapture cap(0);
//    cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
//    cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);
    
    string vidPath = DataManager::getInstance().FULL_PATH_VIDEO + "dumb1.mp4";
    VideoCapture cap(vidPath);
    
    bool isFirstFrame = true;
    
//    Ptr<Feature2D> orb =  xfeatures2d::SIFT::create();
    Ptr<Feature2D> orb = ORB::create();
    std::vector<KeyPoint> keypoints_1, keypoints_2, test;
    BFMatcher matcher;
    Mat descriptors_1, descriptors_2;
    
    Rect cropRect(90,100,90,100);
    
    bool isCarOccupied = false;
    int firstFrameCount = 0;
    Scalar colorGreen(0,200,0);
    Scalar colorRed(0,0,200);
    
    Mat fObj;
    Mat bg;
    Mat out1;
    Mat out;
    while(1){
        cap.read(bg);
        
        if(isFirstFrame){
            cap.read(img_1);
            img_1(cropRect).copyTo(fObj);
            orb->detect( fObj, keypoints_1 );
            orb->detect(img_1, test);
//            orb->compute( fObj, keypoints_1, descriptors_1 );
            drawKeypoints(fObj, keypoints_1, out1);
            Mat qq;
            drawKeypoints(img_1, test, qq);
            imshow("test",qq);
            imshow("first", out1);
            isFirstFrame = false;
            
            for(vector<int>::size_type i = 0; i != keypoints_2.size(); i++) {
                firstFrameCount++;
            }
            continue;
        }
        
    
        
        orb->detect( bg, keypoints_2 );
//        orb->compute( bg, keypoints_2, descriptors_2 );
        
        
//        std::vector< DMatch > matches;
//        matcher.match( descriptors_1, descriptors_2, matches );
//        
//        nth_element(matches.begin(), matches.begin()+24, matches.end());
//        matches.erase(matches.begin() + 25, matches.end());
//        
//        drawMatches(img_1, keypoints_1, bg, keypoints_2, matches, out);
        
        int count = 0;
        for(vector<int>::size_type i = 0; i != keypoints_2.size(); i++) {
            if(keypoints_2[i].pt.x >= 90 && keypoints_2[i].pt.x <= 180 && keypoints_2[i].pt.y >= 100 && keypoints_2[i].pt.y <= 200)
                count++;
        }
        
        drawKeypoints(bg, keypoints_2, out);
//        cout << to_string(count) << endl;
        
        if((count - firstFrameCount) > 30)
            isCarOccupied = true;
        else
            isCarOccupied = false;
        
        if(isCarOccupied)
            rectangle(bg,
                      cv::Point(100, 100),
                      cv::Point(170, 150),
                      colorRed,
                      CV_FILLED
                      );
        else
            rectangle(bg,
                      cv::Point(100, 100),
                      cv::Point(170, 150),
                      colorGreen,
                      CV_FILLED
                      );
        
        imshow("ORB", out);
        waitKey(30);
    }
}

void showFastCorner(string pname){
    Mat out;
    Mat img;
    namedWindow("FAST");
    string imgPath1 = DataManager::getInstance().FULL_PATH_PHOTO + pname;
    img = imread(imgPath1);
    Mat output(img.rows, img.cols, CV_8UC1, Scalar(255));
    cvtColor( img, img, CV_BGR2GRAY );
    vector<KeyPoint> keypoints;
    FAST(img,keypoints,10,true);
    drawKeypoints(output, keypoints, output);
    imshow( "FAST", output );
    waitKey(0);
}


void showSurfFeature(string pname){
    Mat out;
    Mat img;
    namedWindow("SURF");
    string imgPath1 = DataManager::getInstance().FULL_PATH_PHOTO + pname;
    img = imread(imgPath1);

    Ptr<Feature2D> surf =  xfeatures2d::SURF::create();
    
    //-- Step 1: Detect the keypoints:
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    surf->detect( img, keypoints_1 );
    
    //-- Step 2: Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    surf->compute( img, keypoints_1, descriptors_1 );

    drawKeypoints(img, keypoints_1, out);
    imshow("SURF", out);
    
    waitKey(0);
    
}

void showSiftFeature(string pname){
    Mat out;
    Mat img;
    Mat distance;
    namedWindow("SIFT");
    string imgPath1 = DataManager::getInstance().FULL_PATH_PHOTO +pname;
    img = imread(imgPath1);
    
//    cvtColor(img, img, CV_BGR2GRAY);
//    bitwise_not ( img, img );
//    threshold( img, img, 185, 255,CV_THRESH_BINARY );
    
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
    
    //-- Step 1: Detect the keypoints:
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    sift->detect( img, keypoints_1 );
    
    //-- Step 2: Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    sift->compute( img, keypoints_1, descriptors_1 );
    
    drawKeypoints(img, keypoints_1, out);
    imshow("SIFT", out);
    waitKey(0);
}

void showORBFeature(string pname){
    Mat out;
    Mat img;
    namedWindow("ORB");

    string imgPath1 = DataManager::getInstance().FULL_PATH_PHOTO + pname;
    img = imread(imgPath1);

    
    Ptr<Feature2D> orb = ORB::create();
    
    //-- Step 1: Detect the keypoints:
    std::vector<KeyPoint> keypoints_1, keypoints_2, tmpKey;
    orb->detect( img, keypoints_1 );
    
    //-- Step 2: Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    orb->compute( img, keypoints_1, descriptors_1 );
    
    drawKeypoints(img, keypoints_1, out);
    imshow("ORB", out);

    
    waitKey(0);
}


vector<KeyPoint> getFastKeyPoint(string pname){
    Mat img;
    string imgPath = DataManager::getInstance().FULL_PATH_PHOTO + pname;
    img = imread(imgPath);
    cvtColor( img, img, CV_BGR2GRAY );
    vector<KeyPoint> keypoints;
    FAST(img,keypoints,10,true);
    return keypoints;
}

vector<KeyPoint> getFastKeyPoint_tmp(Mat img){
    cvtColor( img, img, CV_BGR2GRAY );
    vector<KeyPoint> keypoints;
    FAST(img,keypoints,10,true);
    return keypoints;
}
vector<KeyPoint> getSiftKeyPoint_tmp(Mat img){
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
    std::vector<KeyPoint> keypoints;
    sift->detect( img, keypoints);
    return keypoints;
}

vector<KeyPoint> getSiftKeyPoint(string pname){
    Mat img;
    string imgPath = DataManager::getInstance().FULL_PATH_PHOTO + pname;
    img = imread(imgPath);
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
    std::vector<KeyPoint> keypoints;
    sift->detect( img, keypoints);
    
    return keypoints;
}

vector<KeyPoint> getSurfKeyPoint(string pname){
    Mat img;
    string imgPath = DataManager::getInstance().FULL_PATH_PHOTO + pname;
    img = imread(imgPath);
    Ptr<Feature2D> surf = xfeatures2d::SURF::create();
    std::vector<KeyPoint> keypoints;
    surf->detect( img, keypoints);
    
    return keypoints;
}

vector<KeyPoint> getOrbKeyPoint(string pname){
    Mat img;
    string imgPath = DataManager::getInstance().FULL_PATH_PHOTO + pname;
    img = imread(imgPath);
    Ptr<Feature2D> orb = ORB::create();
    std::vector<KeyPoint> keypoints;
    orb->detect( img, keypoints);
    
    return keypoints;
}

void predictionStep(){
    //prepare BOW descriptor extractor from the dictionary
    Mat dictionary;
    FileStorage fs("dictionary.yml", FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs.release();
    
    Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create();
    Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();
    Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create("FlannBased");
    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowDE(extractor,matcher);
    //Set the dictionary with the vocabulary we created in the first step
    bowDE.setVocabulary(dictionary);
    
    //open the file to write the resultant descriptor
    FileStorage allDes("all-descriptor.yml", FileStorage::WRITE);
    
    string extension = ".jpg";
    Mat trainingSet;
    Mat labelsMat;
    
    //positive training set
    
    string carName = DataManager::getInstance().FULL_PATH_PHOTO + "/car_set/car";
    for (int index = 1; index <= MAX_CAR - 40; index++) {
        string picName = carName + to_string(index) + extension;
        Mat input = imread(picName, CV_LOAD_IMAGE_GRAYSCALE);
        
        //To store the keypoints that will be extracted by SIFT
        vector<KeyPoint> keypoints;
        //Detect SIFT keypoints (or feature points)
        detector->detect(input,keypoints);
        //To store the BoW (or BoF) representation of the image
        Mat bowDescriptor;
        //extract BoW (or BoF) descriptor from given image
        bowDE.compute2(input,keypoints,bowDescriptor);
        trainingSet.push_back(bowDescriptor);
        labelsMat.push_back(1);
    }
    
    //negative training set
    
    string floorName = DataManager::getInstance().FULL_PATH_PHOTO + "/not_car_set/notcar";
    Mat tranningNegative;
    for (int index = 1; index <= MAX_NOT_CAR; index++) {
        string picName = floorName + to_string(index) + extension;
        Mat input = imread(picName, CV_LOAD_IMAGE_GRAYSCALE);
        
        //To store the keypoints that will be extracted by SIFT
        vector<KeyPoint> keypoints;
        //Detect SIFT keypoints (or feature points)
        detector->detect(input,keypoints);
        //To store the BoW (or BoF) representation of the image
        Mat bowDescriptor;
        //extract BoW (or BoF) descriptor from given image
        bowDE.compute2(input,keypoints,bowDescriptor);
        trainingSet.push_back(bowDescriptor);
        labelsMat.push_back(-1);
    }
    

    //To store the image tag name - only for save the descriptor in a file
    char * imageTag = new char[10];
    
    //prepare the yml (some what similar to xml) file
    sprintf(imageTag,"all-descriptor");
    allDes << imageTag << trainingSet;
    //release the file storage
    allDes.release();
    
    ///////// train SVM ////////
    Ptr<SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setGamma(30);
    svm->setKernel(ml::SVM::RBF);
    svm->setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
    svm->train(trainingSet, ml::ROW_SAMPLE, labelsMat);
    
    ///////// prediction /////////
//    ofstream myfile;
//    myfile.open ("RBF.txt");
//    
//    int falseNegative = 0, falsePositive = 0;
//    for (int index = 1; index <= MAX_CAR; index++) {
//        string targetPhoto = DataManager::getInstance().FULL_PATH_PHOTO + "/car_set/car"
//                            + to_string(index)+ extension;
//        Mat img=imread(targetPhoto, CV_LOAD_IMAGE_GRAYSCALE);
//        vector<KeyPoint> targetKeypoints;
//        detector->detect(img,targetKeypoints);
//        Mat targetBowDescriptor;
//        bowDE.compute2(img,targetKeypoints,targetBowDescriptor);
//        float response = svm->predict(targetBowDescriptor);
//        if(response == -1) falseNegative++;
//        myfile << "car" << to_string(index) << " = " << to_string(response) << endl;
//    }
//    for (int index = 1; index <= MAX_NOT_CAR; index++) {
//        string targetPhoto = DataManager::getInstance().FULL_PATH_PHOTO + "/not_car_set/notcar"
//                            + to_string(index) + extension;
//        Mat img=imread(targetPhoto, CV_LOAD_IMAGE_GRAYSCALE);
//        vector<KeyPoint> targetKeypoints;
//        detector->detect(img,targetKeypoints);
//        Mat targetBowDescriptor;
//        bowDE.compute2(img,targetKeypoints,targetBowDescriptor);
//        float response = svm->predict(targetBowDescriptor);
//        if(response == 1) falsePositive++;
//        myfile << "notcar" << to_string(index) << " = " << to_string(response) << endl;
//    }
//    myfile << endl << endl;
//    myfile << "false positive = " << to_string(falsePositive) << endl;
//    myfile << "false negative = " << to_string(falseNegative) << endl;
//    myfile.close();
    
    
    string named = "Predict Car";
    string targetPhoto = DataManager::getInstance().FULL_PATH_PHOTO + "/car_set/car4.jpg";
    Mat img=imread(targetPhoto, CV_LOAD_IMAGE_GRAYSCALE);
    vector<KeyPoint> targetKeypoints;
    detector->detect(img,targetKeypoints);
    Mat targetBowDescriptor;
    bowDE.compute2(img,targetKeypoints,targetBowDescriptor);
    float response = svm->predict(targetBowDescriptor);
    cout << "Predict: " << response << endl;
    imshow(named, img);
    waitKey(0);
    
}

int main(int argc, const char * argv[]) {
    
#if DICTIONARY_BUILD == 1
    
    vector<KeyPoint> siftKeypoints, surfKeypoints, orbKeypoints, fastKeypoints, parkingKeyPoints;
    
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
    
    string tmpName = DataManager::getInstance().FULL_PATH_PHOTO + "/car_set/car";
    string surname = ".jpg";
    Mat unclusteredDescriptors;
    for (int index = 1; index <= 114; index++) {
        Mat descriptor;
        vector<KeyPoint> keypoints;
        string picName = tmpName + to_string(index) + surname;
        Mat input = imread(picName, CV_LOAD_IMAGE_GRAYSCALE);
        keypoints = getSiftKeyPoint_tmp(input);
        sift->compute(input, keypoints, descriptor);
        unclusteredDescriptors.push_back(descriptor);
    }
    
    //Construct BOWKMeansTrainer
    //the number of bags
    int dictionarySize= 1000;
    //define Term Criteria
    TermCriteria tc(CV_TERMCRIT_ITER,100, 1e-6);
    //retries number
    int retries=1;
    //necessary flags
    int flags=KMEANS_PP_CENTERS;
    //Create the BoW (or BoF) trainer
    BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
    //cluster the feature vectors
    Mat dictionary = bowTrainer.cluster(unclusteredDescriptors);
    
    //store the vocabulary
    FileStorage fs("dictionary.yml", FileStorage::WRITE);
    fs << "vocabulary" << dictionary;
    fs.release();
    cout << "Dictionary created";
    
#else
    predictionStep();
    
//#if MODE == 1 //Run Thresholding
//    string window1 = "thresholding";
//    string window2 = "original";
//    Mat originCar = imread(DataManager::getInstance().FULL_PATH_PHOTO + "sc3.jpg",0);
//    namedWindow(window1);
//    imshow(window1,originCar);
//    threshold( originCar, originCar, 75, 255,1 );
//    imshow(window2, originCar);
//    waitKey(0);
//#else
//    imageRegistrator();
//     string window1 = "thresholding";
//    Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
//    string tmp = "car_set/car40.jpg";
//    Mat descriptor;
//    Mat img = imread(DataManager::getInstance().FULL_PATH_PHOTO + tmp);
//    vector<KeyPoint> keypoints;
//    keypoints = getSiftKeyPoint_tmp(img);
//    sift->compute(img, keypoints, descriptor);
//    cout << descriptor << endl;
//    FileStorage fs("des.yml", FileStorage::WRITE);
//    fs << "descriptor" << descriptor;
//    fs.release();

//    showSiftFeature(tmp);
//#endif

#endif
}