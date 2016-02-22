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

using namespace cv;
using namespace std;
using namespace ml;

const string PHOTO_PATH(DataManager::FULL_PATH_PHOTO);

void ls(string path, vector<string>* fileList){
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

vector<KeyPoint> getSiftKeyPoint_tmp(Mat img){
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
    std::vector<KeyPoint> keypoints;
    sift->detect( img, keypoints);
    return keypoints;
}

vector<KeyPoint> getSiftKeyPoint(string pname){
    Mat img;
    string imgPath = PHOTO_PATH + pname;
    img = imread(imgPath);
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
    std::vector<KeyPoint> keypoints;
    sift->detect( img, keypoints);
    
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
    
    string carName = PHOTO_PATH + "/car_set/car";
//    for (int index = TRAIN_MIN; index <= TRAIN_MAX; index++) {
//        if(index >= TEST_MIN && index <= TEST_MAX)
//            continue;
//        string picName = carName + to_string(index) + extension;
//        Mat input = imread(picName, CV_LOAD_IMAGE_GRAYSCALE);
//        cout << "Positive Train "<<picName << endl;
//        //To store the keypoints that will be extracted by SIFT
//        vector<KeyPoint> keypoints;
//        //Detect SIFT keypoints (or feature points)
//        detector->detect(input,keypoints);
//        //To store the BoW (or BoF) representation of the image
//        Mat bowDescriptor;
//        //extract BoW (or BoF) descriptor from given image
//        bowDE.compute2(input,keypoints,bowDescriptor);
//        trainingSet.push_back(bowDescriptor);
//        labelsMat.push_back(1);
//        
//        if(index == TRAIN_MIN){
//            cout << bowDescriptor << endl;
//        }
//    }
    
    //negative training set
    
    string floorName = PHOTO_PATH + "/not_car_set/notcar";
    Mat tranningNegative;
//    for (int index = TRAIN_MIN; index <= TRAIN_MAX; index++) {
//        if(index >= TEST_MIN && index <= TEST_MAX)
//            continue;
//        string picName = floorName + to_string(index) + extension;
//        Mat input = imread(picName, CV_LOAD_IMAGE_GRAYSCALE);
//        cout << "Negative Train "<<picName << endl;
//        //To store the keypoints that will be extracted by SIFT
//        vector<KeyPoint> keypoints;
//        //Detect SIFT keypoints (or feature points)
//        detector->detect(input,keypoints);
//        //To store the BoW (or BoF) representation of the image
//        Mat bowDescriptor;
//        //extract BoW (or BoF) descriptor from given image
//        bowDE.compute2(input,keypoints,bowDescriptor);
//        trainingSet.push_back(bowDescriptor);
//        labelsMat.push_back(-1);
//    }


    //To store the image tag name - only for save the descriptor in a file
    char * imageTag = new char[10];
    
    //prepare the yml (some what similar to xml) file
//    sprintf(imageTag,"all-descriptor");
//    allDes << imageTag << trainingSet;
//    //release the file storage
//    allDes.release();

    
    ///////// prediction /////////
//    ofstream myfile;
//    myfile.open (RESULT);
//    
//    int falseNegative = 0, falsePositive = 0;
//    for (int index = TEST_MIN; index <= TEST_MAX; index++) {
//        string targetPhoto = PHOTO_PATH + "/car_set/car"
//                            + to_string(index)+ extension;
//        
//        cout << "Positive Test "<< targetPhoto << endl;
//        Mat img=imread(targetPhoto, CV_LOAD_IMAGE_GRAYSCALE);
//        vector<KeyPoint> targetKeypoints;
//        detector->detect(img,targetKeypoints);
//        Mat targetBowDescriptor;
//        bowDE.compute2(img,targetKeypoints,targetBowDescriptor);
//        float response = svm->predict(targetBowDescriptor);
//        if(response == -1) falseNegative++;
//        myfile << "car" << to_string(index) << " = " << to_string(response) << endl;
//    }
//    for (int index = TEST_MIN ; index <= TEST_MAX; index++) {
//        string targetPhoto = PHOTO_PATH + "/not_car_set/notcar"
//                            + to_string(index) + extension;
//        
//        cout << "Negative Test "<< targetPhoto << endl;
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
    

    
}

int main(int argc, const char * argv[]) {
    

}
