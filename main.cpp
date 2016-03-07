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
#include <algorithm>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace ml;
using namespace std;


/************************************************/
/********************* TODO *********************/
/***** set path variable to folder 'images' *****/
/************************************************/
/************************************************/

//global variable

//TODO set path here
const string path = "/Users/synycboom/Dropbox/SeniorProject/Photo/";
const char* extension[] = {".png", ".jpg"};
HOGDescriptor hog;
vector<float> descriptors;

//function prototype
bool replace(std::string& str, const std::string& from, const std::string& to);
void scanDir(string path, vector<string>* fileList);
Mat getSVMDesFromHOG(string filePath);

int main(int argc, const char * argv[]) {
    
    vector<string> posFileList, negFileList;;
    string posPath = path + "/car_set";
    string negPath = path + "/not_car_set";
    replace(posPath, "//", "/");
    replace(negPath, "//", "/");
    scanDir(posPath,&posFileList);
    scanDir(negPath,&negFileList);
    
    Mat allTraining;
    Mat labelsMat;
    
    //get hog descriptor from positive training set
    for (auto &str : posFileList){
        const char* fileName = str.c_str();
        if(strcspn(fileName, extension[0]) > 0 || strcspn(fileName, extension[1])){
            string filePath = posPath + "/" + fileName;
            //replace backslash if duplicated
            replace(filePath, "//", "/");
            
            allTraining.push_back(getSVMDesFromHOG(filePath));
            labelsMat.push_back(1);
            
        }
        descriptors.clear();
    }
    
    
    //get hog descriptor from negative training set
    for (auto &str : negFileList){
        const char* fileName = str.c_str();
        if(strcspn(fileName, extension[0]) > 0 || strcspn(fileName, extension[1])){
            
            string filePath = negPath + "/" + fileName;
            //replace backslash if duplicated
            replace(filePath, "//", "/");
            
            allTraining.push_back(getSVMDesFromHOG(filePath));
            labelsMat.push_back(-1);
            
        }
        descriptors.clear();
    }
    
    // Using linear kernel
    Ptr<SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    
    // Train the SVM
    svm->train(allTraining, ml::ROW_SAMPLE, labelsMat);
    
    cout << "*********** Check the result from model with training set *******************" << endl;
    for(int i = 0; i < allTraining.rows; i++){
        float result = svm-> predict(allTraining.row(i));
        cout << "Predict: " << result << " From Class: " << labelsMat.row(i) << endl;
    }
    
    cout << "*********** Test Model with test set *******************" << endl;
    //testing model with a test img
    string testPosImg = path + "/car_set/car50.jpg";
    string testNegImg = path + "/not_car_set/notcar50.jpg";
    
    //replace '/' if duplicated
    replace(testPosImg, "//", "/");
    replace(testNegImg, "//", "/");
    
    cout << "result: " << svm-> predict(getSVMDesFromHOG(testPosImg)) << endl;
    cout << "result: " << svm-> predict(getSVMDesFromHOG(testNegImg)) << endl;

}

Mat getSVMDesFromHOG(string filePath){
    
    Mat img = imread(filePath, CV_LOAD_IMAGE_GRAYSCALE);
    resize(img, img, hog.winSize);
    hog.compute(img, descriptors);
    Mat descriptorMat(1, descriptors.size(), DataType<float>::type);
    
    for(int i = 0; i < descriptors.size(); i++){
        descriptorMat.at<float>(0,i) = descriptors[i];
    }
    return descriptorMat;
}

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
