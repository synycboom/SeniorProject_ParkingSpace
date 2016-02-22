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

#if numPath == 2
void featureExtract(string dictionaryFile, string positivePath, string negativePath, string outputPath){
    Mat dictionary;
    FileStorage fs(dictionaryFile, FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs.release();
    
    ofstream desFile;
    desFile.open (outputPath);
    
    
    Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create();
    Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();
    Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create("FlannBased");
    BOWImgDescriptorExtractor bowDE(extractor,matcher);
    
    const char* extension[] = {".png", ".jpg"};
    
    bowDE.setVocabulary(dictionary);
    
    Mat trainingSet;
    Mat labelsMat;
    Mat descriptor;
    vector<KeyPoint> keypoints;
    
    vector<string> posFileList, negFileList;;
    scanDir(positivePath,&posFileList);
    scanDir(negativePath,&negFileList);
    
    
    for (auto &str : posFileList){
        const char* fileName = str.c_str();
        if(strcspn(fileName, extension[0]) > 0 || strcspn(fileName, extension[1])){
            string filePath = positivePath + "/" + fileName;
            //replace backslash if duplicated
            replace(filePath, "//", "/");
            
            descriptor.release();
            keypoints.clear();
            
            Mat input = imread(filePath, CV_LOAD_IMAGE_GRAYSCALE);
            detector->detect(input,keypoints);
            bowDE.compute2(input,keypoints,descriptor);
            
            desFile << "+1";
            for(int i = 0; i < descriptor.cols ; i++){
                desFile << " " << i + 1 << ":" << descriptor.at<float>(0, i);
            }
            desFile << endl;
        }
    }
    
    for (auto &str : negFileList){
        const char* fileName = str.c_str();
        if(strcspn(fileName, extension[0]) > 0 || strcspn(fileName, extension[1])){
            string filePath = negativePath + "/" + fileName;
            //replace backslash if duplicated
            replace(filePath, "//", "/");
            
            descriptor.release();
            keypoints.clear();
            
            Mat input = imread(filePath, CV_LOAD_IMAGE_GRAYSCALE);
            detector->detect(input,keypoints);
            bowDE.compute2(input,keypoints,descriptor);
            
            desFile << "-1";
            for(int i = 0; i < descriptor.cols ; i++){
                desFile << " " << i + 1 << ":" << descriptor.at<float>(0, i);
            }
            desFile << endl;
        }
    }

    desFile.close();
    
}

#elif numPath == 1

void featureExtract(string dictionaryFile, string positivePath, string outputPath, string labelNum){
    Mat dictionary;
    FileStorage fs(dictionaryFile, FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs.release();
    
    ofstream desFile;
    desFile.open (outputPath);
    
    
    Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create();
    Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();
    Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create("FlannBased");
    BOWImgDescriptorExtractor bowDE(extractor,matcher);
    
    const char* extension[] = {".png", ".jpg"};
    
    bowDE.setVocabulary(dictionary);
    
    Mat trainingSet;
    Mat labelsMat;
    Mat descriptor;
    vector<KeyPoint> keypoints;
    
    vector<string> fileList;
    scanDir(positivePath,&fileList);
    
    
    for (auto &str : fileList){
        const char* fileName = str.c_str();
        if(strcspn(fileName, extension[0]) > 0 || strcspn(fileName, extension[1])){
            string filePath = positivePath + "/" + fileName;
            //replace backslash if duplicated
            replace(filePath, "//", "/");
            
            descriptor.release();
            keypoints.clear();
            
            Mat input = imread(filePath, CV_LOAD_IMAGE_GRAYSCALE);
            detector->detect(input,keypoints);
            bowDE.compute2(input,keypoints,descriptor);
            
            //set unknown label
            desFile << labelNum;
            for(int i = 0; i < descriptor.cols ; i++){
                desFile << " " << i + 1 << ":" << descriptor.at<float>(0, i);
            }
            desFile << endl;
        }
    }

    desFile.close();
    
}

#endif

int main(int argc, const char * argv[]) {
    
#if numPath == 2
    if(argc != 5){
        cout << "usage: (params) Dictionary_File Pos_Folder Neg_Folder output" << endl;
        return 1;
    }
    string dictionaryFile = argv[1];
    string positivePath = argv[2];
    string negativePath = argv[3];
    string outputPath = argv[4];
    
    featureExtract(dictionaryFile, positivePath, negativePath,outputPath);

#elif numPath == 1
    if(argc != 5){
        cout << "usage: (params) Dictionary_File Images_Folder Label_Number output" << endl;
        return 1;
    }
    string dictionaryFile = argv[1];
    string imagesPath = argv[2];
    string labelNum = argv[3];
    string outputPath = argv[4];
    
    featureExtract(dictionaryFile, imagesPath,outputPath,labelNum);
#endif
}
