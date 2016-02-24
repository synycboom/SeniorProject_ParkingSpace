//
//  main.cpp
//  ParkingSpace
//
//  Created by synycboom on 9/11/2558 BE.
//  Copyright © 2558 HOME. All rights reserved.
//

#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
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
#define SCALEUP 1
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

vector<string> getNextLineAndSplitIntoTokens(istream& str)
{
    vector<string> result;
    string line;
    getline(str,line);
    
    stringstream lineStream(line);
    string cell;
    
    while(getline(lineStream,cell,','))
    {
        result.push_back(cell);
    }
    return result;
}

void fragmentImage(vector<vector<string>> csv,string inputFile, string outputPath){
    Mat img = imread(inputFile);
    
    //Rect(x, y, width, height)
    for(int i = 0; i < csv.size(); i++){
        vector<string> eleArr = csv[i];
        
        Rect slotRect(stoi(eleArr[1]), stoi(eleArr[2]), stoi(eleArr[3]), stoi(eleArr[4]));
        Mat SlotImg = img(slotRect);
        resize(SlotImg, SlotImg, Size(SlotImg.cols * SCALEUP, SlotImg.rows * SCALEUP));
        string _outputPath = outputPath + eleArr[0] + ".jpg";
        replace(_outputPath, "//", "/");
        imwrite(_outputPath ,SlotImg);
    }
    
    

}

int main(int argc, const char * argv[]) {
    
    if(argc != 4){
        cout << "usage: (params) SlotPos_CSV IN_File Out_Path/" << endl;
        return 1;
    }
    string slotPosCsv = argv[1];
    string inputFile = argv[2];
    string outputPath = argv[3];

    std::ifstream file(slotPosCsv);
    
    vector<vector<string>> csv;
    vector<string> line = getNextLineAndSplitIntoTokens(file);
    while(line.size() != 0){
        csv.push_back(line);
        line = getNextLineAndSplitIntoTokens(file);
    }

    fragmentImage(csv, inputFile, outputPath);

}
