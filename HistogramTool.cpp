//
//  HistogramTool.cpp
//  ParkingSpace
//
//  Created by synycboom on 9/15/2558 BE.
//  Copyright © 2558 HOME. All rights reserved.
//

#include "HistogramTool.hpp"
#include "DataManager.cpp"

using namespace std;
using namespace cv;



class HistogramTool{
public:
    
    
    
    void compare(){
        ofstream output;
        
        //compare car and floor
        for(int car = 0;car < DataManager::getInstance().carArr.size() ; car++){
            for(int floor = 0; floor < DataManager::getInstance().floorArr.size(); floor++){
                
                output.open ("floor" + to_string(floor + 1) + "_car"+ to_string(car + 1) + ".txt");
                
                Mat src_base, hsv_base;
                Mat src_test1, hsv_test1;
                
                src_base = imread(DataManager::getInstance().FULL_PATH_PHOTO + DataManager::getInstance().floorArr[floor]);
                src_test1 = imread(DataManager::getInstance().FULL_PATH_PHOTO + DataManager::getInstance().carArr[car]);
                
                /// Convert to HSV
                cvtColor( src_base, hsv_base, COLOR_BGR2HSV );
                cvtColor( src_test1, hsv_test1, COLOR_BGR2HSV );
                
                /// Using 50 bins for hue and 60 for saturation
                int h_bins = 50; int s_bins = 60;
                int histSize[] = { h_bins, s_bins };
                
                // hue varies from 0 to 179, saturation from 0 to 255
                float h_ranges[] = { 0, 180 };
                float s_ranges[] = { 0, 256 };
                
                const float* ranges[] = { h_ranges, s_ranges };
                
                // Use the o-th and 1-st channels
                int channels[] = { 0, 1 };
                
                
                /// Histograms
                MatND hist_base;
                MatND hist_test1;
                
                /// Calculate the histograms for the HSV images
                calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
                normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );
                
                calcHist( &hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false );
                normalize( hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat() );
                
                /// Apply the histogram comparison methods
                for( int compare_method = 0; compare_method < DataManager::getInstance().compareMethod.size(); compare_method++ )
                {
                    double base_test1 = compareHist( hist_base, hist_test1, compare_method );
                    
                    output << DataManager::getInstance().compareMethod[compare_method] << ": " <<base_test1 << endl;
                }
                
                output.close();
                
            }
        }
        
        
        //compare floor and floor
        for(int base_floor = 0;base_floor < DataManager::getInstance().floorArr.size() ; base_floor++){
            for(int floor = 0; floor < DataManager::getInstance().floorArr.size(); floor++){
                
                output.open ("floor" + to_string(base_floor + 1) + "_floor"+ to_string(floor + 1) + ".txt");
                
                Mat src_base, hsv_base;
                Mat src_test1, hsv_test1;
                
                src_base = imread(DataManager::getInstance().FULL_PATH_PHOTO + DataManager::getInstance().floorArr[base_floor]);
                src_test1 = imread(DataManager::getInstance().FULL_PATH_PHOTO + DataManager::getInstance().floorArr[floor]);
                
                /// Convert to HSV
                cvtColor( src_base, hsv_base, COLOR_BGR2HSV );
                cvtColor( src_test1, hsv_test1, COLOR_BGR2HSV );
                
                /// Using 50 bins for hue and 60 for saturation
                int h_bins = 50; int s_bins = 60;
                int histSize[] = { h_bins, s_bins };
                
                // hue varies from 0 to 179, saturation from 0 to 255
                float h_ranges[] = { 0, 180 };
                float s_ranges[] = { 0, 256 };
                
                const float* ranges[] = { h_ranges, s_ranges };
                
                // Use the o-th and 1-st channels
                int channels[] = { 0, 1 };
                
                
                /// Histograms
                MatND hist_base;
                MatND hist_test1;
                
                /// Calculate the histograms for the HSV images
                calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
                normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );
                
                calcHist( &hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false );
                normalize( hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat() );
                
                /// Apply the histogram comparison methods
                for( int compare_method = 0; compare_method < DataManager::getInstance().compareMethod.size(); compare_method++ )
                {
                    double base_test1 = compareHist( hist_base, hist_test1, compare_method );
                    
                    output << DataManager::getInstance().compareMethod[compare_method] << ": " <<base_test1 << endl;
                }
                
                output.close();
                
            }
        }
    }

};