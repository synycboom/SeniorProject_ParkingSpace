//
//  DataManager.cpp
//  ParkingSpace
//
//  Created by synycboom on 9/15/2558 BE.
//  Copyright © 2558 HOME. All rights reserved.
//

#include "DataManager.hpp"
using namespace std;

class DataManager{
public:
    
    const string FULL_PATH_PHOTO = "/Users/synycboom/Dropbox/Senior Project/Photo/";
    const string FULL_PATH_VIDEO = "/Users/synycboom/Dropbox/Senior Project/Video/";
    
    const array<string, 7> carArr {"car1.jpg","car2.jpg","car3.jpg","car4.jpg","car5.jpg","car6.jpg","car7.jpg"};
    const array<string, 7> floorArr {"base_floor1.jpg","base_floor2.jpg","base_floor3.jpg","base_floor4.jpg","base_floor5.jpg","base_floor6.jpg","base_floor7.jpg"};
    const array<string, 4> compareMethod {"Correlation  ", "Chi-square   ", "Intersection ", "Bhattacharyya"};
    
    static DataManager& getInstance(){
        static DataManager instance; // Guaranteed to be destroyed.
        // Instantiated on first use.
        return instance;
    }
    
    
private:
    DataManager() {};// Constructor? (the {} brackets) are needed here.
    
    // C++ 03
    // ========
    // Dont forget to declare these two. You want to make sure they
    // are unacceptable otherwise you may accidentally get copies of
    // your singleton appearing.
    //    DataManager(DataManager const&);              // Don't Implement
    //    void operator = (DataManager const&); // Don't implement
    
    // C++ 11
    // =======
    // We can use the better technique of deleting the methods
    // we don't want.
    DataManager(DataManager const&) = delete;
    void operator=(DataManager const&)  = delete;
};