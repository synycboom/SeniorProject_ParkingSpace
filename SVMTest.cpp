//
//  SVMTest.cpp
//  ParkingSpace
//
//  Created by synycboom on 9/15/2558 BE.
//  Copyright © 2558 HOME. All rights reserved.
//

#include "SVMTest.hpp"

using namespace cv;
using namespace cv::ml;
using namespace std;

class SVMTest{
    public:
    void show(){
        // Data for visual representation
        int width = 512, height = 512;
        Mat image = Mat::zeros(height, width, CV_8UC3);
        
        // Set up training data
        int labels[4] = {1, -1, -1, -1};
        Mat labelsMat(4, 1, CV_32SC1, labels);
        
        float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
        Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
        
        // Set up SVM's parameters
        Ptr<SVM> svm = ml::SVM::create();
        svm->setType(ml::SVM::C_SVC);
        svm->setKernel(ml::SVM::LINEAR);
        svm->setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
        
        // Train the SVM
        svm->train(trainingDataMat, ml::ROW_SAMPLE, labelsMat);
        
        Vec3b green(0,255,0), blue (255,0,0);
        // Show the decision regions given by the SVM
        for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1,2) << j,i);
            float response = svm->predict(sampleMat);
            
            if (response == 1)
            image.at<Vec3b>(i,j)  = green;
            else if (response == -1)
            image.at<Vec3b>(i,j)  = blue;
        }
        
        // Show the training data
        int thickness = -1;
        int lineType = 8;
        circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType);
        circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType);
        circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
        circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType);
        
        // Show support vectors
        thickness = 2;
        lineType  = 8;
        
        Mat supVecs = svm->getSupportVectors();
        cout << supVecs<< endl;
        for (int i = 0; i < supVecs.rows; ++i)
        {
            
            const float* v = supVecs.ptr<float>(i);
            
//            circle(image, Point( (int) v[0], (int) v[1]), 6, Scalar(128, 128, 128), thickness, lineType);
            circle(image, Point( (int) v[0], (int) v[1]), 6, Scalar(128, 128, 128), thickness, lineType);
        }
        //![show_vectors]
//        imwrite("result.png", image);        // save the image
        
        imshow("SVM Simple Example", image); // show it to the user
        waitKey(0);

    }
    static SVMTest& getInstance(){
        static SVMTest instance; // Guaranteed to be destroyed.
        // Instantiated on first use.
        return instance;
    }
    
private:
    SVMTest() {};// Constructor? (the {} brackets) are needed here.
    // C++ 11
    // =======
    // We can use the better technique of deleting the methods
    // we don't want.
    SVMTest(SVMTest const&) = delete;
    void operator=(SVMTest const&)  = delete;
};