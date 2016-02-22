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

#include "ColorDetection.cpp"
#include "DataManager.cpp"

#include <cstring>      // Needed for memset
#include <sys/socket.h> // Needed for the socket functions
#include <netdb.h>      // Needed for the socket functions

#define CODE_SEGMENT 1
#define MODE 2
//#define MAX_CAR 114
#define MAX_CAR 100
//#define MAX_NOT_CAR 107
#define MAX_NOT_CAR 100


#define TRAIN_MIN 1
#define TRAIN_MAX 100

#define TEST_MIN 91
#define TEST_MAX 100

#define RESULT "save-test-2.txt"
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

void cropImage(){

    Mat img = imread(PHOTO_PATH + "Marking_day3/1.JPG");
    Mat origin;
    Mat output;

//    cvtColor(img, img, CV_BGR2HSV);
    Mat yellowMat, greenMat, pinkMat, orangeMat;
    
    inRange(img, Scalar(110,180,150), Scalar(140,255,255), greenMat);
    inRange(img, Scalar(90,150,215), Scalar(100,255,255), orangeMat);
    inRange(img, Scalar(240,170,170), Scalar(250,200,255), pinkMat);
    inRange(img, Scalar(50,170,170), Scalar(70,255,255), yellowMat);
    
    medianBlur(yellowMat, yellowMat, 3);
    medianBlur(orangeMat, orangeMat, 3);
    medianBlur(pinkMat, pinkMat, 3);
    medianBlur(greenMat, greenMat, 3);
    output = greenMat + orangeMat + pinkMat + yellowMat;
    
    Mat canny_yellow,canny_pink,canny_green,canny_orange;
    vector<vector<Point> > contour_yellow,contour_pink,contour_green,contour_orange;
    vector<Vec4i> hierarchy;
    

    Canny( yellowMat, canny_yellow, 100, 200, 3 );
    Canny( orangeMat, canny_orange, 100, 200, 3 );
    Canny( pinkMat, canny_pink, 100, 200, 3 );
    Canny( greenMat, canny_green, 100, 200, 3 );

    findContours( canny_yellow, contour_yellow, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    findContours( canny_pink, contour_pink, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    findContours( canny_green, contour_green, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    findContours( canny_orange, contour_orange, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    
    cv::Point centoid_yellow,centoid_red,centoid_green,centoid_blue;
    
    cv::Moments momYellow= cv::moments(cv::Mat(contour_yellow[0]));
    cv::Moments momRed= cv::moments(cv::Mat(contour_pink[0]));
    cv::Moments momGreen= cv::moments(cv::Mat(contour_green[0]));
    cv::Moments momBlue= cv::moments(cv::Mat(contour_orange[0]));
    
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
    
    source_points[0] = centoid_blue;
    source_points[1] = centoid_red;
    source_points[2] = centoid_green;
    source_points[3] = centoid_yellow;
    
    dest_points[0] = Point(0,0);
    dest_points[1] = Point(maxWidth - 1,0);
    dest_points[2] = Point(maxWidth - 1, maxHeight - 1);
    dest_points[3] = Point(0, maxHeight);
    
    Mat m = getPerspectiveTransform(source_points, dest_points);
    warpPerspective(img, output, m, Size(maxWidth, maxHeight) );
    imshow("output", output);
    waitKey(0);


}






void showSiftFeature(string pname){
    Mat out;
    Mat img;
    Mat distance;
    namedWindow("SIFT");
    string imgPath1 = PHOTO_PATH +pname;
    img = imread(imgPath1);
//    img = imread(pname);
    imshow("SIFT", img);
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
//    cout << keypoints_1.size() << endl;
    
    drawKeypoints(img, keypoints_1, out);
    imshow("SIFT", out);
    waitKey(0);
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
    for (int index = TRAIN_MIN; index <= TRAIN_MAX; index++) {
        if(index >= TEST_MIN && index <= TEST_MAX)
            continue;
        string picName = carName + to_string(index) + extension;
        Mat input = imread(picName, CV_LOAD_IMAGE_GRAYSCALE);
        cout << "Positive Train "<<picName << endl;
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
        
        if(index == TRAIN_MIN){
            cout << bowDescriptor << endl;
        }
    }
    
    //negative training set
    
    string floorName = PHOTO_PATH + "/not_car_set/notcar";
    Mat tranningNegative;
    for (int index = TRAIN_MIN; index <= TRAIN_MAX; index++) {
        if(index >= TEST_MIN && index <= TEST_MAX)
            continue;
        string picName = floorName + to_string(index) + extension;
        Mat input = imread(picName, CV_LOAD_IMAGE_GRAYSCALE);
        cout << "Negative Train "<<picName << endl;
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
//    svm->setType(ml::SVM::C_SVC);
//    svm->setGamma(30);
//    svm->setKernel(ml::SVM::RBF);
//    svm->setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
//    svm->train(trainingSet, ml::ROW_SAMPLE, labelsMat);
//    cv::FileStorage fss("hyperplain_svm.xml",cv::FileStorage::WRITE);
//    svm->write(fss);
//    fss.release();
//    {
//        cv::FileStorage fs("hyperplain_svm.xml", cv::FileStorage::APPEND);
//        fs << "format" << 3; // So "isLegacy" return false;
//    }
//    fs.release();
    
//    cv::FileStorage read("hyperplain_svm.xml",
//                         cv::FileStorage::READ);
//    auto svm = cv::ml::SVM::create();
//    svm->read(read.root());
//    Ptr<SVM> svm = ml::SVM::load<SVM>("hyperplain_svm.xml");
//    cout << svm->getGamma() << endl;
    
    
    
    
    ///////// prediction /////////
    ofstream myfile;
    myfile.open (RESULT);
    
    int falseNegative = 0, falsePositive = 0;
    for (int index = TEST_MIN; index <= TEST_MAX; index++) {
        string targetPhoto = PHOTO_PATH + "/car_set/car"
                            + to_string(index)+ extension;
        
        cout << "Positive Test "<< targetPhoto << endl;
        Mat img=imread(targetPhoto, CV_LOAD_IMAGE_GRAYSCALE);
        vector<KeyPoint> targetKeypoints;
        detector->detect(img,targetKeypoints);
        Mat targetBowDescriptor;
        bowDE.compute2(img,targetKeypoints,targetBowDescriptor);
        float response = svm->predict(targetBowDescriptor);
        if(response == -1) falseNegative++;
        myfile << "car" << to_string(index) << " = " << to_string(response) << endl;
    }
    for (int index = TEST_MIN ; index <= TEST_MAX; index++) {
        string targetPhoto = PHOTO_PATH + "/not_car_set/notcar"
                            + to_string(index) + extension;
        
        cout << "Negative Test "<< targetPhoto << endl;
        Mat img=imread(targetPhoto, CV_LOAD_IMAGE_GRAYSCALE);
        vector<KeyPoint> targetKeypoints;
        detector->detect(img,targetKeypoints);
        Mat targetBowDescriptor;
        bowDE.compute2(img,targetKeypoints,targetBowDescriptor);
        float response = svm->predict(targetBowDescriptor);
        if(response == 1) falsePositive++;
        myfile << "notcar" << to_string(index) << " = " << to_string(response) << endl;
    }
    myfile << endl << endl;
    myfile << "false positive = " << to_string(falsePositive) << endl;
    myfile << "false negative = " << to_string(falseNegative) << endl;
    myfile.close();
    
    
//    string named = "Predict Car";
////    string targetPhoto = DataManager::getInstance().FULL_PATH_PHOTO + "/car_set/slot16.jpg";
//    string targetPhoto = "/Users/synycboom/Desktop/slot12.jpg";
//    Mat img=imread(targetPhoto, CV_LOAD_IMAGE_GRAYSCALE);
//    vector<KeyPoint> targetKeypoints;
//    detector->detect(img,targetKeypoints);
//    Mat targetBowDescriptor;
//    bowDE.compute2(img,targetKeypoints,targetBowDescriptor);
//    float response = svm->predict(targetBowDescriptor);
//    
//    cv::FileStorage fss("hyperplain_svm.xml",cv::FileStorage::WRITE);
//    svm->save("hyperplain_svm.xml");
//    
//    cout << "Predict: " << response << endl;
//    imshow(named, img);
//    waitKey(0);
    
}

void createDictionary(){
    
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
    string folderPath = PHOTO_PATH + "car_set";

    Mat unclusteredDescriptors;
    vector<string> fileList;
    ls(folderPath,&fileList);
    
    char extension[] = ".png";
    for (auto &str : fileList){
        const char* fileName = str.c_str();
        if(strcspn(fileName, extension) > 0){
            string filePath = PHOTO_PATH + "car_set/" + fileName;
//            cout << filePath << endl;
            Mat descriptor;
            vector<KeyPoint> keypoints;
            Mat input = imread(filePath, CV_LOAD_IMAGE_GRAYSCALE);
            keypoints = getSiftKeyPoint_tmp(input);
            sift->compute(input, keypoints, descriptor);
            unclusteredDescriptors.push_back(descriptor);
        }
    }
    cout << "====== Start Training =======" <<endl;
    int dictionarySize= 1000;
    TermCriteria tc(CV_TERMCRIT_ITER,100, 1e-6);
    int retries=1;
    int flags=KMEANS_PP_CENTERS;
    BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
    Mat dictionary = bowTrainer.cluster(unclusteredDescriptors);
    
    FileStorage fs("dictionary.yml", FileStorage::WRITE);
    fs << "vocabulary" << dictionary;
    fs.release();
    cout << "====== Training Ended =======" <<endl;
}

int main(int argc, const char * argv[]) {
    

    
#if CODE_SEGMENT == 1
//    predictionStep();
    cropImage();
//    createDictionary();
    
    

#elif CODE_SEGMENT == 2
    if(argc != 4){
        cout << "usage: ./Predictor dictionaryFile hyperplainFile inputFile" << endl;
        return 1;
    }
    string dictionary_file = argv[1];
    string hyperplain_file = argv[2];
    string input_file = argv[3];
    
    //prepare BOW descriptor extractor from the dictionary
    Mat dictionary;
    FileStorage fs(dictionary_file, FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs.release();
    
    Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create();
    Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();
    Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create("FlannBased");
    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowDE(extractor,matcher);
    //Set the dictionary with the vocabulary we created in the first step
    bowDE.setVocabulary(dictionary);
    
    
//    Ptr<SVM> svm = StatModel::load<SVM>(hyperplain_file);
    FileStorage in(hyperplain_file, FileStorage::READ);
    Ptr<SVM> svm = StatModel::read<SVM>(in.getFirstTopLevelNode());
    string named = "Predict Car";
    Mat img=imread(input_file, CV_LOAD_IMAGE_GRAYSCALE);
    vector<KeyPoint> targetKeypoints;
    detector->detect(img,targetKeypoints);
    Mat targetBowDescriptor;
    bowDE.compute2(img,targetKeypoints,targetBowDescriptor);
    float response = svm->predict(targetBowDescriptor);
    cout << "Predict: " << response << endl;
    
    namedWindow("keypoint");
    Mat test;
    drawKeypoints(img, targetKeypoints, test);
    imshow("keypoint", test);
    waitKey(0);

#elif CODE_SEGMENT == 3
//for Descriptor extracting
    if(argc != 7){
        cout << "usage: ./ dictionaryFile carSetPath/ carAmount notCarSetPath/ notCarAmount outputPath" << endl;
        return 1;
    }
    
    string dictionaryFile = argv[1];
    string carPath        = argv[2];
    string carAmount      = argv[3];
    string notCarPath     = argv[4];
    string notCarAmount   = argv[5];
    string outputPath     = argv[6];
    
    if( carPath.at(carPath.length() - 1) == '/')
        carPath += "car";
    else
       carPath += "/car";
    
    if( notCarPath.at(notCarPath.length() - 1) == '/')
        notCarPath += "notcar";
    else
        notCarPath += "/notcar";
    
    if( outputPath.at(outputPath.length() - 1) != '/')
        outputPath += "/";
        
    
    Mat dictionary;
    FileStorage dictFile(dictionaryFile, FileStorage::READ);
    dictFile["vocabulary"] >> dictionary;
    dictFile.release();
    
    Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create();
    Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();
    Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create("FlannBased");
    BOWImgDescriptorExtractor bowDE(extractor,matcher);
    bowDE.setVocabulary(dictionary);
    
    FileStorage descriptors( outputPath + "Descriptors.yml", FileStorage::WRITE);
    FileStorage labels( outputPath + "Labels.yml", FileStorage::WRITE);
    string descriptorTag = "descriptor";
    string labelTag = "label";
    
    string extension = ".jpg";
    Mat trainingSet;
    Mat labelsMat;
    
    namedWindow("pos");
    
    //positive training set
   
    for (int index = 1; index <= stoi(carAmount); index++) {
        string picName = carPath + to_string(index) + extension;
        Mat input = imread(picName, CV_LOAD_IMAGE_GRAYSCALE);
        vector<KeyPoint> keypoints;
        detector->detect(input,keypoints);
        Mat bowDescriptor;
        bowDE.compute2(input,keypoints,bowDescriptor);
        trainingSet.push_back(bowDescriptor);
        labelsMat.push_back(1);
    }
    
    //negative training set
    for (int index = 1; index <= stoi(notCarAmount); index++) {
        string picName = notCarPath + to_string(index) + extension;
        Mat input = imread(picName, CV_LOAD_IMAGE_GRAYSCALE);
        vector<KeyPoint> keypoints;
        detector->detect(input,keypoints);
        Mat bowDescriptor;
        bowDE.compute2(input,keypoints,bowDescriptor);
        trainingSet.push_back(bowDescriptor);
        labelsMat.push_back(-1);
    }
    
    descriptors << descriptorTag << trainingSet;
    labels << labelTag << labelsMat;

    descriptors.release();
    labels.release();

    return 0;
    
#elif CODE_SEGMENT == 4
//Predictor Service
    if(argc != 4){
        cout << "usage: ./ descriptorPath labelPath portNum" << endl;
        return 1;
    }
    
    string descriptorPath = argv[1];
    string labelPath      = argv[2];
    string portNum        = argv[3];
    const char *portNumber = portNum.c_str();
    
    string descriptorTag = "descriptor";
    string labelTag = "label";
    
    Mat trainingSet, labelsMat;
    
    FileStorage descriptorFile(descriptorPath, FileStorage::READ);
    descriptorFile[descriptorTag] >> trainingSet;
    descriptorFile.release();
    
    FileStorage labelFile(labelPath, FileStorage::READ);
    labelFile[labelTag] >> labelsMat;
    labelFile.release();
    
    ///////// train SVM ////////
    Ptr<SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setGamma(30);
    svm->setKernel(ml::SVM::RBF);
    svm->setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
    svm->train(trainingSet, ml::ROW_SAMPLE, labelsMat);
    
    int status;
    struct addrinfo host_info;       // The struct that getaddrinfo() fills up with data.
    struct addrinfo *host_info_list; // Pointer to the to the linked list of host_info's.
    
    // The MAN page of getaddrinfo() states "All  the other fields in the structure pointed
    // to by hints must contain either 0 or a null pointer, as appropriate." When a struct
    // is created in C++, it will be given a block of memory. This memory is not necessary
    // empty. Therefor we use the memset function to make sure all fields are NULL.
    memset(&host_info, 0, sizeof host_info);
    
    cout << "Setting up the structs..."  << std::endl;
    
    host_info.ai_family = AF_UNSPEC;     // IP version not specified. Can be both.
    host_info.ai_socktype = SOCK_STREAM; // Use SOCK_STREAM for TCP or SOCK_DGRAM for UDP.
    host_info.ai_flags = AI_PASSIVE;
    
    // Now fill up the linked list of host_info structs with google's address information.
    status = getaddrinfo(NULL, portNumber , &host_info, &host_info_list);
    // getaddrinfo returns 0 on succes, or some other value when an error occured.
    // (translated into human readable text by the gai_gai_strerror function).
    if (status != 0)  std::cout << "getaddrinfo error: \n" << gai_strerror(status) ;
    
    cout << "Creating a socket..."  << std::endl;
    int socketfd ; // The socket descripter
    socketfd = socket(host_info_list->ai_family, host_info_list->ai_socktype,
                      host_info_list->ai_protocol);
    if (socketfd == -1)  std::cout << "socket error " ;
    
    std::cout << "Binding socket..."  << std::endl;
    // we make use of the setsockopt() function to make sure the port is not in use.
    // by a previous execution of our code. (see man page for more information)
    int yes = 1;
    status = setsockopt(socketfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int));
    status = ::bind(socketfd,host_info_list->ai_addr, host_info_list->ai_addrlen);
    if (status == -1)  std::cout << "bind error" << std::endl ;
    
    std::cout << "Listening for connections..."  << std::endl;
    status =  listen(socketfd, 1);
    if (status == -1)  std::cout << "listen error" << std::endl ;
    
    while(true){
        
        int new_sd;
        struct sockaddr_storage their_addr;
        socklen_t addr_size = sizeof(their_addr);
        new_sd = accept(socketfd, (struct sockaddr *)&their_addr, &addr_size);
        if (new_sd == -1)
        {
            std::cout << "listen error" << std::endl ;
        }
        else
        {
            std::cout << "Connection accepted. Using new socketfd : "  <<  new_sd << std::endl;
        }
        
        
        std::cout << "Waiting to recieve descriptor file"  << std::endl;
        ssize_t bytes_recieved;
        char incomming_data_buffer[1000];
        bytes_recieved = recv(new_sd, incomming_data_buffer,1000, 0);
        // If no data arrives, the program will just wait here until some data arrives.
        if (bytes_recieved == 0) std::cout << "host shut down." << std::endl ;
        if (bytes_recieved == -1)std::cout << "recieve error!" << std::endl ;
        std::cout << bytes_recieved << " descriptor file recieved :" << std::endl ;
        incomming_data_buffer[bytes_recieved] = '\0';
        std::cout << incomming_data_buffer << std::endl;
        
        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////Prediction//////////////////////////////////////////
        string inputFile(incomming_data_buffer);
        string inputDescriptorTag = "inputDescriptor";
        FileStorage inputDescriptorFile(inputFile,FileStorage::READ);
        Mat inputDescriptor;
        inputDescriptorFile[inputDescriptorTag] >> inputDescriptor;
        int resultInt = (int) svm->predict(inputDescriptor);
        
        ///////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////send result back///////////////////////////////////////
        
        std::cout << "sending back a result..."  << std::endl;
        string res = to_string(resultInt);
        char const *msg = res.c_str();
        int len;
        ssize_t bytes_sent;
        len = strlen(msg);
        bytes_sent = send(new_sd, msg, len, 0);
    }

//    std::cout << "Stopping server..." << std::endl;
//    freeaddrinfo(host_info_list);
//    close(new_sd);
//    close(socketfd);


    
#elif CODE_SEGMENT == 5
//making input for prediction service
    if(argc != 4){
        cout << "usage: ./ dictionary image portNumber" << endl;
        return 1;
    }
    string dictionaryPath = argv[1];
    string imageFile      = argv[2];
    string portNum        = argv[3];
    const char *portNumber = portNum.c_str();
    
    Mat dictionary;
    FileStorage dictFile(dictionaryPath, FileStorage::READ);
    dictFile["vocabulary"] >> dictionary;
    dictFile.release();

    Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create();
    Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();
    Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create("FlannBased");
    BOWImgDescriptorExtractor bowDE(extractor,matcher);
    bowDE.setVocabulary(dictionary);

    Mat i = imread(imageFile);
    vector<KeyPoint> targetKeypoints;
    detector->detect(i,targetKeypoints);
    Mat targetBowDescriptor;
    bowDE.compute2(i,targetKeypoints,targetBowDescriptor);
    
    
    string outName = imageFile.replace(imageFile.length() - 4,imageFile.length() - 1,"");
    outName += ".yml";
    FileStorage descriptorFile(outName, FileStorage::WRITE);
    string descriptorTag = "inputDescriptor";
    descriptorFile << descriptorTag << targetBowDescriptor;
    descriptorFile.release();
    
    int status;
    struct addrinfo host_info;       // The struct that getaddrinfo() fills up with data.
    struct addrinfo *host_info_list; // Pointer to the to the linked list of host_info's.
    
    // The MAN page of getaddrinfo() states "All  the other fields in the structure pointed
    // to by hints must contain either 0 or a null pointer, as appropriate." When a struct
    // is created in C++, it will be given a block of memory. This memory is not necessary
    // empty. Therefor we use the memset function to make sure all fields are NULL.
    memset(&host_info, 0, sizeof host_info);
    
    cout << "Setting up the structs..."  << std::endl;
    
    host_info.ai_family = AF_UNSPEC;     // IP version not specified. Can be both.
    host_info.ai_socktype = SOCK_STREAM; // Use SOCK_STREAM for TCP or SOCK_DGRAM for UDP.
    
    // Now fill up the linked list of host_info structs with google's address information.
    status = getaddrinfo("127.0.0.1", portNumber, &host_info, &host_info_list);
    // getaddrinfo returns 0 on succes, or some other value when an error occured.
    // (translated into human readable text by the gai_gai_strerror function).
    if (status != 0)  std::cout << "getaddrinfo error" << gai_strerror(status) ;
    
    cout << "Creating a socket..."  << std::endl;
    int socketfd ; // The socket descripter
    socketfd = socket(host_info_list->ai_family, host_info_list->ai_socktype,
                      host_info_list->ai_protocol);
    if (socketfd == -1)  std::cout << "socket error " ;
    
    cout << "Connect()ing..."  << std::endl;
    status = connect(socketfd, host_info_list->ai_addr, host_info_list->ai_addrlen);
    if (status == -1)  std::cout << "connect error" ;
    
    cout << "sending message..."  << std::endl;
    const char *msg = outName.c_str();
    int len;
    ssize_t bytes_sent;
    len = strlen(msg);
    bytes_sent = send(socketfd, msg, len, 0);
    
    cout << "Waiting for result..."  << std::endl;
    ssize_t bytes_recieved;
    char incoming_data_buffer[1000];
    bytes_recieved = recv(socketfd, incoming_data_buffer,1000, 0);
    // If no data arrives, the program will just wait here until some data arrives.
    if (bytes_recieved == 0) std::cout << "host shut down." << std::endl ;
    if (bytes_recieved == -1)std::cout << "recieve error!" << std::endl ;
    std::cout << bytes_recieved << " result recieved :" << std::endl ;
    std::cout << incoming_data_buffer << std::endl;
    
    std::cout << "Receiving complete. Closing socket..." << std::endl;
    freeaddrinfo(host_info_list);
    close(socketfd);
#endif
}
