#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <sstream>
using namespace cv;
using namespace std;


std::vector<std::string> split(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}


cv::Mat_<double> read_matrix(String filepath, int rows, int cols, int skiprows)
{
    ifstream infile{ filepath };
    string file_contents{ istreambuf_iterator<char>(infile), istreambuf_iterator<char>() };
    if (file_contents.substr(file_contents.length() - 1).compare("\n") == 0)
    {
        file_contents = file_contents.substr(0, file_contents.length() - 1);
    }


    String delim = "\n";
    String delim2 = " ";
    cv::Mat_<double> P(rows, cols);
    std::vector<std::string> v = split(file_contents, delim);
    
    for (int i = skiprows; i < v.size(); i++)
    {
        std::vector<std::string> vr = split(v[i], delim2);

        for (int j = 0; j < vr.size(); j++)
        {
            P.at<double>(i - skiprows, j) = stod(vr[j]);
        }   
    }
    return P;
}

void write_matrix(String filepath, Mat_<double> P, bool isHeader) {
    ofstream wmat;
    wmat.open(filepath);
    if (isHeader) {
        wmat << "# " << P.rows << '\n';
        wmat << "# " << P.cols << '\n';
    }
    for (int i = 0; i < P.rows; i++) {
        for (int j = 0; j < P.cols; j++) {
            wmat << P.at<double>(i, j) << ' ';
        }
        wmat << '\n';
    }
    wmat.close();
}

void load_lr_images(String img_folder, String ext, bool isGray, vector<Mat>& imagesL, vector<Mat>& imagesR) {
    String target = img_folder + "*" + ext;
    std::vector<String> fn;
    cv::glob(target, fn, false);
    size_t count = fn.size();
    for (size_t i = 0; i < count; i++) {
        String checkStr = fn[i];
        if (checkStr.find("cam1") != std::string::npos) {
            if (isGray) {
                imagesL.push_back(imread(checkStr, IMREAD_GRAYSCALE));
            }
            else {
                imagesL.push_back(imread(checkStr));
            }
        }
        else {
            if (isGray) {
                imagesR.push_back(imread(checkStr, IMREAD_GRAYSCALE));
            }
            else {
                imagesR.push_back(imread(checkStr));
            }
        }
    }
}

void rectify_pair(Mat_<double> kL, Mat_<double> kR, Mat_<double> R, Mat_<double> t, Mat imgL, Mat imgR, Mat& imgL_rect, Mat& imgR_rect) {

}




int main()
{
    String data_folder = "C:/Users/Admin/Documents/GitHub/StereoReconstruction/test_data/testset1/";
    String img_folder = data_folder + "bulb-multi/b1/";

    std::vector<String> fn;
    cv::glob(img_folder + "*.jpg", fn, false);
    vector<Mat> imagesL;
    vector<Mat> imagesR;
    bool isGray = false;
    load_lr_images(img_folder, ".jpg", isGray, imagesL, imagesR);
    cv::Mat img1 = imagesL[0];
    cv::Mat img2 = imagesR[0];
    cv::Mat img3;
    hconcat(img2, img2, img3);
    int new_width = 1456;
    int new_height = 544;
    Mat img3_disp;
    cv::resize(img3, img3_disp, cv::Size(new_width, new_height));
    
    cv::imshow("pr", img3_disp);
    cv::waitKey(0);
    
    
    String matrices_folder = data_folder + "matrices/";
    String Rpath = matrices_folder + "R.txt";
    String tPath = matrices_folder + "t.txt";
    String FPath = matrices_folder + "F.txt";
    


    cv::Mat_<double> R = read_matrix(Rpath,3,3,2);
    write_matrix("R.txt", R, true);

    

    return 0;
}