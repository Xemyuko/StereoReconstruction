#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <sstream>
using namespace cv;
using namespace std;

String readfile(String filepath)
{
    ifstream infile{filepath};
    string file_contents{ istreambuf_iterator<char>(infile), istreambuf_iterator<char>() };
    
    return file_contents;
}

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



int main()
{

    /*
    cv::Mat img = cv::imread("C:/Users/Admin/Documents/GitHub/StereoReconstruction/test_data/testset1/bulb-multi/b1/cam1_pos_0000pattern_0000.jpg");
    namedWindow("imtest", WINDOW_AUTOSIZE);
    cv::imshow("imtest", img);
    cv::moveWindow("imtest", 0, 45);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    
    String filepath = "C:/Users/Admin/Documents/GitHub/StereoReconstruction/test_data/testset1/matrices/R.txt";


    cv::Mat_<double> Q = read_matrix(filepath,3,3,2);

    cout << Q << '\n';
    */

    return 0;
}