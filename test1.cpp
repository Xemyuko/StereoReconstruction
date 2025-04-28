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
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

Mat half_dim(Mat im) {
    Mat res;
    cv::resize(im, res, cv::Size(im.cols / 2, im.rows / 2));
    return res;
}

Mat create_stereo(Mat im0, Mat im1, bool isSave = false) {
    Mat res;
    hconcat(im0, im1, res);
    res = half_dim(res);
    if (isSave) {
        imwrite("stereo.jpg", res);
    }
    return res;
}


Mat create_4_view(Mat im0, Mat im1, Mat im2, Mat im3, bool isSave = false) {
    Mat res1;
    Mat res2;
    hconcat(im0, im1, res1);
    hconcat(im2, im3, res2);
    Mat res3;
    vconcat(res1, res2, res3);
    res3 = half_dim(res3);
    if (isSave) {
        imwrite("comp4view.jpg", res3);
    }
    return res3;
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

void find_pts_SIFT(Mat img1, Mat img2, std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2) {
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    std::vector < cv::KeyPoint > keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    /*
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
        Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    imshow("Good Matches", half_dim(img_matches));
    waitKey();
    */



    for (int i = 0; i < good_matches.size(); i++) {
        pts1.push_back(keypoints1[good_matches[i].queryIdx].pt);
        pts2.push_back(keypoints1[good_matches[i].trainIdx].pt);
    }

}

void rectify_pair_SIFT(Mat img1, Mat img2, Mat F, Mat& rect1, Mat& rect2) {
    Mat H1, H2;
    std::vector<cv::Point2f> pts1, pts2;
    find_pts_SIFT(img1, img2, pts1, pts2);
    cv::stereoRectifyUncalibrated(pts1, pts2, F, img1.size(), H1, H2);

    cv::warpPerspective(img1, rect1, H1, img1.size());
    cv::warpPerspective(img2, rect2, H2, img1.size());
}



void rectify_pair(Mat imgL, Mat imgR, Mat_<double> kL, Mat_<double> kR, Mat_<double> distL, Mat_<double> distR, Mat_<double> R, Mat_<double> t, Mat& imgL_rect, Mat& imgR_rect) {
    Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(kL, distL, kR, distR, imgL.size(), R, t, R1, R2, P1, P2, Q);
    Mat rmap[2][2];
    cv::initUndistortRectifyMap(kL, distL, R1, P1, imgL.size(), CV_16SC2, rmap[0][0], rmap[0][1]);
    cv::initUndistortRectifyMap(kR, distR, R2, P2, imgL.size(), CV_16SC2, rmap[1][0], rmap[1][1]);

    cv::remap(imgL, imgL_rect, rmap[0][0], rmap[0][1], INTER_LINEAR);
    cv::remap(imgR, imgR_rect, rmap[1][0], rmap[1][1], INTER_LINEAR);
}

void camcal_single(vector<Mat> imgs, int rows, int cols, float scf) {

}


int main()
{
    String data_folder = "C:/Users/Admin/Documents/GitHub/StereoReconstruction/test_data/testset1/";

    String img_folder = data_folder + "bulb-multi/b1/";

    vector<Mat> imagesL;
    vector<Mat> imagesR;
    bool isGray = false;
    load_lr_images(img_folder, ".jpg", isGray, imagesL, imagesR);

    cv::Mat img1 = imagesL[0];
    cv::Mat img2 = imagesR[0];

    //cv::Mat img3 = create_stereo(img1,img2);
    //cv::imshow("pr", img3);
    //cv::waitKey(0);


    String mat_fol = data_folder + "matrices/";
    cv::Mat_<double> R = read_matrix(mat_fol + "R.txt", 3, 3, 2);
    cv::Mat_<double> t = read_matrix(mat_fol + "t.txt", 3, 1, 2);
    cv::Mat_<double> F = read_matrix(mat_fol + "f.txt", 3, 3, 2);
    cv::Mat_<double> kL = read_matrix(mat_fol + "kL.txt", 3, 3, 2);
    cv::Mat_<double> kR = read_matrix(mat_fol + "kR.txt", 3, 3, 2);
    cv::Mat_<double> distL = read_matrix(mat_fol + "distL.txt", 1, 5, 2);
    cv::Mat_<double> distR = read_matrix(mat_fol + "distR.txt", 1, 5, 2);


    Mat rectL, rectR;
    rectify_pair_SIFT(img1, img2, F, rectL, rectR);
    cv::Mat res = half_dim(create_4_view(img1, img2, rectL, rectR));
    cv::imshow("pr", res);
    cv::waitKey(0);





    return 0;
}