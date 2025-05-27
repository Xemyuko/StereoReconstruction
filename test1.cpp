
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
#include <chrono>
#include <numeric>
#include <algorithm>
#include "happly.h"




using namespace cv;
using namespace std;

double flo_chk = 1e-6;
void create_ply(vector<array<double, 3>> vertexPositions, vector<array<double, 3>> colors, String filename = "test.ply") {
    happly::PLYData plyOut;
    plyOut.addVertexPositions(vertexPositions);
    plyOut.addVertexColors(colors);
    plyOut.write(filename, happly::DataFormat::ASCII);
}

vector<array<double, 3>> make_blk_col_list(int len_num) {
    vector<array<double, 3>> res;
    for (int i = 0; i < len_num; i++) {
        array<double, 3> res_add = { 0,0,0 };
        res.push_back(res_add);
    }
    return res;
}


void make_pt_list(vector<double> x_list, vector<double> y_list, vector<double> z_list, vector<array<double, 3>>& res) {
    for (int i = 0; i < x_list.size(); i++) {
        array<double, 3> pt = { x_list[i], y_list[i], z_list[i] };
        res.push_back(pt);
    }
}

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

void write_image_list(vector<Mat> images, String base_name, String folder, String ext) {
    String base = folder + base_name;
    for (int i = 0; i < images.size(); i++) {
        String filename = base + std::to_string(i) + ext;
        cv::imwrite(filename, images[i]);
    }
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

void rectify_list(vector<Mat> imgs1, vector<Mat> imgs2, Mat F, vector<Mat>& rect_imgs1, vector<Mat>& rect_imgs2, Mat_<double>& H1, Mat_<double>& H2) {
    std::vector<cv::Point2f> pts1, pts2;
    find_pts_SIFT(imgs1[0], imgs2[0], pts1, pts2);
    cv::stereoRectifyUncalibrated(pts1, pts2, F, imgs1[0].size(), H1, H2);
    for (int i = 0; i < imgs1.size(); i++) {
        Mat rect1, rect2;
        cv::warpPerspective(imgs1[i], rect1, H1, imgs1[0].size());
        cv::warpPerspective(imgs2[i], rect2, H2, imgs1[0].size());
        rect_imgs1.push_back(rect1);
        rect_imgs2.push_back(rect2);
    }

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



void camcal_single(vector<Mat> imgs, int rows, int cols, float scf, Mat& cameraMatrix, Mat& distCoeffs, bool isGray = false) {
    int CHECKERBOARD[2]{ rows,cols };
    std::vector<std::vector<cv::Point3f> > objpoints;
    std::vector<std::vector<cv::Point2f> > imgpoints;
    std::vector<cv::Point3f> objp;
    for (int i{ 0 }; i < CHECKERBOARD[1]; i++)
    {
        for (int j{ 0 }; j < CHECKERBOARD[0]; j++)
            objp.push_back(cv::Point3f(j, i, 0));
    }
    cv::Mat frame, gray;
    std::vector<cv::Point2f> corner_pts;
    bool success = false;
    for (int i = 0; i < imgs.size(); i++) {
        if (isGray) {
            gray = imgs[i];
        }
        else {
            cv::Mat frame = imgs[i];
            cvtColor(frame, gray, COLOR_BGR2GRAY);
        }

        success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        if (success) {
            cv::TermCriteria criteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 30, 0.001);
            cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);

        }
    }
    if (success) {
        cv::Mat R, T;
        cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);
    }
    else {
        cout << "Calibration Failure" << '\n';
    }


}

void camcal_stereo(vector<Mat> imgsL, vector<Mat> imgsR, int rows, int cols, float scf, Mat_<double>& kL,
    Mat_<double>& kR, Mat_<double>& distL, Mat_<double>& distR, Mat_<double>& R, Mat_<double>& t, Mat_<double>& E, Mat_<double>& F, bool isGray = false) {
    vector< vector< Point3f > > object_points;
    vector< vector< Point2f > > imagePoints1, imagePoints2;
    vector< Point2f > corners1, corners2;
    vector< vector< Point2f > > left_img_points, right_img_points;
    Size board_size = Size(rows, cols);

    Mat img1, img2, gray1, gray2;

    for (int i = 1; i < imgsL.size(); i++) {
        if (isGray) {
            gray1 = imgsL[i];
            gray2 = imgsR[i];

        }
        else {
            img1 = imgsL[i];
            img2 = imgsR[i];
            cvtColor(img1, gray1, COLOR_BGR2GRAY);
            cvtColor(img2, gray2, COLOR_BGR2GRAY);
        }

        bool found1 = false, found2 = false;
        found1 = cv::findChessboardCorners(gray1, board_size, corners1,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
        found2 = cv::findChessboardCorners(gray2, board_size, corners2,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
        if (found1)
        {
            cv::cornerSubPix(gray1, corners1, cv::Size(5, 5), cv::Size(-1, -1),
                cv::TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));

        }

        if (found2)
        {
            cv::cornerSubPix(gray2, corners2, cv::Size(5, 5), cv::Size(-1, -1),
                cv::TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));

        }

        vector< Point3f > obj;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                obj.push_back(Point3f((float)j * scf, (float)i * scf, 0));
        if (found1 && found2) {
            imagePoints1.push_back(corners1);
            imagePoints2.push_back(corners2);
            object_points.push_back(obj);
        }
    }
    if (imagePoints1.size() > 0) {
        for (int i = 0; i < imagePoints1.size(); i++) {
            vector< Point2f > v1, v2;
            for (int j = 0; j < imagePoints1[i].size(); j++) {
                v1.push_back(Point2f((double)imagePoints1[i][j].x, (double)imagePoints1[i][j].y));
                v2.push_back(Point2f((double)imagePoints2[i][j].x, (double)imagePoints2[i][j].y));
            }
            left_img_points.push_back(v1);
            right_img_points.push_back(v2);
        }
        stereoCalibrate(object_points, left_img_points, right_img_points, kL, distL, kR, distR, img1.size(), R, t, E, F);
    }
    else {
        cout << "Stereo Calibration Failure" << '\n';
    }

}

void comp_cor(int xCol, int yRow, double cor,
    vector<int> xList, vector<double> corList, vector<int> yList,
    bool& addVal, bool& removeVal, int& removePos) {
    for (int i = 0; i < xList.size(); i++) {
        if (xList[i] == xCol and yList[i] == yRow) {
            cout << "A: " << i << '\n';
            if (cor > corList[i]) {
                addVal = true;
                removePos = i;
                removeVal = true;
            }
            else {
                addVal = false;
            }
            break;
        }
    }

}

void ncc_correlate(vector<Mat> imagesL, vector<Mat> imagesR, double cor_thresh,
    vector<int>& xList, vector<int>& xMatch_list, vector<int>& yList, vector<int>& modY_list, vector<int>& modX_list, vector<double>& cor_list) {
    int offset = 10;
    int rows = imagesL[0].size().height;
    int cols = imagesL[0].size().width;
    int n = imagesR.size();
    for (int i = offset; i < rows - offset; i += 10) {
        //loop through rows of stack 1 and stack 2
        if (i % 10 == 0) {
            cout << i << '/' << rows - offset << '\n';
            cout << "Matches: " << xList.size() << '\n';
        }


        for (int j = offset; j < cols - offset; j += 10) {

            //loop through columns of stack 1
            vector<int> Gi; // define pixel stack values of stack 1
            for (int k = 0; k < n; k++) {
                // loop through images of stack 1
                Gi.push_back((int)imagesL[k].at<uchar>(i, j));
            }

            if (std::accumulate(Gi.begin(), Gi.end(), 0) > flo_chk) {
                double agi = std::accumulate(Gi.begin(), Gi.end(), 0) / n;
                double val_i = 0.0;
                double max_cor = 0.0;
                int max_ind = -1;
                int max_modY = 0;
                int max_modX = 0;
                for (int vc_i = 0;vc_i < Gi.size(); vc_i++) {
                    val_i += std::pow((Gi[vc_i] - agi), 2);
                }

                for (int a = offset; a < cols - offset; a++) {
                    //loop through columns of stack 2
                    vector<int> Gt; // define pixel stack values of stack 2 to compare with stack 1
                    for (int b = 0; b < n; b++) {

                        Gt.push_back((int)imagesR[b].at<uchar>(i, a));
                    }
                    double agt = std::accumulate(Gt.begin(), Gt.end(), 0) / n;
                    double val_t = 0.0;
                    for (int vc_t = 0;vc_t < Gt.size(); vc_t++) {
                        val_t += std::pow((Gt[vc_t] - agt), 2);
                    }
                    if (val_i > flo_chk and val_t > flo_chk) {
                        double cor_o = 0.0;
                        for (int c_ind = 0; c_ind < Gi.size(); c_ind++) {
                            cor_o += (Gi[c_ind] - agi) * (Gt[c_ind] - agt);
                        }
                        double cor_d = std::sqrt(val_i * val_t);
                        double cor = cor_o / cor_d;
                        if (cor > max_cor) {
                            max_cor = cor;
                            max_ind = a;
                        }
                    }
                }
                vector<int> Gup;
                for (int b = 0; b < n; b++) {
                    Gup.push_back((int)imagesR[b].at<uchar>(i - 1, max_ind));
                }
                double agup = std::accumulate(Gup.begin(), Gup.end(), 0) / n;
                double val_up = 0.0;
                for (int vc_t = 0;vc_t < Gup.size(); vc_t++) {
                    val_up += std::pow((Gup[vc_t] - agup), 2);
                }
                if (val_i > flo_chk and val_up > flo_chk) {
                    double cor_o = 0.0;
                    for (int c_ind = 0; c_ind < Gi.size(); c_ind++) {
                        cor_o += (Gi[c_ind] - agi) * (Gup[c_ind] - agup);
                    }
                    double cor_d = std::sqrt(val_i * val_up);
                    double cor = cor_o / cor_d;
                    if (cor > max_cor) {
                        max_cor = cor;
                        max_modY = -1;
                    }
                }
                vector<int> Gdn;
                for (int b = 0; b < n; b++) {
                    Gdn.push_back((int)imagesR[b].at<uchar>(i + 1, max_ind));
                }
                double agdn = std::accumulate(Gdn.begin(), Gdn.end(), 0) / n;
                double val_dn = 0.0;
                for (int vc_t = 0;vc_t < Gdn.size(); vc_t++) {
                    val_dn += std::pow((Gdn[vc_t] - agdn), 2);
                }
                if (val_i > flo_chk and val_dn > flo_chk) {
                    double cor_o = 0.0;
                    for (int c_ind = 0; c_ind < Gi.size(); c_ind++) {
                        cor_o += (Gi[c_ind] - agi) * (Gdn[c_ind] - agdn);
                    }
                    double cor_d = std::sqrt(val_i * val_dn);
                    double cor = cor_o / cor_d;
                    if (cor > max_cor) {
                        max_cor = cor;
                        max_modY = 1;
                    }
                }

                //Check if found match should be added to list of matches - no duplicates, highest correlation score stays
                bool addVal = true;
                bool removeVal = false;
                int remove_pos = 0;
                //check correlation value against threshold value and check if found position is valid
                if (max_cor > cor_thresh and max_ind > 0 and max_ind < cols) {

                    comp_cor(j, i, max_cor, xList, cor_list, yList, addVal, removeVal, remove_pos);

                    if (removeVal) {
                        xList.erase(xList.begin() + remove_pos);
                        xMatch_list.erase(xList.begin() + remove_pos);
                        yList.erase(xList.begin() + remove_pos);
                        cor_list.erase(cor_list.begin() + remove_pos);
                        modY_list.erase(modY_list.begin() + remove_pos);
                        modX_list.erase(modX_list.begin() + remove_pos);

                    }
                    if (addVal) {
                        xList.push_back(j);
                        yList.push_back(i);
                        xMatch_list.push_back(max_ind);
                        cor_list.push_back(max_cor);
                        modY_list.push_back(max_modY);
                        modX_list.push_back(max_modX);
                    }
                }


            }
        }
    }
}
void convert_rect_matches(vector<int> xList, vector<int> xMatch_list, vector<int> yList, vector<int> xModList, vector<int> yModList, Mat H1, Mat H2,
    vector<double>& x1_list, vector<double>& y1_list, vector<double>& x2_list, vector<double>& y2_list) {
    Mat invH1 = H1.inv();
    Mat invH2 = H2.inv();
    for (int i = 0; i < xList.size(); i++) {
        int xL = xList[i];
        int y = yList[i];
        int xR = xMatch_list[i] + xModList[i];
        double xL_u = (invH1.at<double>(0, 0) * xL + invH1.at<double>(0, 1) * y + invH1.at<double>(0, 2)) /
            (invH1.at<double>(2, 0) * xL + invH1.at<double>(2, 1) * y + invH1.at<double>(2, 2));
        double yL_u = (invH1.at<double>(1, 0) * xL + invH1.at<double>(1, 1) * y + invH1.at<double>(1, 2)) /
            (invH1.at<double>(2, 0) * xL + invH1.at<double>(2, 1) * y + invH1.at<double>(2, 2));
        double xR_u = (invH2.at<double>(0, 0) * xR + invH2.at < double>(0, 1) * (y + yModList[i]) + invH2.at < double>(0, 2)) /
            (invH2.at < double>(2, 0) * xL + invH2.at<double>(2, 1) * (y + yModList[i]) + invH2.at<double>(2, 2));
        double yR_u = (invH2.at<double>(1, 0) * xR + invH2.at<double>(1, 1) * (y + yModList[i]) + invH2.at<double>(1, 2)) /
            (invH2.at<double>(2, 0) * xL + invH2.at<double>(2, 1) * (y + yModList[i]) + invH2.at < double>(2, 2));
        x1_list.push_back(xL_u);
        y1_list.push_back(yL_u);
        x2_list.push_back(xR_u);
        y2_list.push_back(yR_u);
    }
}

void triangulate(double x1, double y1, double x2, double y2, Mat_<double> kL, Mat_<double> kR, Mat_<double> R, Mat_<double> t, double& resX, double& resY, double& resZ) {
    double zer_col[] = { 0,0,0 };
    Mat_<double> col0 = Mat(3, 1, CV_64F, zer_col);
    Mat_<double> Al;
    hconcat(kL, col0, Al);
    Mat_<double> RT;
    hconcat(R, t, RT);
    Mat_<double> Ar = kR * RT;
    Mat_<double> sol0;
    subtract(x1 * Al.row(2), Al.row(1), sol0);
    Mat_<double> sol1;
    add(-y1 * Al.row(2), Al.row(0), sol1);
    Mat_<double> sol2;
    subtract(x2 * Ar.row(2), Ar.row(1), sol2);
    Mat_<double> sol3;
    add(-y2 * Ar.row(2), Ar.row(0), sol3);

    Mat_<double> solMat;
    vconcat(sol0, sol1, solMat);
    vconcat(solMat, sol2, solMat);
    vconcat(solMat, sol3, solMat);
    Mat_<double> w, u, vt;
    SVD::compute(solMat, w, u, vt);
    Mat_<double> Q = vt.row(3);
    Q = Q / Q.at<double>(0, 3);
    resX = Q.at<double>(0, 0);
    resY = Q.at<double>(0, 1);
    resZ = Q.at<double>(0, 2);
}

void triangulate_list(vector<double> x1_list, vector<double> y1_list, vector<double> x2_list, vector<double> y2_list, Mat kL, Mat kR, Mat R, Mat t,
    vector<double>& x_res, vector<double>& y_res, vector<double>& z_res) {
    for (int i = 0; i < x1_list.size(); i++) {
        double xR, yR, zR;
        triangulate(x1_list[i], y1_list[i], x2_list[i], y2_list[i], kL, kR, R, t, xR, yR, zR);
        x_res.push_back(xR);
        y_res.push_back(yR);
        z_res.push_back(zR);
    }
}



int main()
{
    auto beg = chrono::high_resolution_clock::now();
    String data_folder = "C:/Users/Admin/Documents/GitHub/StereoReconstruction/test_data/testset1/";
    //String data_folder = "C:/Users/myuey/Documents/GitHub/StereoReconstruction/test_data/testset1/";
    String img_folder = data_folder + "bulb-multi/b1/";
    /* Calibration
    String cal_folder = data_folder + "checkerboards/";
    vector<Mat> imagesLCal, imagesRCal;
    bool isGrayCal = true;
    load_lr_images(cal_folder, ".jpg", isGrayCal, imagesLCal, imagesRCal);

    Mat_<double> kS, distS;
    camcal_single(imagesL, 4, 7, 0.008, kS,distS, true);
    cout << kS << '\n';
    //cv::imshow("pr", imagesL[0]);
    //cv::waitKey(0);
    Mat_<double> kL, distL, kR, distR, R, t, E, F;
    camcal_stereo(imagesL, imagesR, 3,6,0.008,kL, distL, kR, distR, R, t, E, F, true);
    cout << R << '\n';
    */

    vector<Mat> imagesL, imagesR;
    bool isGray = true;
    load_lr_images(img_folder, ".jpg", isGray, imagesL, imagesR);

    cv::Mat img1 = imagesL[0];
    cv::Mat img2 = imagesR[0];
    /*
    int filt_thr = 30;
    for (int a = 0; a < imagesL.size(); a++) {
        imagesL[a].setTo(0, imagesL[a]<filt_thr);
        imagesR[a].setTo(0, imagesR[a] < filt_thr);
    }
    */
    String mat_fol = data_folder + "matrices/";

    cv::Mat_<double> R = read_matrix(mat_fol + "R.txt", 3, 3, 2);
    cv::Mat_<double> t = read_matrix(mat_fol + "t.txt", 3, 1, 2);

    cv::Mat_<double> kL = read_matrix(mat_fol + "kL.txt", 3, 3, 2);
    cv::Mat_<double> kR = read_matrix(mat_fol + "kR.txt", 3, 3, 2);

    cv::Mat_<double> distL = read_matrix(mat_fol + "distL.txt", 1, 5, 2);
    cv::Mat_<double> distR = read_matrix(mat_fol + "distR.txt", 1, 5, 2);

    cv::Mat_<double> F = read_matrix(mat_fol + "f.txt", 3, 3, 2);
    vector<Mat> rectL_images, rectR_images;
    Mat_<double> H1, H2;
    rectify_list(imagesL, imagesR, F, rectL_images, rectR_images, H1, H2);


    vector<int> xList, xMatch_list, yList, modY_list, modX_list;
    vector<double> cor_list;
    ncc_correlate(rectL_images, rectR_images, 0.9, xList, xMatch_list, yList, modY_list, modX_list, cor_list);
    cout << "xVal: " << xList[0] << " yVal: " << yList[0] << " xMatchVal: " << xMatch_list[0] << " corVal: " << cor_list[0] << '\n';
    cout << "xVal: " << xList[10] << " yVal: " << yList[10] << " xMatchVal: " << xMatch_list[10] << " corVal: " << cor_list[10] << '\n';
    cout << "xVal: " << xList[20] << " yVal: " << yList[20] << " xMatchVal: " << xMatch_list[20] << " corVal: " << cor_list[20] << '\n';
    vector<double> x1_c, y1_c, x2_c, y2_c;
    convert_rect_matches(xList, xMatch_list, yList, modY_list, modX_list, H1, H2, x1_c, y1_c, x2_c, y2_c);
    cout << "X1_c size: " << x1_c.size() << '\n';
    vector<double> x_list, y_list, z_list;
    triangulate_list(x1_c, y1_c, x2_c, y2_c, kL, kR, R, t, x_list, y_list, z_list);
    vector<array<double, 3>> pt_list, cols;
    make_pt_list(x_list, y_list, z_list, pt_list);
    cols = make_blk_col_list(pt_list.size());
    create_ply(pt_list, cols);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - beg);
    std::cout << "Elapsed Time: " << duration.count() / 1000000.0 << " seconds\n";
    std::cout << "Elapsed Time: " << duration.count() / 1000000.0 / 60 << " minutes\n";
    //cv::Mat res = half_dim(create_4_view(img1, img2, rectL_images[0], rectR_images[0]));
    //cv::imshow("pr", res);
    //cv::waitKey(0);

    /*
    double x, y, z;
    triangulate(10.0, 20.0, 150.0, 250.0, kL, kR, R, t, x, y, z);
    cout << x << ' ' << y << ' ' << z << '\n';
    */
    return 0;
}