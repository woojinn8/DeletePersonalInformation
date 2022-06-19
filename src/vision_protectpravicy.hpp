#include <iostream>
#include "opencv2/opencv.hpp"


class ProtectPravicy{

private:
    bool removePersonalInfo_singleframe(cv::Mat &frame);

public:
    ProtectPravicy(std::string config);
    ~ProtectPravicy(){};

    bool removePersonalInfo();

};