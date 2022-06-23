#ifndef vision_protectpravicy_hpp
#define vision_protectpravicy_hpp
#include "datatype.hpp"

void debug_cmd(const std::string fmt, ...);

class Detector
{
public:
    Detector(){};
    virtual ~Detector(){};
    virtual bool initialize(const ModelInfo &model_info) = 0;
    virtual bool detect(cv::Mat &img, std::vector<DetectInfo> &detect_list) = 0;

private:
};

class ProtectPravicy
{

private:
    bool removePersonalInfo_singleframe(cv::Mat &frame);
    ModelInfo m_modelinfo;
    std::shared_ptr<Detector> face_detector;

public:
    ProtectPravicy(ModelInfo model_info);
    ~ProtectPravicy(){};

    bool removePersonalInfo();
};

#endif