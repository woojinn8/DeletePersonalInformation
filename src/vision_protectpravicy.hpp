#include <iostream>
#include "opencv2/opencv.hpp"

#include <memory>

// information of detection result
typedef struct _DetectInfo
{
    cv::Rect_<float> rect;
    float prob;

    _DetectInfo()
    {
        prob = 0.0;
    }
} DetectInfo;

// information of inferenced model
typedef struct _ModelInfo
{
    std::string model_param, model_bin;

    int mode_detector;
    
    bool use_fp16;
    bool use_gpu;
    int num_thread;
    int powersave;

    int image_width;
    int image_height;

    float score_threshold;
    float iou_threshold;

} ModelInfo;


class ProtectPravicy{

private:
    bool removePersonalInfo_singleframe(cv::Mat &frame);
    ModelInfo m_modelinfo;

    
public:
    ProtectPravicy(std::string config);
    ProtectPravicy(ModelInfo model_info);
    ~ProtectPravicy(){};

    bool removePersonalInfo();

};


class Detector
{
public:
    static std::shared_ptr<Detector> create(std::string framework);

    Detector() {};
    virtual ~Detector() {};
    virtual bool Initialize(ModelInfo &model_info) = 0;
    virtual bool detect(cv::Mat &img, std::vector<DetectInfo> &detect_info) = 0;
};
