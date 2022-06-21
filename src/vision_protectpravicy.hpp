#include <iostream>
#include "opencv2/opencv.hpp"

#include <memory>

#define DEBUG_CMD

void debug_cmd(const std::string fmt, ...)
{
#ifdef DEBUG_CMD
    int size = ((int)fmt.size()) * 2;
    std::string message;
    va_list ap;
    while (1)
    {
        message.resize(size);
        va_start(ap, fmt);
        int n = vsnprintf((char *)message.data(), size, fmt.c_str(), ap);
        va_end(ap);
        if (n > -1 && n < size)
        {
            message.resize(n);
            break;
        }
        if (n > -1)
            size = n + 1;
        else
            size *= 2;
    }
    printf("%s", message.c_str());
#endif
}

// information of detection result
typedef struct _DetectInfo
{
    cv::Rect_<float> rect;
    float prob;

    _DetectInfo()
    {
        prob = 0.0;
        rect.x = 0.0;
        rect.y = 0.0;
        rect.width = 0.0;
        rect.height = 0.0;
    }
} DetectInfo;

// information of inferenced model
typedef struct _ModelInfo
{
    std::string model_param, model_bin;

    std::string framework;
    
    bool use_fp16;
    bool use_gpu;
    int num_thread;
    int powersave;

    int image_process_width;
    int image_process_height;

    float score_threshold;
    float iou_threshold;

    _ModelInfo()
    {
        framework = "ncnn";

        use_fp16 = false;
        use_gpu = false;
        num_thread = 1;
        powersave = 0;

        image_process_width = 640;
        image_process_height = 640;

        score_threshold = 0.25;
        iou_threshold = 0.45;
    }
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

    Detector(ModelInfo model_info) {};
    virtual ~Detector() {};
    virtual bool detect(cv::Mat &img, std::vector<DetectInfo> &detect_info) = 0;
};
