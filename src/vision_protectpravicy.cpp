#include "vision_protectpravicy.hpp"
#include "ncnn_yolo5face.hpp"


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


ProtectPravicy::ProtectPravicy(ModelInfo model_info)
{
    face_detector = std::make_shared<Facedetector::Detector_Yolo5face>();
    //face_detector->Initialize(model_info);
}

bool ProtectPravicy::removePersonalInfo()
{

    return true;
}

bool ProtectPravicy::removePersonalInfo_singleframe(cv::Mat &frame)
{

    return true;
}
