#include "gpu.h"
#include "net.h"
#include "mat.h"
#include "cpu.h"
#include "datareader.h"

#include "vision_protectpravicy.hpp"

namespace NCNNEngine
{
    class Detector_NCNN : public Detector
    {
    public:
        Detector_NCNN();
        ~Detector_NCNN();
        bool Initialize(ModelInfo &model_info);
        bool detect(cv::Mat &img, std::vector<DetectInfo> &detect_info);
    private:
    };
}