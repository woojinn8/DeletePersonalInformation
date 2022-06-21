#include "vision_protectpravicy.hpp"
#include "ncnn/ncnn_detector.hpp"

ProtectPravicy::ProtectPravicy(std::string config)
{
    // parse model_info


}

ProtectPravicy::ProtectPravicy(ModelInfo model_info)
{
    std::shared_ptr<Detector> pDetector = NULL;
    if(model_info.framework == "ncnn")   // NCNN Mode
    {
        pDetector = std::make_shared<NCNNEngine::Detector_NCNN>(model_info);
    }
    

}

ProtectPravicy::~ProtectPravicy()
{


}



bool ProtectPravicy::removePersonalInfo()
{


}

bool ProtectPravicy::removePersonalInfo_singleframe(cv::Mat &frame)
{


}
