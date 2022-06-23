#include <iostream>
#include "vision_protectpravicy.hpp"


int main(int argc, char** argv)
{   
    // Detector Setting for NCNN Engine
    std::string model_param = argv[1];
    std::string model_bin = argv[2];
    
    int width = std::atoi(argv[3]);
    int height = std::atoi(argv[4]);

    float score_thrshold = std::atof(argv[5]);
    float iou_threshold = std::atof(argv[6]);

    int num_thread = std::atoi(argv[7]);
    int powersave = std::atoi(argv[8]);

    bool use_fp16 = std::atoi(argv[9])? true : false;
    bool use_gpu = std::atoi(argv[10])? true : false;
    

    // 0. get config
    ModelInfo config;
    config.model_param = model_param;
    config.model_bin = model_bin;

    config.image_process_width = width;
    config.image_process_height = width;

    config.score_threshold = score_thrshold;
    config.iou_threshold = iou_threshold;

    config.num_thread = num_thread;
    config.powersave = powersave;

    config.use_fp16 = use_fp16;
    config.use_gpu = use_gpu;


    // 1. create instance and initialize
    ProtectPravicy* protector = new ProtectPravicy(config);

    // 2. run protector
    protector->removePersonalInfo();

    // 3. release protector instance
    protector->~ProtectPravicy();
    delete protector;

    return 0;
}