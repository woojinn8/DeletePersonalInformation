#include "NCNN_Engine_C.hpp"

#define num_featuremap 4
#define hard_nms 1
#define blending_nms 2

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

void pretty_print(const ncnn::Mat &m)
{
    printf("%d, %d, %d\n", m.c, m.h, m.w);
}
//int cnt = 0;
namespace NCNNEngine
{
    int mode_face_detector, mode_landmark_detector, ps_d;
    ncnn::Net net_face, net_landmark;
    bool has_kps;
    //float max_score;

    bool initialize_SCRFD(ncnn::Net &net, const std::string &param_path, const std::string &bin_path);
    bool initialize_YOLO5Face(ncnn::Net &net, const std::string &param_path, const std::string &bin_path);

    bool detect_SCRFD(ncnn::Net &net, cv::Mat &img, std::vector<FaceInfo> &face_list, int target_input_width, int target_input_height);
    bool detect_YOLO5Face(ncnn::Net &net, cv::Mat &img, std::vector<FaceInfo> &face_list, int target_input_width, int target_input_height);

    bool initialize_SCRFD2(ncnn::Net &net, const std::string &param_path, const std::string &bin_path);
    bool detect_SCRFD2(ncnn::Net &net, cv::Mat &img, std::vector<FaceInfo> &face_list, int target_input_width, int target_input_height);


    Detector_NCNN::Detector_NCNN()
    {
    }

    Detector_NCNN::~Detector_NCNN()
    {
        net_face.clear();
        net_landmark.clear();
    }

    // Variables
    int num_thread_detector;
    float iou_threshold, score_threshold, score_threshold_init;
    // int speedup_r;
    bool useGPU_d, useFP16_d;

    int input_width_face, input_height_face;
    int input_width_landmark, input_height_landmark;

    int landmark_detector_size = 160;

    // Variables (YoloV5Face)
    typedef struct
    {
        int grid0;
        int grid1;
        int stride;
        float width;
        float height;
    } YOLO5FaceAnchor;

    typedef struct
    {
        float ratio;
        int dw;
        int dh;
        bool flag;
    } YOLO5FaceScaleParams;

    int iWidth;
    int iHeight;
    int max_face = 5;
    std::vector<unsigned int> strides = {8, 16, 32};
    static constexpr const unsigned int nms_pre = 1000;
    static constexpr const unsigned int max_nms = 30000;
    bool center_anchors_is_update;
    std::unordered_map<unsigned int, std::vector<YOLO5FaceAnchor>> center_anchors;

    
    // SCRFD2
    typedef struct
    {
        float ratio;
        int dw;
        int dh;
        bool flag;
    } ScaleParams;

    typedef struct
    {
      float cx;
      float cy;
      float stride;
    } SCRFDPoint;
    
    std::vector<int> feat_stride_fpn = {8, 16, 32}; // steps, may [8, 16, 32, 64, 128]
    // if num_anchors>1, then stack points in col major -> (height*num_anchor*width,2)
    // anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
    std::unordered_map<int, std::vector<SCRFDPoint>> center_points;
    bool center_points_is_update = false;

    unsigned int fmc = 3; // feature map count
    //unsigned int num_anchors = 2;
    //static constexpr const unsigned int nms_pre = 1000;
    //static constexpr const unsigned int max_nms = 30000;
    int iWidth_scrfd;
    int iHeight_scrfd;
    
    // Variables (Ultra, Retina)
    std::vector<std::vector<float>> priors = {};             // Ultra
    int image_w, image_h, in_w, in_h, num_anchors, nms_type; // Ultra
    int speedup_r;                                           // Retina

    int second_detector_size = 160;



    // common
    float intersection_area(const FaceInfo &a, const FaceInfo &b);
    void qsort_descent_inplace(std::vector<FaceInfo> &faceobjects, int left, int right);
    void qsort_descent_inplace(std::vector<FaceInfo> &faceobjects);
    void nms_sorted_bboxes(const std::vector<FaceInfo> &faceobjects, std::vector<int> &picked, float iou_threshold);

    // for SCRFD
    ncnn::Mat generate_anchors_scrfd(int base_size, const ncnn::Mat &ratios, const ncnn::Mat &scales);
    void generate_proposals_scrfd(const ncnn::Mat &anchors, int feat_stride, const ncnn::Mat &score_blob, const ncnn::Mat &bbox_blob, const ncnn::Mat &kps_blob, float prob_threshold, std::vector<FaceInfo> &faceobjects);

    // for YoloV5Face
    void generate_bboxes_kps(const YOLO5FaceScaleParams &scale_params,
                             std::vector<FaceInfo> &bbox_kps_collection,
                             ncnn::Extractor &extractor, float score_threshold,
                             float img_height, float img_width);
    void generate_anchors(unsigned int target_height, unsigned int target_width);
    void generate_bboxes_kps_single_stride(const YOLO5FaceScaleParams &scale_params,
                                           ncnn::Mat &det_pred, unsigned int stride,
                                           float score_threshold, float img_height, float img_width,
                                           std::vector<FaceInfo> &bbox_kps_collection);
    void nms_bboxes_kps(std::vector<FaceInfo> &input,
                        std::vector<FaceInfo> &output,
                        float iou_threshold, unsigned int topk);
    void resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                        int target_height, int target_width,
                        YOLO5FaceScaleParams &scale_params);
    float cal_iou(FaceInfo a, FaceInfo b);
    static inline float sigmoid(float x);

    // for SCRFD2
    void generate_points(const int target_height, const int target_width);
    void generate_bboxes_kps_SCRFD(const YOLO5FaceScaleParams &scale_params,
                        std::vector<FaceInfo> &bbox_kps_collection,
                        ncnn::Extractor &extractor, float score_threshold,
                        float img_height, float img_width);
    void generate_bboxes_kps_single_stride_SCRFD(
        const YOLO5FaceScaleParams &scale_params, ncnn::Mat &score_pred, ncnn::Mat &bbox_pred,
        ncnn::Mat &kps_pred, unsigned int stride, float score_threshold, float img_height,
        float img_width, std::vector<FaceInfo> &bbox_kps_collection);




    bool Detector_NCNN::Initialize(DevInfo &dev_info)
    {
        bool bInit_face = false;
        bool bInit_landmark = false;
        mode_face_detector = dev_info.mode_face_detector;
        mode_landmark_detector = dev_info.mode_landmark_detector;
        ps_d = dev_info.powersave_detector;

        num_thread_detector = dev_info.num_thread_detector;
        useGPU_d = dev_info.use_gpu_detector; 
        useFP16_d = dev_info.use_fp16;
        score_threshold_init = dev_info.score_threshold;
        iou_threshold = dev_info.iou_threshold;
        //max_score = 0.0;

        Utils::Logging("        <<< S1 NCNN Detector >>> : Initialize - face param : %s\n", dev_info.detector_face_param.c_str());
        Utils::Logging("        <<< S1 NCNN Detector >>> : Initialize - face bin : %s\n", dev_info.detector_face_bin.c_str());
        Utils::Logging("        <<< S1 NCNN Detector >>> : Initialize - landmark param : %s\n", dev_info.detector_landmark_param.c_str());
        Utils::Logging("        <<< S1 NCNN Detector >>> : Initialize - landmark bin : %s\n", dev_info.detector_landmark_bin.c_str());
        Utils::Logging("        <<< S1 NCNN Detector >>> : Initialize - num_threads : %d\n", dev_info.num_thread_detector);
        Utils::Logging("        <<< S1 NCNN Detector >>> : Initialize - powersave(0=all,1=little,2=big) : %d\n", dev_info.powersave_detector);
        Utils::Logging("        <<< S1 NCNN Detector >>> : Initialize - backend(0=cpu, 1=gpu) : %d\n", dev_info.use_gpu_detector);
        Utils::Logging("        <<< S1 NCNN Detector >>> : Initialize - mode : %d - %d\n", mode_face_detector, mode_landmark_detector);

        // mode
        // 0 : SCRFD
        // 1 : Yolo5Face

        input_width_face = dev_info.face_detector_width;
        input_height_face = dev_info.face_detector_height;
        input_width_landmark = dev_info.face_landmark_width;
        input_height_landmark = dev_info.face_landmark_height;

        switch (mode_face_detector)
        {
        case 0:
            bInit_face = initialize_SCRFD(net_face, dev_info.detector_face_param, dev_info.detector_face_bin);
            break;
        case 1:
            bInit_face = initialize_YOLO5Face(net_face, dev_info.detector_face_param, dev_info.detector_face_bin);
            break;
        case 2:
            bInit_face = initialize_SCRFD2(net_face, dev_info.detector_face_param, dev_info.detector_face_bin);
            break;
        default:
            break;
        }
        
        switch (mode_landmark_detector)
        {
        case 0:
            bInit_landmark = initialize_SCRFD(net_landmark, dev_info.detector_landmark_param, dev_info.detector_landmark_bin);
            break;
        case 1:
            bInit_landmark = initialize_YOLO5Face(net_landmark, dev_info.detector_landmark_param, dev_info.detector_landmark_bin);
            break;
        case 2:
            bInit_face = initialize_SCRFD2(net_landmark, dev_info.detector_landmark_param, dev_info.detector_landmark_bin);
            break;
        default:
            break;
        }

        return bInit_landmark;
    }

    bool Detector_NCNN::detect_face(cv::Mat &img, std::vector<FaceInfo> &face_list)
    {
        bool bDetected = false;

        score_threshold = score_threshold_init;

        switch (mode_face_detector)
        {
        case 0:
            bDetected = detect_SCRFD(net_face, img, face_list, input_width_face, input_height_face);
            break;
        case 1:
            bDetected = detect_YOLO5Face(net_face, img, face_list, input_width_face, input_height_face);
            break;
        case 2:
            bDetected = detect_SCRFD2(net_face, img, face_list, input_width_face, input_height_face);
        default:
            break;
        }
        
        // cal eye_distance
        for (int i = 0; i < face_list.size(); i++)
		{
			face_list[i].eye_distance = Utils::get_eye_distance(face_list[i].landmark);
            std::vector<float> pose;
            Utils::estimate_pose(face_list[i].landmark, pose);

            face_list[i].roll = pose[0];
            face_list[i].pitch = pose[2];
            face_list[i].yaw = pose[1];
        }
        return bDetected;
    }

    bool Detector_NCNN::detect_landmark(cv::Rect rect_first, cv::Mat &img, std::vector<FaceInfo> &faceobjects_landmark)
    {
        //max_score = 0.0;
        score_threshold = 0.1;

        auto s_landmark_preprocess = std::chrono::steady_clock::now();

        // Set rect to square
        cv::Rect FaceRectSquare = rect_first;
        if (FaceRectSquare.width > FaceRectSquare.height)
        {
            int diffBetweenWidthHeight = FaceRectSquare.width - FaceRectSquare.height;
            FaceRectSquare.y = MAX(0, FaceRectSquare.y - (diffBetweenWidthHeight / 2));
            FaceRectSquare.height = std::min(img.rows - FaceRectSquare.y, FaceRectSquare.height);
        }
        else
        {
            int diffBetweenWidthHeight = FaceRectSquare.height - FaceRectSquare.width;
            FaceRectSquare.x = MAX(0, FaceRectSquare.x - (diffBetweenWidthHeight / 2));
            FaceRectSquare.width = std::min(img.cols - FaceRectSquare.x, FaceRectSquare.width);
        }
        

        cv::Mat imgFaceArea = img(FaceRectSquare);
        
        float resizeScale = 1.0;

        auto e_landmark_preprocess = std::chrono::steady_clock::now();
        std::chrono::duration<float> d_landmark_preprocess = e_landmark_preprocess - s_landmark_preprocess;
        Utils::Logging_csv("%s,%f\n", "landmark_preprocess", d_landmark_preprocess.count() * 1000.0);

        auto s_landmark = std::chrono::steady_clock::now();
        bool isdetectlandmark = false;
        switch (mode_landmark_detector)
        {
        case 0:
            isdetectlandmark = detect_SCRFD(net_landmark, imgFaceArea, faceobjects_landmark, input_width_landmark, input_height_landmark);
            break;
        case 1:
            isdetectlandmark = detect_YOLO5Face(net_landmark, imgFaceArea, faceobjects_landmark, input_width_landmark, input_height_landmark);
            break;
        case 2:
            isdetectlandmark = detect_SCRFD2(net_landmark, imgFaceArea, faceobjects_landmark, input_width_landmark, input_height_landmark);
            break;
        default:
            Utils::Logging("        <<< S1 NCNN Detector >>> - %s : Fail to landmark detection because of mode selection\n", __FUNCTION__);
            break;
        }
		// 랜드마크 검출 결과가 2개 이상일때, 최고 스코어 한개만 남김
		if (faceobjects_landmark.size() > 1)
			faceobjects_landmark.resize(1);

        for (int i = 0; i < faceobjects_landmark.size(); i++)
		{
			faceobjects_landmark[i].eye_distance = Utils::get_eye_distance(faceobjects_landmark[i].landmark);
       
            std::vector<float> pose;
            Utils::estimate_pose(faceobjects_landmark[i].landmark, pose);

            faceobjects_landmark[i].roll = pose[0];
            faceobjects_landmark[i].pitch = pose[2];
            faceobjects_landmark[i].yaw = pose[1];
		}

        auto e_landmark = std::chrono::steady_clock::now();
        std::chrono::duration<float> d_landmark = e_landmark - s_landmark;
        Utils::Logging_csv("%s,%f\n", "detect_landmark", d_landmark.count() * 1000.0);

        auto s_rescale_landmark = std::chrono::steady_clock::now();

        cv::Scalar mean_value = cv::mean(img);
        double current_brightness = mean_value.val[0];

        // cv::Mat result_save = imgFaceArea.clone();        

        if (faceobjects_landmark.size() > 0)
        {
            // cv::circle(result_save, faceobjects_landmark[0].landmark[0], 1, cv::Scalar(0, 0, 255), 2);
            // cv::circle(result_save, faceobjects_landmark[0].landmark[1], 1, cv::Scalar(0, 0, 255), 2);
            // cv::circle(result_save, faceobjects_landmark[0].landmark[2], 1, cv::Scalar(0, 0, 255), 2);
            // cv::circle(result_save, faceobjects_landmark[0].landmark[3], 1, cv::Scalar(0, 0, 255), 2);
            // cv::circle(result_save, faceobjects_landmark[0].landmark[4], 1, cv::Scalar(0, 0, 255), 2);
            for (int i = 0; i < faceobjects_landmark.size(); i++)
            {
                // scale up
                faceobjects_landmark[i].eye_distance *= resizeScale;
                for (int iter_landmark = 0; iter_landmark < 5; iter_landmark++)
                {
                    faceobjects_landmark[i].landmark[iter_landmark].x = faceobjects_landmark[i].landmark[iter_landmark].x * resizeScale + FaceRectSquare.x;
                    faceobjects_landmark[i].landmark[iter_landmark].y = faceobjects_landmark[i].landmark[iter_landmark].y * resizeScale + FaceRectSquare.y;
                }
                faceobjects_landmark[i].rect.x = faceobjects_landmark[i].rect.x * resizeScale + FaceRectSquare.x;
                faceobjects_landmark[i].rect.y = faceobjects_landmark[i].rect.y * resizeScale + FaceRectSquare.y;
                faceobjects_landmark[i].rect.width = faceobjects_landmark[i].rect.width * resizeScale;
                faceobjects_landmark[i].rect.height = faceobjects_landmark[i].rect.height * resizeScale;
            }
        }
        else
        {
            // Utils::SaveImage(result_save, cnt, "FAIL_to_detect_landmark_" + std::to_string(max_score) + "_" + std::to_string(current_brightness), "", true);
        }

        auto e_rescale_landmark = std::chrono::steady_clock::now();
        std::chrono::duration<float> d_rescale_landmark = e_rescale_landmark - s_rescale_landmark;
        // Utils::Logging("        <<< S1 NCNN Detector >>> - %s : detect_rescale complete : %f\n", __FUNCTION__, d_rescale_landmark.count() * 1000);
        Utils::Logging_csv("%s,%f\n", "landmark_rescale", d_rescale_landmark.count() * 1000.0);


        /*
        if (cnt == 10000)
            cnt = 0;
        else
            cnt++;
        */
        return isdetectlandmark;
    }


    bool initialize_SCRFD(ncnn::Net &net, const std::string &param_path, const std::string &bin_path)
    {
        has_kps = true;

        int sucess_load_param = net.load_param_enc(param_path.c_str(), 6, "s1face");
        int sucess_load_model = net.load_model_enc(bin_path.c_str(), 6, "s1face");
        int sucess_load_result = sucess_load_model + sucess_load_param;

        if (sucess_load_result)
        {
            if (sucess_load_result)
                Utils::Logging("        <<< S1 NCNN Detector >>> : %s - Fail to load param\n", __FUNCTION__);

            if (sucess_load_result)
                Utils::Logging("        <<< S1 NCNN Detector >>> : %s - Fail to load model\n", __FUNCTION__);
        }
        else
        {
            Utils::Logging("        <<< S1 NCNN Detector >>> : %s - Load model result : %d %d\n", __FUNCTION__, sucess_load_param, sucess_load_model);
        }

        bool init_sucess = false;
        if (sucess_load_result == 0)
            init_sucess = true;

        return init_sucess;
    }

    bool initialize_YOLO5Face(ncnn::Net &net, const std::string &param_path, const std::string &bin_path)
    {
        has_kps = true;

        int sucess_load_param = net.load_param_enc(param_path.c_str(), 6, "s1face");
        int sucess_load_model = net.load_model_enc(bin_path.c_str(), 6, "s1face");
        int sucess_load_result = sucess_load_model + sucess_load_param;
        if (sucess_load_result)
        {
            if (sucess_load_result)
                Utils::Logging("        <<< S1 NCNN Detector >>> : %s - Fail to load param\n", __FUNCTION__);

            if (sucess_load_result)
                Utils::Logging("        <<< S1 NCNN Detector >>> : %s - Fail to load model\n", __FUNCTION__);
        }
        else
        {
            Utils::Logging("        <<< S1 NCNN Detector >>> : %s - Load model result : %d %d\n", __FUNCTION__, sucess_load_param, sucess_load_model);
        }

        center_anchors_is_update = false;

        bool init_sucess = false;
        if (sucess_load_result == 0)
            init_sucess = true;

        return init_sucess;
    }

    bool initialize_SCRFD2(ncnn::Net &net, const std::string &param_path, const std::string &bin_path)
    {
        has_kps = true;
        
        if (useFP16_d)
        {
            Logging("        <<< S1 NCNN DetectorS >>> : useFP16\n");
            net.opt.use_fp16_packed = true;
            net.opt.use_fp16_storage = true;
            net.opt.use_fp16_arithmetic = true;
        }
        else
        {
            net.opt.use_fp16_packed = false;
            net.opt.use_fp16_storage = false;
            net.opt.use_fp16_arithmetic = false;
        }

        int sucess_load_param = 1;
        int sucess_load_model = 1;

        sucess_load_param = net.load_param_enc(param_path.c_str(), 6, "s1face");
        sucess_load_model = net.load_model_enc(bin_path.c_str(), 6, "s1face");

        int sucess_load_result = sucess_load_model + sucess_load_param;

        if (sucess_load_result)
        {
            if (sucess_load_result)
                Utils::Logging("        <<< S1 NCNN Detector >>> : %s - Fail to load param\n",__FUNCTION__);

            if (sucess_load_result)
                Utils::Logging("        <<< S1 NCNN Detector >>> : %s - Fail to load model\n",__FUNCTION__);
        }
        else
        {
            Utils::Logging("        <<< S1 NCNN Detector >>> : %s - Load model result : %d %d\n",__FUNCTION__, sucess_load_param, sucess_load_model);
        }

        num_anchors = 2;

        bool init_sucess = false;
        if (sucess_load_result == 0)
            init_sucess = true;

        return init_sucess;
    }

    bool detect_SCRFD(ncnn::Net &net, cv::Mat &img, std::vector<FaceInfo> &faceobjects, int target_input_width, int target_input_height)
    {
        int width = img.cols;
        int height = img.rows;

        float prob_threshold = 0.50;
        float nms_threshold = 0.45;

        // pad to multiple of 32
        int target_size = MAX(target_input_width, target_input_height);
        int w = width;
        int h = height;
        float scale = 1.f;
        bool basis_width = false;
        if (w > h)
        {
            basis_width = true;
            scale = (float)target_size / w;
            w = target_size;
            h = h * scale;
        }
        else
        {
            scale = (float)target_size / h;
            h = target_size;
            w = w * scale;
        }

        cv::Mat img_resize;
        cv::resize(img, img_resize, cv::Size(img.cols * scale, img.rows * scale));

        ncnn::Mat in = ncnn::Mat::from_pixels(img_resize.data, ncnn::Mat::PIXEL_RGB, w, h);

        // pad to target_size rectangle
        int pad = 32; // 32;//(int)((float)target_size/10.0);
        int wpad = (w + pad - 1) / pad * pad - w;
        int hpad = (h + pad - 1) / pad * pad - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
        const float norm_vals[3] = {1 / 128.f, 1 / 128.f, 1 / 128.f};
        in_pad.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = net.create_extractor();
        ;
        /*
        ex.set_num_threads(num_thread_detector);
        ncnn::CpuSet ps_d_ = ncnn::get_cpu_thread_affinity_mask(ps_d);
        ncnn::set_cpu_thread_affinity(ps_d_);
        */
        ex.input("input.1", in_pad);

        std::vector<FaceInfo> faceproposals;
        // std::cout << "faceproposals" << std::endl;
        //  stride 8
        {
            ncnn::Mat score_blob, bbox_blob, kps_blob;
            ex.extract("score_8", score_blob);
            ex.extract("bbox_8", bbox_blob);
            if (has_kps)
                ex.extract("kps_8", kps_blob);

            const int base_size = 16;
            const int feat_stride = 8;
            ncnn::Mat ratios(1);
            ratios[0] = 1.f;
            ncnn::Mat scales(2);
            scales[0] = 1.f;
            scales[1] = 2.f;
            ncnn::Mat anchors = generate_anchors_scrfd(base_size, ratios, scales);

            std::vector<FaceInfo> faceobjects32;
            generate_proposals_scrfd(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, faceobjects32);

            // pretty_print(bbox_blob);
            // pretty_print(kps_blob);
            // pretty_print(score_blob);

            faceproposals.insert(faceproposals.end(), faceobjects32.begin(), faceobjects32.end());
        }
        // std::cout << "faceproposals 8" << std::endl;
        //  stride 16
        {
            ncnn::Mat score_blob, bbox_blob, kps_blob;
            ex.extract("score_16", score_blob);
            ex.extract("bbox_16", bbox_blob);
            if (has_kps)
                ex.extract("kps_16", kps_blob);

            const int base_size = 64;
            const int feat_stride = 16;
            ncnn::Mat ratios(1);
            ratios[0] = 1.f;
            ncnn::Mat scales(2);
            scales[0] = 1.f;
            scales[1] = 2.f;
            ncnn::Mat anchors = generate_anchors_scrfd(base_size, ratios, scales);

            std::vector<FaceInfo> faceobjects16;
            generate_proposals_scrfd(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, faceobjects16);

            // pretty_print(bbox_blob);
            // pretty_print(kps_blob);
            // pretty_print(score_blob);

            faceproposals.insert(faceproposals.end(), faceobjects16.begin(), faceobjects16.end());
        }
        // std::cout << "faceproposals 16" << std::endl;
        //  stride 32
        {
            ncnn::Mat score_blob, bbox_blob, kps_blob;
            ex.extract("score_32", score_blob);
            ex.extract("bbox_32", bbox_blob);
            if (has_kps)
                ex.extract("kps_32", kps_blob);

            const int base_size = 256;
            const int feat_stride = 32;
            ncnn::Mat ratios(1);
            ratios[0] = 1.f;
            ncnn::Mat scales(2);
            scales[0] = 1.f;
            scales[1] = 2.f;
            ncnn::Mat anchors = generate_anchors_scrfd(base_size, ratios, scales);

            std::vector<FaceInfo> faceobjects8;
            generate_proposals_scrfd(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, faceobjects8);

            // pretty_print(bbox_blob);
            // pretty_print(kps_blob);
            // pretty_print(score_blob);

            faceproposals.insert(faceproposals.end(), faceobjects8.begin(), faceobjects8.end());
        }
        // std::cout << "faceproposals 32" << std::endl;

        // sort all proposals by score from highest to lowest
        // std::cout << "qsort_descent_inplace" << std::endl;
        qsort_descent_inplace(faceproposals);

        // apply nms with nms_threshold
        std::vector<int> picked;
        // std::cout << "nms_sorted_bboxes" << std::endl;
        nms_sorted_bboxes(faceproposals, picked, nms_threshold);

        int face_count = picked.size();

        faceobjects.resize(face_count);
        for (int i = 0; i < face_count; i++)
        {
            faceobjects[i] = faceproposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (faceobjects[i].rect.x - (wpad / 2)) / scale;
            float y0 = (faceobjects[i].rect.y - (hpad / 2)) / scale;
            float x1 = (faceobjects[i].rect.x + faceobjects[i].rect.width - (wpad / 2)) / scale;
            float y1 = (faceobjects[i].rect.y + faceobjects[i].rect.height - (hpad / 2)) / scale;

            x0 = std::max(std::min(x0, (float)width - 1), 0.f);
            y0 = std::max(std::min(y0, (float)height - 1), 0.f);
            x1 = std::max(std::min(x1, (float)width - 1), 0.f);
            y1 = std::max(std::min(y1, (float)height - 1), 0.f);

            faceobjects[i].rect.x = x0;
            faceobjects[i].rect.y = y0;
            faceobjects[i].rect.width = x1 - x0;
            faceobjects[i].rect.height = y1 - y0;

            if (has_kps)
            {
                float x0 = (faceobjects[i].landmark[0].x - (wpad / 2)) / scale;
                float y0 = (faceobjects[i].landmark[0].y - (hpad / 2)) / scale;
                float x1 = (faceobjects[i].landmark[1].x - (wpad / 2)) / scale;
                float y1 = (faceobjects[i].landmark[1].y - (hpad / 2)) / scale;
                float x2 = (faceobjects[i].landmark[2].x - (wpad / 2)) / scale;
                float y2 = (faceobjects[i].landmark[2].y - (hpad / 2)) / scale;
                float x3 = (faceobjects[i].landmark[3].x - (wpad / 2)) / scale;
                float y3 = (faceobjects[i].landmark[3].y - (hpad / 2)) / scale;
                float x4 = (faceobjects[i].landmark[4].x - (wpad / 2)) / scale;
                float y4 = (faceobjects[i].landmark[4].y - (hpad / 2)) / scale;

                faceobjects[i].landmark[0].x = std::max(std::min(x0, (float)width - 1), 0.f);
                faceobjects[i].landmark[0].y = std::max(std::min(y0, (float)height - 1), 0.f);
                faceobjects[i].landmark[1].x = std::max(std::min(x1, (float)width - 1), 0.f);
                faceobjects[i].landmark[1].y = std::max(std::min(y1, (float)height - 1), 0.f);
                faceobjects[i].landmark[2].x = std::max(std::min(x2, (float)width - 1), 0.f);
                faceobjects[i].landmark[2].y = std::max(std::min(y2, (float)height - 1), 0.f);
                faceobjects[i].landmark[3].x = std::max(std::min(x3, (float)width - 1), 0.f);
                faceobjects[i].landmark[3].y = std::max(std::min(y3, (float)height - 1), 0.f);
                faceobjects[i].landmark[4].x = std::max(std::min(x4, (float)width - 1), 0.f);
                faceobjects[i].landmark[4].y = std::max(std::min(y4, (float)height - 1), 0.f);
            }
        }

        if (faceobjects.size() > 0)
            return true;
        else
            return false;
    }

    bool detect_YOLO5Face(ncnn::Net &net, cv::Mat &img, std::vector<FaceInfo> &face_list, int target_input_width, int target_input_height)
    {
        const float mean_vals[3] = {0.f, 0.f, 0.f}; // RGB
        const float norm_vals[3] = {1.0 / 255.f, 1.0 / 255.f, 1.0 / 255.f};

        if (img.empty())
            return false;
        // 0. image resize to square with padding
        int img_height = static_cast<int>(img.rows);
        int img_width = static_cast<int>(img.cols);
        cv::Mat img_resize;
        YOLO5FaceScaleParams scale_params;

        iWidth = target_input_width;
        iHeight = target_input_height;

        resize_unscale(img, img_resize, iHeight, iWidth, scale_params);

        // 1. make input tensor
        ncnn::Mat input;
        input = ncnn::Mat::from_pixels(img_resize.data, ncnn::Mat::PIXEL_RGB, iWidth, iHeight);
        input.substract_mean_normalize(mean_vals, norm_vals);

        // 2. inference & extract
        auto extractor = net.create_extractor();
        /*
        extractor.set_light_mode(false); // default
        extractor.set_num_threads(num_thread_detector);
        ncnn::CpuSet ps_d_ = ncnn::get_cpu_thread_affinity_mask(ps_d);
        ncnn::set_cpu_thread_affinity(ps_d_);
        */
        extractor.input("input", input);

        // 3. rescale & exclude.
        std::vector<FaceInfo> bbox_kps_collection;
        generate_bboxes_kps(scale_params, bbox_kps_collection, extractor,
                            score_threshold, img_height, img_width);

        // 4. hard nms with topk.
        nms_bboxes_kps(bbox_kps_collection, face_list, iou_threshold, max_face);

        if(face_list.size() > 0)
            return true;
        else
            return false;
    }

    bool detect_SCRFD2(ncnn::Net &net, cv::Mat &img, std::vector<FaceInfo> &face_list, int target_input_width, int target_input_height)
    {
        
        center_points_is_update = false;
         
        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
        const float norm_vals[3] = {1 / 128.f, 1 / 128.f, 1 / 128.f};
      
        if (img.empty())
            return false;
        // 0. image resize to square with padding
        int img_height = static_cast<int>(img.rows);
        int img_width = static_cast<int>(img.cols);

        iWidth_scrfd = target_input_width;
        iHeight_scrfd = target_input_height;

        cv::Mat img_resize;
        YOLO5FaceScaleParams scale_params;
        
        resize_unscale(img, img_resize, iHeight_scrfd, iWidth_scrfd, scale_params);

        // 1. make input tensor
        ncnn::Mat input = ncnn::Mat::from_pixels(img_resize.data, ncnn::Mat::PIXEL_RGB, iWidth_scrfd, iHeight_scrfd);
        input.substract_mean_normalize(mean_vals, norm_vals);

        // 2. inference & extract
        auto extractor = net.create_extractor();
        /*
        extractor.set_light_mode(false); // default
        extractor.set_num_threads(num_thread_detector);
        ncnn::CpuSet ps_d_ = ncnn::get_cpu_thread_affinity_mask(ps_d);
        ncnn::set_cpu_thread_affinity(ps_d_);
        */
        extractor.input("input.1", input);
		
        // 3. rescale & exclude.
        std::vector<FaceInfo> bbox_kps_collection;
		/*
        if(iHeight_scrfd == 160 && iWidth_scrfd == 160)
			generate_bboxes_kps_SCRFD(scale_params, bbox_kps_collection, extractor, 0.1, img_height, img_width);
		else
		*/
		generate_bboxes_kps_SCRFD(scale_params, bbox_kps_collection, extractor, score_threshold, img_height, img_width);

        // 4. hard nms with topk.
        nms_bboxes_kps(bbox_kps_collection, face_list, iou_threshold, max_face);
		/*
		char score[256] = {"0.0000"};
		if (face_list.size() > 0)
			sprintf(score, "%0.4f", face_list[0].prob);
	
		if(face_list.size() > 0)
		{ 
			for (int i = 0; i < face_list[0].landmark.size(); i++)
			{
				cv::Point2f origin_pt = face_list[0].landmark[i] * scale_params.ratio + cv::Point2f(scale_params.dw, scale_params.dh);
				cv::circle(img_resize, origin_pt, 2, cv::Scalar(0, 0, 255), 1);
			}
			SaveImage(img_resize, cnt, "detect_SCRFD2_input_FD_SUC", score, true);
		}
		else
			SaveImage(img_resize, cnt, "detect_SCRFD2_input_FD_FAIL", score, true);
		if (cnt == 10000)
			cnt = 0;
		else
			cnt++;
*/
        if(face_list.size() > 0)
            return true;
        else    
            return false;
    }

    void resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                        int target_height, int target_width,
                        YOLO5FaceScaleParams &scale_params)
    {
        if (mat.empty())
            return;
        int img_height = static_cast<int>(mat.rows);
        int img_width = static_cast<int>(mat.cols);

        mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
                         cv::Scalar(0, 0, 0));
        // scale ratio (new / old) new_shape(h,w)
        float w_r = (float)target_width / (float)img_width;
        float h_r = (float)target_height / (float)img_height;
        float r = std::min(w_r, h_r);
        // compute padding
        int new_unpad_w = static_cast<int>((float)img_width * r);  // floor
        int new_unpad_h = static_cast<int>((float)img_height * r); // floor
        int pad_w = target_width - new_unpad_w;                    // >=0
        int pad_h = target_height - new_unpad_h;                   // >=0

        int dw = pad_w / 2;
        int dh = pad_h / 2;

        // resize with unscaling
        cv::Mat new_unpad_mat = mat.clone();
        cv::resize(new_unpad_mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
        new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

        // record scale params.
        scale_params.ratio = r;
        scale_params.dw = dw;
        scale_params.dh = dh;
        scale_params.flag = true;
    }

    void generate_bboxes_kps(const YOLO5FaceScaleParams &scale_params,
                             std::vector<FaceInfo> &bbox_kps_collection,
                             ncnn::Extractor &extractor, float score_threshold,
                             float img_height, float img_width)
    {
        // (1,n,16=4+1+10+1=cxcy+cwch+obj_conf+5kps+cls_conf)
        ncnn::Mat det_stride_8, det_stride_16, det_stride_32;
        extractor.extract("det_stride_8", det_stride_8);
        extractor.extract("det_stride_16", det_stride_16);
        extractor.extract("det_stride_32", det_stride_32);

        generate_anchors(iHeight, iWidth);

        // generate bounding boxes.
        bbox_kps_collection.clear();

        generate_bboxes_kps_single_stride(scale_params, det_stride_8, 8, score_threshold,
                                          img_height, img_width, bbox_kps_collection);
        generate_bboxes_kps_single_stride(scale_params, det_stride_16, 16, score_threshold,
                                          img_height, img_width, bbox_kps_collection);
        generate_bboxes_kps_single_stride(scale_params, det_stride_32, 32, score_threshold,
                                          img_height, img_width, bbox_kps_collection);

        // std::cout << "generate_bboxes_kps num: " << bbox_kps_collection.size() << "\n";
    }

    void generate_anchors(unsigned int target_height, unsigned int target_width)
    {
        if (center_anchors_is_update)
            return;

        for (auto stride : strides)
        {
            unsigned int num_grid_w = target_width / stride;
            unsigned int num_grid_h = target_height / stride;
            std::vector<YOLO5FaceAnchor> anchors;

            if (stride == 8)
            {
                // 0 anchor
                for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
                {
                    for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
                    {
                        YOLO5FaceAnchor anchor;
                        anchor.grid0 = g0;
                        anchor.grid1 = g1;
                        anchor.stride = stride;
                        anchor.width = 4.f;
                        anchor.height = 5.f;
                        anchors.push_back(anchor);
                    }
                }
                // 1 anchor
                for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
                {
                    for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
                    {
                        YOLO5FaceAnchor anchor;
                        anchor.grid0 = g0;
                        anchor.grid1 = g1;
                        anchor.stride = stride;
                        anchor.width = 8.f;
                        anchor.height = 10.f;
                        anchors.push_back(anchor);
                    }
                }
                // 2 anchor
                for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
                {
                    for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
                    {
                        YOLO5FaceAnchor anchor;
                        anchor.grid0 = g0;
                        anchor.grid1 = g1;
                        anchor.stride = stride;
                        anchor.width = 13.f;
                        anchor.height = 16.f;
                        anchors.push_back(anchor);
                    }
                }
            } // 16
            else if (stride == 16)
            {
                // 0 anchor
                for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
                {
                    for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
                    {
                        YOLO5FaceAnchor anchor;
                        anchor.grid0 = g0;
                        anchor.grid1 = g1;
                        anchor.stride = stride;
                        anchor.width = 23.f;
                        anchor.height = 29.f;
                        anchors.push_back(anchor);
                    }
                }
                // 1 anchor
                for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
                {
                    for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
                    {
                        YOLO5FaceAnchor anchor;
                        anchor.grid0 = g0;
                        anchor.grid1 = g1;
                        anchor.stride = stride;
                        anchor.width = 43.f;
                        anchor.height = 55.f;
                        anchors.push_back(anchor);
                    }
                }
                // 2 anchor
                for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
                {
                    for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
                    {
                        YOLO5FaceAnchor anchor;
                        anchor.grid0 = g0;
                        anchor.grid1 = g1;
                        anchor.stride = stride;
                        anchor.width = 73.f;
                        anchor.height = 105.f;
                        anchors.push_back(anchor);
                    }
                }
            } // 32
            else
            {
                // 0 anchor
                for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
                {
                    for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
                    {
                        YOLO5FaceAnchor anchor;
                        anchor.grid0 = g0;
                        anchor.grid1 = g1;
                        anchor.stride = stride;
                        anchor.width = 146.f;
                        anchor.height = 217.f;
                        anchors.push_back(anchor);
                    }
                }
                // 1 anchor
                for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
                {
                    for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
                    {
                        YOLO5FaceAnchor anchor;
                        anchor.grid0 = g0;
                        anchor.grid1 = g1;
                        anchor.stride = stride;
                        anchor.width = 231.f;
                        anchor.height = 300.f;
                        anchors.push_back(anchor);
                    }
                }
                // 2 anchor
                for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
                {
                    for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
                    {
                        YOLO5FaceAnchor anchor;
                        anchor.grid0 = g0;
                        anchor.grid1 = g1;
                        anchor.stride = stride;
                        anchor.width = 335.f;
                        anchor.height = 433.f;
                        anchors.push_back(anchor);
                    }
                }
            }
            center_anchors[stride] = anchors;
        }

        center_anchors_is_update = true;
    }

    void generate_bboxes_kps_single_stride(const YOLO5FaceScaleParams &scale_params,
                                           ncnn::Mat &det_pred, unsigned int stride,
                                           float score_threshold, float img_height, float img_width,
                                           std::vector<FaceInfo> &bbox_kps_collection)
    {
        unsigned int nms_pre_ = (stride / 8) * nms_pre; // 1 * 1000,2*1000,...
        nms_pre_ = nms_pre_ >= nms_pre ? nms_pre_ : nms_pre;

        const unsigned int f_h = (unsigned int)iHeight / stride;
        const unsigned int f_w = (unsigned int)iWidth / stride;
        // e.g, 3*80*80 + 3*40*40 + 3*20*20 = 25200
        const unsigned int num_anchors = 3 * f_h * f_w;
        const float *output_ptr = (float *)det_pred.data;

        float r_ = scale_params.ratio;
        int dw_ = scale_params.dw;
        int dh_ = scale_params.dh;

        // have c=3 indicate 3 anchors at one grid
        unsigned int count = 0;
        auto &stride_anchors = center_anchors[stride];

        for (unsigned int i = 0; i < num_anchors; ++i)
        {
            const float *row_ptr = output_ptr + i * 16;
            float obj_conf = sigmoid(row_ptr[4]);
            if (obj_conf < score_threshold)
                continue; // filter first.
            float cls_conf = sigmoid(row_ptr[15]);
            if (cls_conf < score_threshold)
                continue; // face score.

            int grid0 = stride_anchors.at(i).grid0; // w
            int grid1 = stride_anchors.at(i).grid1; // h
            float anchor_w = stride_anchors.at(i).width;
            float anchor_h = stride_anchors.at(i).height;

            // bounding box
            const float *offsets = row_ptr;
            float dx = sigmoid(offsets[0]);
            float dy = sigmoid(offsets[1]);
            float dw = sigmoid(offsets[2]);
            float dh = sigmoid(offsets[3]);

            float cx = (dx * 2.f - 0.5f + (float)grid0) * (float)stride;
            float cy = (dy * 2.f - 0.5f + (float)grid1) * (float)stride;
            float w = std::pow(dw * 2.f, 2) * anchor_w;
            float h = std::pow(dh * 2.f, 2) * anchor_h;

            FaceInfo box_kps;
            float x1 = ((cx - w / 2.f) - (float)dw_) / r_;
            float y1 = ((cy - h / 2.f) - (float)dh_) / r_;
            float x2 = ((cx + w / 2.f) - (float)dw_) / r_;
            float y2 = ((cy + h / 2.f) - (float)dh_) / r_;

            box_kps.rect.x = std::max(0.f, x1);
            box_kps.rect.y = std::max(0.f, y1);
            box_kps.rect.width = std::min(img_width, x2 - x1);
            box_kps.rect.height = std::min(img_height, y2 - y1);
            box_kps.prob = obj_conf * cls_conf;

            // landmarks
            const float *kps_offsets = row_ptr + 5;
            for (unsigned int j = 0; j < 10; j += 2)
            {
                float kps_dx = kps_offsets[j];
                float kps_dy = kps_offsets[j + 1];
                float kps_x = (kps_dx * anchor_w + grid0 * (float)stride);
                float kps_y = (kps_dy * anchor_h + grid1 * (float)stride);

                cv::Point2f kps;
                kps_x = (kps_x - (float)dw_) / r_;
                kps_y = (kps_y - (float)dh_) / r_;
                kps.x = std::min(std::max(0.f, kps_x), img_width);
                kps.y = std::min(std::max(0.f, kps_y), img_height);
                box_kps.landmark.push_back(kps);
            }

            bbox_kps_collection.push_back(box_kps);

            count += 1; // limit boxes for nms.
            if (count > max_nms)
                break;
        }

        if (bbox_kps_collection.size() > nms_pre_)
        {
            std::sort(
                bbox_kps_collection.begin(), bbox_kps_collection.end(),
                [](const FaceInfo &a, const FaceInfo &b)
                { return a.prob > b.prob; }); // sort inplace
            // trunc
            bbox_kps_collection.resize(nms_pre_);
        }
    }

    void nms_bboxes_kps(std::vector<FaceInfo> &input,
                        std::vector<FaceInfo> &output,
                        float iou_threshold, unsigned int topk)
    {
        if (input.empty())
            return;
        std::sort(
            input.begin(), input.end(),
            [](const FaceInfo &a, const FaceInfo &b)
            { return a.prob > b.prob; });
        const unsigned int box_num = input.size();
        std::vector<int> merged(box_num, 0);

        unsigned int count = 0;
        for (unsigned int i = 0; i < box_num; ++i)
        {
            if (merged[i])
                continue;
            std::vector<FaceInfo> buf;

            buf.push_back(input[i]);
            merged[i] = 1;

            for (unsigned int j = i + 1; j < box_num; ++j)
            {
                if (merged[j])
                    continue;

                float iou = cal_iou(input[i], input[j]);

                if (iou > iou_threshold)
                {
                    merged[j] = 1;
                    buf.push_back(input[j]);
                }
            }
            output.push_back(buf[0]);

            // keep top k
            count += 1;
            if (count >= topk)
                break;
        }
    }

    float cal_iou(FaceInfo a, FaceInfo b)
    {
        // intersection over union
        float inter_area = intersection_area(a, b);
        float union_area = a.rect.area() + b.rect.area() - inter_area;

        float iou_result = inter_area / union_area;
        return iou_result;
    }

    // inner function
    static inline float sigmoid(float x)
    {
        return static_cast<float>(1.f / (1.f + std::exp(-x)));
    }

    float intersection_area(const FaceInfo &a, const FaceInfo &b)
    {
        cv::Rect_<float> inter = a.rect & b.rect;
        return inter.area();
    }

    void qsort_descent_inplace(std::vector<FaceInfo> &faceobjects, int left, int right)
    {
        int i = left;
        int j = right;
        float p = faceobjects[(left + right) / 2].prob;

        while (i <= j)
        {
            while (faceobjects[i].prob > p)
                i++;

            while (faceobjects[j].prob < p)
                j--;

            if (i <= j)
            {
                // swap
                std::swap(faceobjects[i], faceobjects[j]);

                i++;
                j--;
            }
        }

#pragma omp parallel sections
        {
#pragma omp section
            {
                if (left < j)
                    qsort_descent_inplace(faceobjects, left, j);
            }
#pragma omp section
            {
                if (i < right)
                    qsort_descent_inplace(faceobjects, i, right);
            }
        }
    }

    void qsort_descent_inplace(std::vector<FaceInfo> &faceobjects)
    {
        if (faceobjects.empty())
            return;

        qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
    }

    void nms_sorted_bboxes(const std::vector<FaceInfo> &faceobjects, std::vector<int> &picked, float iou_threshold)
    {
        picked.clear();

        const int n = faceobjects.size();

        std::vector<float> areas(n);
        for (int i = 0; i < n; i++)
        {
            areas[i] = faceobjects[i].rect.area();
        }

        for (int i = 0; i < n; i++)
        {
            const FaceInfo &a = faceobjects[i];

            int keep = 1;
            for (int j = 0; j < (int)picked.size(); j++)
            {
                const FaceInfo &b = faceobjects[picked[j]];

                // intersection over union
                float inter_area = intersection_area(a, b);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                //             float IoU = inter_area / union_area
                if (inter_area / union_area > iou_threshold)
                    keep = 0;
            }

            if (keep)
                picked.push_back(i);
        }
    }

    ncnn::Mat generate_anchors_scrfd(int base_size, const ncnn::Mat &ratios, const ncnn::Mat &scales)
    {
        int num_ratio = ratios.w;
        int num_scale = scales.w;

        ncnn::Mat anchors;
        anchors.create(4, num_ratio * num_scale);

        const float cx = 0;
        const float cy = 0;

        for (int i = 0; i < num_ratio; i++)
        {
            float ar = ratios[i];

            int r_w = round(base_size / sqrt(ar));
            int r_h = round(r_w * ar); // round(base_size * sqrt(ar));

            for (int j = 0; j < num_scale; j++)
            {
                float scale = scales[j];

                float rs_w = r_w * scale;
                float rs_h = r_h * scale;

                float *anchor = anchors.row(i * num_scale + j);

                anchor[0] = cx - rs_w * 0.5f;
                anchor[1] = cy - rs_h * 0.5f;
                anchor[2] = cx + rs_w * 0.5f;
                anchor[3] = cy + rs_h * 0.5f;
            }
        }

        return anchors;
    }

    void generate_proposals_scrfd(const ncnn::Mat &anchors, int feat_stride, const ncnn::Mat &score_blob, const ncnn::Mat &bbox_blob, const ncnn::Mat &kps_blob, float prob_threshold, std::vector<FaceInfo> &faceobjects)
    {
        int w = score_blob.w;
        int h = score_blob.h;

        // generate face proposal from bbox deltas and shifted anchors
        const int num_anchors = anchors.h;

        for (int q = 0; q < num_anchors; q++)
        {
            const float *anchor = anchors.row(q);

            const ncnn::Mat score = score_blob.channel(q);
            const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);

            // shifted anchor
            float anchor_y = anchor[1];

            float anchor_w = anchor[2] - anchor[0];
            float anchor_h = anchor[3] - anchor[1];

            for (int i = 0; i < h; i++)
            {
                float anchor_x = anchor[0];

                for (int j = 0; j < w; j++)
                {
                    int index = i * w + j;

                    float prob = score[index];
                    if (prob >= prob_threshold)
                    {
                        // insightface/detection/scrfd/mmdet/models/dense_heads/scrfd_head.py _get_bboxes_single()
                        float dx = bbox.channel(0)[index] * feat_stride;
                        float dy = bbox.channel(1)[index] * feat_stride;
                        float dw = bbox.channel(2)[index] * feat_stride;
                        float dh = bbox.channel(3)[index] * feat_stride;

                        // insightface/detection/scrfd/mmdet/core/bbox/transforms.py distance2bbox()
                        float cx = anchor_x + anchor_w * 0.5f;
                        float cy = anchor_y + anchor_h * 0.5f;

                        float x0 = cx - dx;
                        float y0 = cy - dy;
                        float x1 = cx + dw;
                        float y1 = cy + dh;

                        FaceInfo obj;
                        obj.rect.x = x0;
                        obj.rect.y = y0;
                        obj.rect.width = x1 - x0 + 1;
                        obj.rect.height = y1 - y0 + 1;
                        obj.prob = prob;

                        if (!kps_blob.empty())
                        {
                            const ncnn::Mat kps = kps_blob.channel_range(q * 10, 10);

                            // obj.landmark[0].x = cx + kps.channel(0)[index] * feat_stride;
                            // obj.landmark[0].y = cy + kps.channel(1)[index] * feat_stride;
                            // obj.landmark[1].x = cx + kps.channel(2)[index] * feat_stride;
                            // obj.landmark[1].y = cy + kps.channel(3)[index] * feat_stride;
                            // obj.landmark[2].x = cx + kps.channel(4)[index] * feat_stride;
                            // obj.landmark[2].y = cy + kps.channel(5)[index] * feat_stride;
                            // obj.landmark[3].x = cx + kps.channel(6)[index] * feat_stride;
                            // obj.landmark[3].y = cy + kps.channel(7)[index] * feat_stride;
                            // obj.landmark[4].x = cx + kps.channel(8)[index] * feat_stride;
                            // obj.landmark[4].y = cy + kps.channel(9)[index] * feat_stride;

                            for (int i = 0; i < 5; i++)
                            {
                                float x = cx + kps.channel(2 * i)[index] * feat_stride;
                                float y = cy + kps.channel(2 * i + 1)[index] * feat_stride;

                                obj.landmark.push_back(cv::Point2f(x, y));
                            }
                        }

                        faceobjects.push_back(obj);
                    }

                    anchor_x += feat_stride;
                }

                anchor_y += feat_stride;
            }
        }
    }


    
    void generate_points(const int target_height, const int target_width)
    {
        if (center_points_is_update)
            return;

        center_points.clear();
        // 8, 16, 32
        for (auto stride : feat_stride_fpn)
        {
            unsigned int num_grid_w = target_width / stride;
            unsigned int num_grid_h = target_height / stride;

           
            // y
            for (unsigned int i = 0; i < num_grid_h; ++i)
            {
                // x
                for (unsigned int j = 0; j < num_grid_w; ++j)
                {
                    // num_anchors, col major
                    for (unsigned int k = 0; k < num_anchors; ++k)
                    {
                        SCRFDPoint point;
                        point.cx = (float)j;
                        point.cy = (float)i;
                        point.stride = (float)stride;
                        center_points[stride].push_back(point);
                    }
                }
            }
        }

        center_points_is_update = true;
    }

    void generate_bboxes_kps_SCRFD(const YOLO5FaceScaleParams &scale_params,
                        std::vector<FaceInfo> &bbox_kps_collection,
                        ncnn::Extractor &extractor, float score_threshold,
                        float img_height, float img_width)
    {

        // score_8,score_16,score_32,bbox_8,bbox_16,bbox_32
        ncnn::Mat score_8, score_16, score_32, bbox_8, bbox_16, bbox_32;
        
        extractor.extract("score_8", score_8);
        extractor.extract("score_16", score_16);
        extractor.extract("score_32", score_32);
        extractor.extract("bbox_8", bbox_8);
        extractor.extract("bbox_16", bbox_16);
        extractor.extract("bbox_32", bbox_32);
        /*
        extractor.extract("435", score_8);//"score_8", score_8);
        extractor.extract("460", score_16);//score_16", score_16);
        extractor.extract("485", score_32);//score_32", score_32);
        extractor.extract("438", bbox_8);//bbox_8", bbox_8);
        extractor.extract("463", bbox_16);//bbox_16", bbox_16);
        extractor.extract("488", bbox_32);//bbox_32", bbox_32);
        */

        generate_points(iHeight_scrfd, iWidth_scrfd);

        bbox_kps_collection.clear();

        if (has_kps)
        {
            ncnn::Mat kps_8, kps_16, kps_32;
            
            extractor.extract("kps_8", kps_8);
            extractor.extract("kps_16", kps_16);
            extractor.extract("kps_32", kps_32);
            /*
            extractor.extract("441", kps_8);//kps_8", kps_8);
            extractor.extract("466", kps_16);//kps_16", kps_16);
            extractor.extract("491", kps_32);//kps_32", kps_32);
            */
            // level 8 & 16 & 32 with kps

            generate_bboxes_kps_single_stride_SCRFD(scale_params, score_8, bbox_8, kps_8, 8, score_threshold,
                                                          img_height, img_width, bbox_kps_collection);
            generate_bboxes_kps_single_stride_SCRFD(scale_params, score_16, bbox_16, kps_16, 16, score_threshold,
                                                          img_height, img_width, bbox_kps_collection);
            generate_bboxes_kps_single_stride_SCRFD(scale_params, score_32, bbox_32, kps_32, 32, score_threshold,
                                                          img_height, img_width, bbox_kps_collection);

        } // no kps
    }

    void generate_bboxes_kps_single_stride_SCRFD(
        const YOLO5FaceScaleParams &scale_params, ncnn::Mat &score_pred, ncnn::Mat &bbox_pred,
        ncnn::Mat &kps_pred, unsigned int stride, float score_threshold, float img_height,
        float img_width, std::vector<FaceInfo> &bbox_kps_collection)
    {
        unsigned int nms_pre_ = (stride / 8) * nms_pre; // 1 * 1000,2*1000,...
        nms_pre_ = nms_pre_ >= nms_pre ? nms_pre_ : nms_pre;

        //printf("stride : %d, score : (%d,%d), bbox : (%d,%d), kps : (%d,%d)\n", stride, score_pred.w, score_pred.h, bbox_pred.w, bbox_pred.h, kps_pred.w, kps_pred.h);
        const unsigned int num_points = score_pred.h;      // 12800
        const float *score_ptr = (float *)score_pred.data; // [1,12800,1]
        const float *bbox_ptr = (float *)bbox_pred.data;   // [1,12800,4]
        const float *kps_ptr = (float *)kps_pred.data;     // [1,12800,10]

        float ratio = scale_params.ratio;
        int dw = scale_params.dw;
        int dh = scale_params.dh;

        unsigned int count = 0;
        auto &stride_points = center_points[stride];

        for (unsigned int i = 0; i < num_points; ++i)
        {
            const float cls_conf = score_ptr[i];

            //if(max_score < cls_conf)
            //    max_score = cls_conf;

            if (cls_conf < score_threshold)
                continue; // filter

            auto &point = stride_points[i];//.at(i);
            const float cx = point.cx;    // cx
            const float cy = point.cy;    // cy
            const float s = point.stride; // stride

            // bbox
            const float *offsets = bbox_ptr + i * 4;
            float l = offsets[0]; // left
            float t = offsets[1]; // top
            float r = offsets[2]; // right
            float b = offsets[3]; // bottom

            FaceInfo box_kps;
            float x1 = ((cx - l) * s - (float)dw) / ratio; // cx - l x1
            float y1 = ((cy - t) * s - (float)dh) / ratio; // cy - t y1
            float x2 = ((cx + r) * s - (float)dw) / ratio; // cx + r x2
            float y2 = ((cy + b) * s - (float)dh) / ratio; // cy + b y2
            box_kps.rect.x = std::max(0.f, x1);
            box_kps.rect.y = std::max(0.f, y1);
            box_kps.rect.width = std::min(img_width, x2 - x1);
            box_kps.rect.height = std::min(img_height, y2 - y1);
            box_kps.prob = cls_conf;

            // landmarks
            const float *kps_offsets = kps_ptr + i * 10;
            for (unsigned int j = 0; j < 10; j += 2)
            {
                cv::Point2f kps;
                float kps_l = kps_offsets[j];
                float kps_t = kps_offsets[j + 1];
                float kps_x = ((cx + kps_l) * s - (float)dw) / ratio; // cx - l x
                float kps_y = ((cy + kps_t) * s - (float)dh) / ratio; // cy - t y
                kps.x = std::min(std::max(0.f, kps_x), img_width);
                kps.y = std::min(std::max(0.f, kps_y), img_height);
                box_kps.landmark.push_back(kps);
            }

            if(x1 < 0 || y1 < 0 || x2 > img_width || y2 > img_height)
            {
                //continue;
            }   

            bbox_kps_collection.push_back(box_kps);

            count += 1; // limit boxes for nms.
            if (count > max_nms)
                break;
        }
        
        if (bbox_kps_collection.size() > nms_pre_)
        {
            std::sort(
                bbox_kps_collection.begin(), bbox_kps_collection.end(),
                [](const FaceInfo &a, const FaceInfo &b)
                { return a.prob > b.prob; }); // sort inplace
            // trunc
            bbox_kps_collection.resize(nms_pre_);
        }
    }


    bool initialize_SCRFD3(ncnn::Net &net, const std::string &param_path, const std::string &bin_path)
    {
        has_kps = true;

        int sucess_load_param = net.load_param_enc(param_path.c_str(), 6, "s1face");
        int sucess_load_model = net.load_model_enc(bin_path.c_str(), 6, "s1face");
        int sucess_load_result = sucess_load_model + sucess_load_param;

        if (sucess_load_result)
        {
            if (sucess_load_result)
                Utils::Logging("        <<< S1 NCNN Detector >>> : %s - Fail to load param\n", __FUNCTION__);

            if (sucess_load_result)
                Utils::Logging("        <<< S1 NCNN Detector >>> : %s - Fail to load model\n", __FUNCTION__);
        }
        else
        {
            Utils::Logging("        <<< S1 NCNN Detector >>> : %s - Load model result : %d %d\n", __FUNCTION__, sucess_load_param, sucess_load_model);
        }

        bool init_sucess = false;
        if (sucess_load_result == 0)
            init_sucess = true;

        return init_sucess;
    }


    bool detect_SCRFD3(ncnn::Net &net, cv::Mat &img, std::vector<FaceInfo> &faceobjects, int t_width, int t_height)
    {

       int width = img.cols;
        int height = img.rows;

        float prob_threshold = 0.50;
        float nms_threshold = 0.45;

        /************************************************************************************************/
        float scale = std::min(t_width / (float)width, t_height / (float)height);
        float scale_w = t_width / (float)width; //0.266666
        float scale_h = t_height / (float)height;   //0.25

        int nw = width * scale;
        if (nw % 32 > 0)
            nw += 32 - nw % 32;
        int nh = height * scale;
        if (nh % 32 > 0)
            nh += 32 - nh % 32;

        int max_side = std::max(nw, nh);

        cv::Mat img_resize;
        cv::resize(img, img_resize, cv::Size(0, 0), scale, scale);

        cv::Mat img_pad = cv::Mat::zeros(cv::Size(t_width, t_height), CV_8UC3);
        img_resize.copyTo(img_pad(cv::Rect(0, 0, img_resize.cols, img_resize.rows)));

        ncnn::Mat in_pad = ncnn::Mat::from_pixels(img_pad.data, ncnn::Mat::PIXEL_RGB, t_width, t_height);
        /************************************************************************************************/
        int wpad = 0;
        int hpad = 0;

        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
        const float norm_vals[3] = {1 / 128.f, 1 / 128.f, 1 / 128.f};
        in_pad.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = net.create_extractor();

        ex.input("input.1", in_pad);

        std::vector<FaceInfo> faceproposals;

        std::vector<std::string> sNames{"score_8", "score_16", "score_32"};//{"435", "460", "485"};
        std::vector<std::string> bNames{"bbox_8", "bbox_16", "bbox_32"};//{"438", "463", "488"};
        std::vector<std::string> lNames{"kps_8", "kps_16", "kps_32"};//{"441", "466", "491"};
        std::vector<int> strides{8, 16, 32};
        
        cv::Rect2f image_Rect = cv::Rect2f(0.0f, 0.0f, (float)img.cols, (float)img.rows);

        for (int sIdx = 0; sIdx < 3; sIdx++)
        {
            int stride = strides[sIdx];
        
            ncnn::Mat score_blob, bbox_blob, kps_blob;
            ex.extract(sNames[sIdx].c_str(), score_blob);
            ex.extract(bNames[sIdx].c_str(), bbox_blob);
            ex.extract(lNames[sIdx].c_str(), kps_blob);

            int w_fSize = t_width / stride;   //ex) 160 / 8 = 20  // <- width용
            int h_fSize = t_height / stride;

            int len = score_blob.h;
            //bbox_blob = bbox_blob.reshape(len, 4);
            //kps_blob = kps_blob.reshape(len, 10);
            
            const float* score_ptr = (float*)score_blob.data;
    
            //const ncnn::Mat bbox = bbox_blob.row_range(0, 4);
            //const ncnn::Mat kps = kps_blob.row_range(0, 10);
            printf("score : (%d, %d), bbox : (%d, %d), kps : (%d, %d)\n", score_blob.w, score_blob.h, bbox_blob.w, bbox_blob.h, kps_blob.w, kps_blob.h);
            
            for(int iIdx = 0; iIdx < len/2; iIdx++)
                for(int ianchor=0; ianchor<2; ianchor++)
                {
                    int index_score = iIdx + ianchor*len/2;
                    int index_box = iIdx + ianchor*len * 2;
                    int index_landmark = iIdx + ianchor*len*5;

                    if (score_blob[index_score] < prob_threshold)
                        continue;
                    
                    if (score_blob[index_score] == 1)
                        continue;
                    
                    FaceInfo face;
                    face.prob = score_blob[index_score];

                    int au = iIdx % w_fSize;
                    int av = iIdx / w_fSize;

                    /*
                    printf("ianchor : %d, iIdx : %d, index : %d, score : %f, box : (%f, %f, %f, %f), landmark(%f, %f, %f, %f)\n", 
                    ianchor, iIdx, index, 
                    score_blob[index_score], 
                    bbox_blob[index_box], bbox_blob[index_box + len/2], bbox_blob[index_box + 2*len/2], bbox_blob[index_box + 3*len/2],
                    kps_blob[index_landmark], kps_blob[index_landmark + len/2], kps_blob[index_landmark + len/2*2], kps_blob[index_landmark + len/2*3]);
                    */

                    float x_l = au - bbox_blob[index_box];
                    float y_l = av - bbox_blob[index_box + len/2];

                    face.rect.x = std::min((float)width - 1, std::max(0.f, x_l)) * stride / scale_h;
                    face.rect.y = std::min((float)height - 1, std::max(0.f, y_l)) * stride / scale_w;
                    
                    float x_r = au + bbox_blob[index_box + 2*len/2];
                    float y_r = av + bbox_blob[index_box + 3*len/2];
                    face.rect.width = std::min((float)width - 1, std::max(0.f, x_r - x_l)) * stride / scale_h;
                    face.rect.height = std::min((float)height - 1, std::max(0.f, y_r - y_l)) * stride / scale_w;

                    for (int i = 0; i < 5; i++)
                    {
                        float x = (kps_blob[index_landmark + 2*i*len/2] + au) * stride / scale_h;
                        float y = (kps_blob[index_landmark + (2*i+1)*len/2] + av) * stride / scale_w;
                        face.landmark.push_back(cv::Point2f(x, y));
                    }

                    cv::Rect2f iRect = image_Rect & face.rect;
                    if(image_Rect.area() / face.rect.area() < 0.3)
                        continue;

                    faceproposals.push_back(face);

                }
            }

        // sort all proposals by score from highest to lowest
        qsort_descent_inplace(faceproposals);

        // apply nms with nms_threshold
        std::vector<int> picked;
        nms_sorted_bboxes(faceproposals, picked, nms_threshold);

        int face_count = picked.size();

        faceobjects.resize(face_count);
        for (int i = 0; i < face_count; i++)
        {
             faceobjects[i] = faceproposals[picked[i]];
        }
        // for (int i = 0; i < face_count; i++)
        // {
        //     faceobjects[i] = faceproposals[picked[i]];

        //     // adjust offset to original unpadded
        //     float x0 = (faceobjects[i].rect.x - (wpad / 2)) / scale;
        //     float y0 = (faceobjects[i].rect.y - (hpad / 2)) / scale;
        //     float x1 = (faceobjects[i].rect.x + faceobjects[i].rect.width - (wpad / 2)) / scale;
        //     float y1 = (faceobjects[i].rect.y + faceobjects[i].rect.height - (hpad / 2)) / scale;

        //     x0 = std::max(std::min(x0, (float)width - 1), 0.f);
        //     y0 = std::max(std::min(y0, (float)height - 1), 0.f);
        //     x1 = std::max(std::min(x1, (float)width - 1), 0.f);
        //     y1 = std::max(std::min(y1, (float)height - 1), 0.f);

        //     faceobjects[i].rect.x = x0;
        //     faceobjects[i].rect.y = y0;
        //     faceobjects[i].rect.width = x1 - x0;
        //     faceobjects[i].rect.height = y1 - y0;

        //     if (has_kps)
        //     {
        //         float x0 = (faceobjects[i].landmark[0].x - (wpad / 2)) / scale;
        //         float y0 = (faceobjects[i].landmark[0].y - (hpad / 2)) / scale;
        //         float x1 = (faceobjects[i].landmark[1].x - (wpad / 2)) / scale;
        //         float y1 = (faceobjects[i].landmark[1].y - (hpad / 2)) / scale;
        //         float x2 = (faceobjects[i].landmark[2].x - (wpad / 2)) / scale;
        //         float y2 = (faceobjects[i].landmark[2].y - (hpad / 2)) / scale;
        //         float x3 = (faceobjects[i].landmark[3].x - (wpad / 2)) / scale;
        //         float y3 = (faceobjects[i].landmark[3].y - (hpad / 2)) / scale;
        //         float x4 = (faceobjects[i].landmark[4].x - (wpad / 2)) / scale;
        //         float y4 = (faceobjects[i].landmark[4].y - (hpad / 2)) / scale;

        //         faceobjects[i].landmark[0].x = std::max(std::min(x0, (float)width - 1), 0.f);
        //         faceobjects[i].landmark[0].y = std::max(std::min(y0, (float)height - 1), 0.f);
        //         faceobjects[i].landmark[1].x = std::max(std::min(x1, (float)width - 1), 0.f);
        //         faceobjects[i].landmark[1].y = std::max(std::min(y1, (float)height - 1), 0.f);
        //         faceobjects[i].landmark[2].x = std::max(std::min(x2, (float)width - 1), 0.f);
        //         faceobjects[i].landmark[2].y = std::max(std::min(y2, (float)height - 1), 0.f);
        //         faceobjects[i].landmark[3].x = std::max(std::min(x3, (float)width - 1), 0.f);
        //         faceobjects[i].landmark[3].y = std::max(std::min(y3, (float)height - 1), 0.f);
        //         faceobjects[i].landmark[4].x = std::max(std::min(x4, (float)width - 1), 0.f);
        //         faceobjects[i].landmark[4].y = std::max(std::min(y4, (float)height - 1), 0.f);
        //     }
        // }

        if (faceobjects.size() > 0)
            return true;
        else
            return false;
    }



} // namespace NCNNEngine
