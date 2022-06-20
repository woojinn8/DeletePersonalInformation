#include "ncnn_detector.hpp"

namespace NCNNEngine
{
    // Variables
    ncnn::Net net_detector;
    bool has_kps;

    int mode_detector;
    int num_thread, powersave;
    float iou_threshold, score_threshold;
    bool useGPU, useFP16;
    int input_width, input_height;
    
    int iWidth;
    int iHeight;

    typedef struct
    {
        float ratio;
        int dw;
        int dh;
        bool flag;
    } ScaleParams;

    // Variables (YoloV5Face)
    typedef struct
    {
        int grid0;
        int grid1;
        int stride;
        float width;
        float height;
    } YOLO5FaceAnchor;

    int max_face = 5;
    std::vector<unsigned int> strides = {8, 16, 32};
    static constexpr const unsigned int nms_pre = 1000;
    static constexpr const unsigned int max_nms = 30000;
    bool center_anchors_is_update;
    std::unordered_map<unsigned int, std::vector<YOLO5FaceAnchor>> center_anchors;

    
    // Variables (SCRFD)
    typedef struct
    {
      float cx;
      float cy;
      float stride;
    } SCRFDPoint;
    
    std::vector<int> feat_stride_fpn = {8, 16, 32}; // steps, may [8, 16, 32, 64, 128]
    std::unordered_map<int, std::vector<SCRFDPoint>> center_points;
    bool center_points_is_update = false;
    unsigned int fmc = 3; // feature map count

    
    // func
    bool initialize_YOLO5Face(ncnn::Net &net, const std::string &param_path, const std::string &bin_path);
    bool detect_YOLO5Face(ncnn::Net &net, cv::Mat &img, std::vector<FaceInfo> &face_list, int target_input_width, int target_input_height);

    bool initialize_SCRFD2(ncnn::Net &net, const std::string &param_path, const std::string &bin_path);
    bool detect_SCRFD2(ncnn::Net &net, cv::Mat &img, std::vector<FaceInfo> &face_list, int target_input_width, int target_input_height);

    Detector_NCNN::Detector_NCNN()
    {
    }

    Detector_NCNN::~Detector_NCNN()
    {
        net_detector.clear();
    }

    // common
    float intersection_area(const FaceInfo &a, const FaceInfo &b);
    void qsort_descent_inplace(std::vector<FaceInfo> &faceobjects, int left, int right);
    void qsort_descent_inplace(std::vector<FaceInfo> &faceobjects);
    void nms_sorted_bboxes(const std::vector<FaceInfo> &faceobjects, std::vector<int> &picked, float iou_threshold);
    float cal_iou(FaceInfo a, FaceInfo b);
    static inline float sigmoid(float x);
    void nms_bboxes_kps(std::vector<FaceInfo> &input,
                        std::vector<FaceInfo> &output,
                        float iou_threshold, unsigned int topk);
    void resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                        int target_height, int target_width,
                        ScaleParams &scale_params);
    // for YoloV5Face
    void generate_bboxes_kps_YOLO5FACE(const ScaleParams &scale_params,
                             std::vector<FaceInfo> &bbox_kps_collection,
                             ncnn::Extractor &extractor, float score_threshold,
                             float img_height, float img_width);

    void generate_anchors_YOLO5FACE(unsigned int target_height, unsigned int target_width);
    void generate_bboxes_kps_single_stride_YOLO5FACE(const ScaleParams &scale_params,
                                           ncnn::Mat &det_pred, unsigned int stride,
                                           float score_threshold, float img_height, float img_width,
                                           std::vector<FaceInfo> &bbox_kps_collection);

    // for SCRFD
    void generate_points(const int target_height, const int target_width);
    void generate_bboxes_kps_SCRFD(const ScaleParams &scale_params,
                        std::vector<FaceInfo> &bbox_kps_collection,
                        ncnn::Extractor &extractor, float score_threshold,
                        float img_height, float img_width);

    void generate_bboxes_kps_single_stride_SCRFD(
        const ScaleParams &scale_params, ncnn::Mat &score_pred, ncnn::Mat &bbox_pred,
        ncnn::Mat &kps_pred, unsigned int stride, float score_threshold, float img_height,
        float img_width, std::vector<FaceInfo> &bbox_kps_collection);


    bool Detector_NCNN::Initialize(ModelInfo &mode_info)
    {
        bool bInit = false;
        
        mode_detector = mode_info.mode_detector;

        num_thread = mode_info.num_thread;
        powersave = mode_info.powersave;

        useGPU = mode_info.use_gpu; 
        useFP16 = mode_info.use_fp16;

        score_threshold = mode_info.score_threshold;
        iou_threshold = mode_info.iou_threshold;

        input_width = mode_info.image_width;
        input_height = mode_info.image_height;

        // mode
        // 0 : SCRFD
        // 1 : Yolo5Face
        switch (mode_detector)
        {
        case 0:
            bInit = initialize_YOLO5Face(net_detector, mode_info.model_param, mode_info.model_bin);
            break;
        case 1:
            bInit = initialize_SCRFD2(net_detector, mode_info.model_param, mode_info.model_bin);
            break;
        default:
            break;
        }

        return bInit;
    }

    bool Detector_NCNN::detect(cv::Mat &img, std::vector<DetectInfo> &detected_list)
    {
        bool bDetected = false;

        switch (mode_detector)
        {
        case 0:
            bDetected = detect_YOLO5Face(net_detector, img, detected_list, input_width, input_height);
            break;
        case 1:
            bDetected = detect_SCRFD2(net_detector, img, detected_list, input_width, input_height);
        default:
            break;
        }
        
        return bDetected;
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
        ScaleParams scale_params;

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
        generate_bboxes_kps_YOLO5FACE(scale_params, bbox_kps_collection, extractor,
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
        ScaleParams scale_params;
        
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
                        ScaleParams &scale_params)
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

    void generate_bboxes_kps_YOLO5FACE(const ScaleParams &scale_params,
                             std::vector<FaceInfo> &bbox_kps_collection,
                             ncnn::Extractor &extractor, float score_threshold,
                             float img_height, float img_width)
    {
        // (1,n,16=4+1+10+1=cxcy+cwch+obj_conf+5kps+cls_conf)
        ncnn::Mat det_stride_8, det_stride_16, det_stride_32;
        extractor.extract("det_stride_8", det_stride_8);
        extractor.extract("det_stride_16", det_stride_16);
        extractor.extract("det_stride_32", det_stride_32);

        generate_anchors_YOLO5FACE(iHeight, iWidth);

        // generate bounding boxes.
        bbox_kps_collection.clear();

        generate_bboxes_kps_single_stride_YOLO5FACE(scale_params, det_stride_8, 8, score_threshold,
                                          img_height, img_width, bbox_kps_collection);
        generate_bboxes_kps_single_stride_YOLO5FACE(scale_params, det_stride_16, 16, score_threshold,
                                          img_height, img_width, bbox_kps_collection);
        generate_bboxes_kps_single_stride_YOLO5FACE(scale_params, det_stride_32, 32, score_threshold,
                                          img_height, img_width, bbox_kps_collection);

        // std::cout << "generate_bboxes_kps num: " << bbox_kps_collection.size() << "\n";
    }

    void generate_anchors_YOLO5FACE(unsigned int target_height, unsigned int target_width)
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

    void generate_bboxes_kps_single_stride_YOLO5FACE(const ScaleParams &scale_params,
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

    void generate_bboxes_kps_SCRFD(const ScaleParams &scale_params,
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
        const ScaleParams &scale_params, ncnn::Mat &score_pred, ncnn::Mat &bbox_pred,
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
    
} // namespace NCNNEngine
