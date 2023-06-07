/*
yolov8 segmentation simplest demo using opencv dnn and onnxruntime
winxos 20230607
*/
#include<iostream>
#include<memory>
#include <chrono>
using namespace std;

#include<opencv2/opencv.hpp>
using namespace cv;
using namespace cv::dnn;

#include<onnxruntime_cxx_api.h>
using namespace Ort;

struct Obj {
	int id = 0;
	float accu = 0.0;
	Rect bound;
	Mat mask;
};
int seg_ch = 32;
int seg_w = 160, seg_h = 160;
int net_w = 640, net_h = 640;
float accu_thresh = 0.25, mask_thresh = 0.5;

struct ImageInfo {
	Size raw_size;
	Vec4d trans;
};
vector<string> class_names = { "bean","corn","impurity","rice","ricebrown","ricemilled","wheat","xian" };

void get_mask(const Mat& mask_info, const Mat& mask_data, const ImageInfo& para, Rect bound, Mat& mast_out)
{
	Vec4f trans = para.trans;
	int r_x = floor((bound.x * trans[0] + trans[2]) / net_w * seg_w);
	int r_y = floor((bound.y * trans[1] + trans[3]) / net_h * seg_h);
	int r_w = ceil(((bound.x + bound.width) * trans[0] + trans[2]) / net_w * seg_w) - r_x;
	int r_h = ceil(((bound.y + bound.height) * trans[1] + trans[3]) / net_h * seg_h) - r_y;
	r_w = MAX(r_w, 1);
	r_h = MAX(r_h, 1);
	if (r_x + r_w > seg_w) //crop
	{
		seg_w - r_x > 0 ? r_w = seg_w - r_x : r_x -= 1;
	}
	if (r_y + r_h > seg_h)
	{
		seg_h - r_y > 0 ? r_h = seg_h - r_y : r_y -= 1;
	}
	vector<Range> roi_rangs = { Range(0, 1) ,Range::all() , Range(r_y, r_h + r_y) ,Range(r_x, r_w + r_x) };
	Mat temp_mask = mask_data(roi_rangs).clone();
	Mat protos = temp_mask.reshape(0, { seg_ch,r_w * r_h });
	Mat matmul_res = (mask_info * protos).t();
	Mat masks_feature = matmul_res.reshape(1, { r_h,r_w });
	Mat dest;
	exp(-masks_feature, dest);//sigmoid
	dest = 1.0 / (1.0 + dest);
	int left = floor((net_w / seg_w * r_x - trans[2]) / trans[0]);
	int top = floor((net_h / seg_h * r_y - trans[3]) / trans[1]);
	int width = ceil(net_w / seg_w * r_w / trans[0]);
	int height = ceil(net_h / seg_h * r_h / trans[1]);
	Mat mask;
	resize(dest, mask, Size(width, height));
	mast_out = mask(bound - Point(left, top)) > mask_thresh;
}

void draw_result(Mat img, vector<Obj>& result, vector<Scalar> color)
{
	Mat mask = img.clone();
	for (int i = 0; i < result.size(); i++)
	{
		int left, top;
		left = result[i].bound.x;
		top = result[i].bound.y;
		int color_num = i;
		rectangle(img, result[i].bound, color[result[i].id], 8);
		if (result[i].mask.rows && result[i].mask.cols > 0)
		{
			mask(result[i].bound).setTo(color[result[i].id], result[i].mask);
		}
		string label = std::format("{}:{:.2f}", class_names[result[i].id], result[i].accu);
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 2, color[result[i].id], 4);
	}
	addWeighted(img, 0.6, mask, 0.4, 0, img); //add mask to src
	resize(img, img, Size(640, 640));
	imshow("img", img);
	waitKey();
}
void decode_output(Mat& output0, Mat& output1, ImageInfo para, vector<Obj>& output)
{
	output.clear();
	vector<int> class_ids;
	vector<float> accus;
	vector<Rect> boxes;
	vector<vector<float>> masks;
	int data_width = class_names.size() + 4 + 32;
	int rows = output0.rows;
	float* pdata = (float*)output0.data;
	for (int r = 0; r < rows; ++r)
	{
		Mat scores(1, class_names.size(), CV_32FC1, pdata + 4);
		Point class_id;
		double max_socre;
		minMaxLoc(scores, 0, &max_socre, 0, &class_id);
		if (max_socre >= accu_thresh)
		{
			masks.push_back(vector<float>(pdata + 4 + class_names.size(), pdata + data_width));
			float w = pdata[2] / para.trans[0];
			float h = pdata[3] / para.trans[1];
			int left = MAX(int((pdata[0] - para.trans[2]) / para.trans[0] - 0.5 * w + 0.5), 0);
			int top = MAX(int((pdata[1] - para.trans[3]) / para.trans[1] - 0.5 * h + 0.5), 0);
			class_ids.push_back(class_id.x);
			accus.push_back(max_socre);
			boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
		}
		pdata += data_width;//next line
	}
	vector<int> nms_result;
	NMSBoxes(boxes, accus, accu_thresh, mask_thresh, nms_result);
	for (int i = 0; i < nms_result.size(); ++i)
	{
		int idx = nms_result[i];
		boxes[idx] = boxes[idx] & Rect(0, 0, para.raw_size.width, para.raw_size.height);
		Obj result = { class_ids[idx] ,accus[idx] ,boxes[idx] };
		get_mask(Mat(masks[idx]).t(), output1, para, boxes[idx], result.mask);
		output.push_back(result);
	}
}
bool detect_seg_dnn(string& model_path, Mat& srcImg, vector<Obj>& objs)
{
	Net net = readNet(model_path.c_str()); //default DNN_TARGET_CPU
	ImageInfo para = { srcImg.size() ,{ 640.0 / srcImg.cols ,640.0 / srcImg.rows,0,0 } };
	Mat image;
	resize(srcImg, image, Size(640, 640));
	Mat blob = blobFromImage(image, 1 / 255.0, Size(640, 640), Scalar(0, 0, 0), true, false);
	net.setInput(blob);
	vector<Mat> outputs;
	auto start = chrono::high_resolution_clock::now();
	net.forward(outputs, vector<string>{ "output0", "output1" }); //get outputs
	auto end = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
	cout << "dnn time: " << duration << " millis." << endl;
	Mat output0 = Mat(Size(outputs[0].size[2], outputs[0].size[1]), CV_32F, (float*)outputs[0].data).t();
	decode_output(output0, outputs[1], para, objs);
	return objs.size() > 0;
}
bool detect_seg_ort(string& mpath, Mat& img, vector<Obj>& objs)
{
	Mat image;
	resize(img, image, Size(640, 640));
	Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "yolov8");
	Session* session = new Session(env,
		wstring(mpath.begin(), mpath.end()).c_str(), SessionOptions());
	vector<const char*> input_names = { "images" };
	vector<const char*> output_names = { "output0","output1" };
	vector<int64_t> input_shape = { 1, 3, 640, 640 };
	Mat blob = blobFromImage(image, 1 / 255.0, Size(640, 640), Scalar(0, 0, 0), true, false);
	Value input_tensor = Value::CreateTensor<float>(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
		(float*)blob.data, 3 * 640 * 640, input_shape.data(), input_shape.size());
	auto start = chrono::high_resolution_clock::now();
	auto outputs = session->Run(RunOptions{ nullptr },
		input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());
	auto end = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
	cout << "ort time: " << duration << " millis.";
	float* all_data = outputs[0].GetTensorMutableData<float>();
	auto data_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	Mat output0 = Mat(Size((int)data_shape[2], (int)data_shape[1]), CV_32F, all_data).t();
	auto mask_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
	vector<int> mask_sz = { 1,(int)mask_shape[1],(int)mask_shape[2],(int)mask_shape[3] };
	Mat output1 = Mat(mask_sz, CV_32F, outputs[1].GetTensorMutableData<float>());
	ImageInfo img_info = { img.size(), { 640.0 / img.cols ,640.0 / img.rows,0,0 } };
	decode_output(output0, output1, img_info, objs);
	return objs.size() > 0;
}
int main()
{
	srand(time(0));
	vector<Scalar> color;
	for (int i = 0; i < class_names.size(); i++)
	{
		color.push_back(Scalar(rand() % 128 + 128, rand() % 128 + 128, rand() % 128 + 128));
	}
	string model_path = "./bestm.onnx";
	Mat img1 = imread("./1.jpg");
	Mat img2 = img1.clone();
	vector<Obj> result;
	if (detect_seg_dnn(model_path, img1, result))
	{
		draw_result(img1, result, color);
	}
	if (detect_seg_ort(model_path, img2, result))
	{
		draw_result(img2, result, color);
	}
	return 0;
}
