
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <torch/script.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <time.h>
#include <USBProbe.hpp>


std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float score_thresh = 0.5, float iou_thresh = 0.5)
{
	std::vector<torch::Tensor> output;
	for (size_t i = 0; i < preds.sizes()[0]; ++i)
	{
		torch::Tensor pred = preds.select(0, i);

		
		torch::Tensor scores = pred.select(1, 4) * std::get<0>(torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));
		pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
		if (pred.sizes()[0] == 0) continue;

		
		pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
		pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
		pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
		pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

		
		std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
		pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
		pred.select(1, 5) = std::get<1>(max_tuple);

		torch::Tensor  dets = pred.slice(1, 0, 6);

		torch::Tensor keep = torch::empty({ dets.sizes()[0] });
		torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
		std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
		torch::Tensor v = std::get<0>(indexes_tuple);
		torch::Tensor indexes = std::get<1>(indexes_tuple);
		int count = 0;
		while (indexes.sizes()[0] > 0)
		{
			keep[count] = (indexes[0].item().toInt());
			count += 1;

			
			torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1);
			torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1);
			torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1);
			torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1);
			torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1);
			torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1);
			for (size_t i = 0; i < indexes.sizes()[0] - 1; ++i)
			{
				lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i + 1]][0].item().toFloat());
				tops[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i + 1]][1].item().toFloat());
				rights[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i + 1]][2].item().toFloat());
				bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[i + 1]][3].item().toFloat());
				widths[i] = std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
				heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
			}
			torch::Tensor overlaps = widths * heights;

			
			torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
			indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
		}
		keep = keep.toType(torch::kInt64);
		output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
	}
	return output;
}

using namespace cv;
using namespace std;

int main()
{
	
	torch::jit::script::Module module = torch::jit::load("../yamada_best.torchscript256.pt");

	std::vector<std::string> classnames;
	std::ifstream f("../yamada.names");
	std::string name = "";
	while (std::getline(f, name))
	{
		classnames.push_back(name);
	}

	//-------------------------------she xiang tou
	VideoCapture capture(0);

	if (!capture.isOpened())
		return -1;

	capture.set(CV_CAP_PROP_FRAME_WIDTH, 720);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
	capture.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
	capture.set(CV_CAP_PROP_FPS, 15);


	Mat frame_camera;

	chrono::system_clock::time_point start, end;

	//-------------------------------du qu wen ben
	
	
	string data;
	string num1;
	string num2;
	string num3;
	string num4;


	Point text_position1(943, 325);
	Point text_position2(1273, 325);
	Point text_position3(943, 410);
	Point text_position4(1273, 410);

	Point text_position5(428, 795);
	Point text_position6(635, 788);


	//-------------------------------
	USBProbe usb_probe;
	bool freeze_flg = false;
	bool working = true;
	usb_probe.setConversionParams(220, 750, 60);
	usb_probe.disableFreeze();

	cv::Mat frame, img, raw,raw1;
	//----------------------------------

	for (;;) {
		//------------------------
		
		//-----------------------du qu wen dang
		ifstream file;
		file.open("D://classification//test.txt");

		while (getline(file, data)) {
			num1 = data.substr(11, 6);
			num2 = data.substr(40, 6);
			num3 = data.substr(59, 6);
			num4 = data.substr(71, 6);
		}

		double num1_1 = atof(num1.c_str()) * 100;
		double num2_2 = atof(num2.c_str()) * 100;
		double num3_3 = atof(num3.c_str()) * 100;
		double num4_4 = atof(num4.c_str()) * 100;

		//string num1_1_1 = to_string(num1_1);
		//string num2_2_2 = to_string(num2_2);;
		//string num3_3_3 = to_string(num3_3);;
		//string num4_4_4 = to_string(num4_4);;


		cout << num4_4 << "%" << endl;


		//-----------------------shexiangtou

		capture >> frame_camera;
		flip(frame_camera, frame_camera, 1);
		cv::resize(frame_camera, frame_camera, cv::Size(380,380));
		//cv::imwrite("pic.jpg", frame_camera);

		//-----------------------

		auto raw = usb_probe.read();
		clock_t start = clock();

		if (!raw.empty()) {
			
			cv:: resize(raw, raw1, cv::Size(224,224));
			cv::imwrite("pic.jpg", raw1);
			cv::cvtColor(raw, frame, cv::COLOR_GRAY2RGB);
			cv::resize(frame, img, cv::Size(192, 256));
			//cv::imwrite("pic3.jpg", img);
			

			torch::Tensor imgTensor = torch::from_blob(img.data, { img.rows, img.cols,3 }, torch::kByte);
			imgTensor = imgTensor.permute({ 2,0,1 });
			imgTensor = imgTensor.toType(torch::kFloat);
			imgTensor = imgTensor.div(255);
			imgTensor = imgTensor.unsqueeze(0);
			torch::Tensor preds = module.forward({ imgTensor }).toTuple()->elements()[0].toTensor();
			std::vector<torch::Tensor> dets = non_max_suppression(preds, 0.4, 0.5);

			if (dets.size() > 0 )
			{
				cv::Mat src_background = cv::imread("test10.png");
				for (size_t i = 0; i < dets[0].sizes()[0]; ++i)
				{
					float left = dets[0][i][0].item().toFloat() * frame.cols / 192;
					float top = dets[0][i][1].item().toFloat() * frame.rows / 256;
					float right = dets[0][i][2].item().toFloat() * frame.cols / 192;
					float bottom = dets[0][i][3].item().toFloat() * frame.rows / 256;
					float score = dets[0][i][4].item().toFloat();
					int classID = dets[0][i][5].item().toInt();


					cv::rectangle(frame, cv::Rect(left, top, (right - left), (bottom - top)), cv::Scalar(0, 255, 0), 2);

					//-------------------------------------------------------------------
					Point center1 = ((left + (right - left) / 2), (top + (bottom - top) / 2) );
					cout << center1.x << endl;
					stringstream ss;
					int num_center1 = (center1.x - 379.5) * 0.0974967 * 0.5;
					string num_center11;
					ss << num_center1;
					ss >> num_center11;
					//-------------------------------------------------------------------


					//cv::putText(frame,
					//	classnames[classID] + ": " + cv::format("%.2f", score),
					//	cv::Point(left, ( top -15 )),
					//	cv::FONT_HERSHEY_SIMPLEX, (right - left) / 300, cv::Scalar(0, 255, 0), 1.5);

					Mat imageRoi1 = src_background(Rect(10, 130, frame.cols, frame.rows));
					Mat imageRoi2 = src_background(Rect(1020, 485, frame_camera.cols, frame_camera.rows));

					Mat mask1 = frame;
					Mat mask2 = frame_camera;

					frame.copyTo(imageRoi1, mask1);
					frame_camera.copyTo(imageRoi2, mask2);

					putText(src_background, num1, text_position1, FONT_HERSHEY_TRIPLEX, 1, Scalar(255, 255, 255), 2.5);
					putText(src_background, num2, text_position2, FONT_HERSHEY_TRIPLEX, 1, Scalar(255, 255, 255), 2.5);
					putText(src_background, num3, text_position3, FONT_HERSHEY_TRIPLEX, 1, Scalar(255, 255, 255), 2.5);
					putText(src_background, num4, text_position4, FONT_HERSHEY_TRIPLEX, 1, Scalar(255, 255, 255), 2.5);

					putText(src_background, cv::format("%.2f", score), text_position5, FONT_HERSHEY_TRIPLEX, 1.5, Scalar(0, 255, 0), 2.5);
					putText(src_background, num_center11, text_position6, FONT_HERSHEY_TRIPLEX, 0.9, Scalar(255, 191, 0), 2.5);
				}	
				cv::imshow("test", src_background);
			}
			else
			{
				cv::Mat src_background = cv::imread("test7.png");
				Mat imageRoi1 = src_background(Rect(10, 130, frame.cols, frame.rows));
				Mat imageRoi2 = src_background(Rect(1020, 485, frame_camera.cols, frame_camera.rows));

				Mat mask1 = frame;
				Mat mask2 = frame_camera;

				frame.copyTo(imageRoi1, mask1);
				frame_camera.copyTo(imageRoi2, mask2);

				putText(src_background, num1, text_position1, FONT_HERSHEY_TRIPLEX, 1, Scalar(255, 255, 255), 2.5);
				putText(src_background, num2, text_position2, FONT_HERSHEY_TRIPLEX, 1, Scalar(255, 255, 255), 2.5);
				putText(src_background, num3, text_position3, FONT_HERSHEY_TRIPLEX, 1, Scalar(255, 255, 255), 2.5);
				putText(src_background, num4, text_position4, FONT_HERSHEY_TRIPLEX, 1, Scalar(255, 255, 255), 2.5);

				cv::imshow("test", src_background);
			}
			
			if (cv::waitKey(1) == 27) break;


		}

	}

	return 0;
}
