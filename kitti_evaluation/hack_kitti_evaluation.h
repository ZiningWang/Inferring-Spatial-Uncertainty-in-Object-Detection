#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <numeric>
#include <strings.h>
using namespace std;
typedef vector<vector<int32_t> > vvvi;
typedef vector<vector<double> >  vvvd;
typedef vector<vector<bool> > vvvb;


void write_detection_file(string result_dir, vector<int32_t> indices, vvvi &accepted_det_3D, vvvi &accepted_det_G, vvvi &accepted_det_2D,
    vvvb &fp_det_3D, vvvb &fp_det_G, vvvb &fp_det_2D, vvvd &overlap_det_3D, vvvd &overlap_det_G, vvvd &overlap_det_2D, string difficulty){
	ofstream fdet;
	fdet.open((result_dir + "/hack_det_" + difficulty + "_evaluation.txt").c_str());
	fdet << "frame, index, fp_det3D, fpGr, fp2D, associated_gt3D, gtGr, gt2D, overlap_3D,ov_Gr,ov_2D" << endl;
	size_t nframe = accepted_det_2D.size();
	assert(nframe==indices.size());
	int fp_det3, fp_detG, fp_det2;
	int ass_gt3, ass_gtG, ass_gt2;
	for (int frameNum=0; frameNum<accepted_det_3D.size();frameNum++){
		vector<int32_t> i_a_det2 = accepted_det_2D[frameNum];
		vector<double> i_o_det2 = overlap_det_2D[frameNum];
		vector<bool> i_fp_det2 = fp_det_2D[frameNum];
		vector<int32_t> i_a_det3;
		vector<double> i_o_det3;
		vector<bool> i_fp_det3;
		vector<int32_t> i_a_detG;
		vector<double> i_o_detG;
		vector<bool> i_fp_detG;

		size_t n = i_a_det2.size();
		if (accepted_det_3D.size()>frameNum){
			i_a_det3 = accepted_det_3D[frameNum];
			i_o_det3 = overlap_det_3D[frameNum];
			i_fp_det3 = fp_det_3D[frameNum];
		}else{
			i_a_det3 = vector<int32_t>(n,-1);
			i_o_det3 = vector<double>(n,0);
			i_fp_det3 = vector<bool>(n,false);
		}
		if (accepted_det_G.size()>frameNum){
			i_a_detG = accepted_det_G[frameNum];
			i_o_detG = overlap_det_G[frameNum];
			i_fp_detG = fp_det_G[frameNum];
		}else{
			i_a_detG = vector<int32_t>(n,-1);
			i_o_detG = vector<double>(n,0);
			i_fp_detG = vector<bool>(n,false);
		}
		for (int j=0; j<i_a_det3.size(); j++){
			if (i_a_det3[j] >= 0) {ass_gt3=i_a_det3[j];} else {ass_gt3=-1;};
			if (i_a_detG[j] >= 0) {ass_gtG=i_a_detG[j];} else {ass_gtG=-1;};
			if (i_a_det2[j] >= 0) {ass_gt2=i_a_det2[j];} else {ass_gt2=-1;};
			if (i_fp_det3[j]) {fp_det3=1;} else {fp_det3=0;};
			if (i_fp_detG[j]) {fp_detG=1;} else {fp_detG=0;};
			if (i_fp_det2[j]) {fp_det2=1;} else {fp_det2=0;};

			fdet << setfill(' ') << setw(5) << indices[frameNum] << ", " << setw(5) << j << ", " << setw(4+4) << fp_det3 
			<< ", " << setw(4) << fp_detG << ", " << setw(4) << fp_det2 << ", " << setw(5+11) << ass_gt3 << ", " <<
			setw(5) << ass_gtG << ", " << setw(5) << ass_gt2 << ", " << setprecision(2) << setw(5+4) << i_o_det3[j] << ", " << setw(5)<<
			i_o_detG[j] << ", " << setw(5) << i_o_det2[j] << endl;
			//fprintf(fdet, "%4s, %4s, %4s, %4s, %4s, %.2f, %.2f, %.2f\n",(indices[frameNum]),(j),
			//	(i_a_det3[j]),(i_a_detG[j]),(i_a_det2[j]),i_o_det3[j],i_o_detG[j],i_o_det2[j]);
		}
	}
	fdet.close();
}

void write_gt_file(string result_dir, vector<int32_t> indices, vvvi &detected_gt_3D, vvvi &detected_gt_G, vvvi &detected_gt_2D, string difficulty){
	ofstream fgt;
	fgt.open((result_dir + "/hack_gt_" + difficulty + "_evaluation.txt").c_str());
	fgt << "frame, index, tp_gt_3D, tpGr, tp2D, fn_gt_3D, fnGr, fn2D, associated_det, det_Gr, det_2D" << endl;
	int tp_gt3, tp_gtG, tp_gt2;
	int fn_gt3, fn_gtG, fn_gt2;
	int ass_gt3, ass_gtG, ass_gt2;
	size_t nframe = detected_gt_2D.size();
	assert(nframe==indices.size());
	for (int frameNum=0; frameNum<nframe;frameNum++){
		vector<int32_t> i_d_gt2 = detected_gt_2D[frameNum];
		size_t n = i_d_gt2.size();
		vector<int32_t> i_d_gt3;
		i_d_gt3 = detected_gt_3D.size()>frameNum ? detected_gt_3D[frameNum]:vector<int32_t>(n,-1);
		vector<int32_t> i_d_gtG;
		i_d_gtG = detected_gt_G.size()>frameNum ? detected_gt_G[frameNum]:vector<int32_t>(n,-1);
		for (int j=0; j<i_d_gt3.size(); j++){
			if (i_d_gt3[j] >= 0) {tp_gt3=1;ass_gt3=i_d_gt3[j];} else {tp_gt3=0;ass_gt3=-1;};
			if (i_d_gtG[j] >= 0) {tp_gtG=1;ass_gtG=i_d_gtG[j];} else {tp_gtG=0;ass_gtG=-1;};
			if (i_d_gt2[j] >= 0) {tp_gt2=1;ass_gt2=i_d_gt2[j];} else {tp_gt2=0;ass_gt2=-1;};
			if (i_d_gt3[j] == -10) {fn_gt3=1;} else {fn_gt3=0;};
			if (i_d_gtG[j] == -10) {fn_gtG=1;} else {fn_gtG=0;};
			if (i_d_gt2[j] == -10) {fn_gt2=1;} else {fn_gt2=0;};
			fgt << setfill(' ') << setw(5) << indices[frameNum] << ", "  << setw(5) << j << ", " << setw(4+4) << tp_gt3 
			<< ", " << setw(4) << tp_gtG << ", " << setw(4) << tp_gt2 << ", "  << setw(4+4) << fn_gt3 
			<< ", " << setw(4) << fn_gtG << ", " << setw(4) << fn_gt2 << ", " << setw(5+9) << 
			ass_gt3 << ", " << setw(5) << ass_gtG << ", " << setw(5) << ass_gt2 << endl;
			//fprintf(fgt, "%4s, %4s, %4s, %4s, %4s, %4s, %4s, %4s\n",to_string(indices[frameNum]),j,
			//	to_string(tp_gt3), to_string(tp_gtG), to_string(tp_gt2), to_string(i_d_gt3[j]), to_string(i_d_gtG[j]), to_string(i_d_gt2[j]));
		}
	}
	fgt.close();
}
