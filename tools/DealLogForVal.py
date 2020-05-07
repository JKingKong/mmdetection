import os
import json
relative_path = '../'
root_path = 'log_faster_rcnn_fpn_RoIAlign' +  "/"
mAP_file = 'mAP.log.json'
loss_file = 'loss.log.json'


mAP_save_path = relative_path + root_path + mAP_file
loss_save_path = relative_path + root_path + loss_file

mAP_fp = open(mAP_save_path, 'a+', encoding='utf_8')
loss_fp = open(loss_save_path, 'a+', encoding='utf_8')

# 必要的头部
must_head = {"env_info": "sys.platform: linux\nPython: 3.6.9 (default, Nov  7 2019, 10:44:02) [GCC 8.3.0]\nCUDA available: True\nCUDA_HOME: /usr/local/cuda\nNVCC: Cuda compilation tools, release 10.1, V10.1.243\nGPU 0: Tesla P4\nGCC: gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0\nPyTorch: 1.4.0\nPyTorch compiling details: PyTorch built with:\n  - GCC 7.3\n  - Intel(R) Math Kernel Library Version 2019.0.4 Product Build 20190411 for Intel(R) 64 architecture applications\n  - Intel(R) MKL-DNN v0.21.1 (Git Hash 7d2fd500bc78936d1d648ca713b901012f470dbc)\n  - OpenMP 201511 (a.k.a. OpenMP 4.5)\n  - NNPACK is enabled\n  - CUDA Runtime 10.1\n  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_37,code=compute_37\n  - CuDNN 7.6.3\n  - Magma 2.5.1\n  - Build settings: BLAS=MKL, BUILD_NAMEDTENSOR=OFF, BUILD_TYPE=Release, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -fopenmp -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -O2 -fPIC -Wno-narrowing -Wall -Wextra -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Wno-stringop-overflow, DISABLE_NUMA=1, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, USE_CUDA=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, USE_STATIC_DISPATCH=OFF, \n\nTorchVision: 0.5.0\nOpenCV: 4.1.2\nMMCV: 0.4.4\nMMDetection: 1.1.0+5f3437a\nMMDetection Compiler: GCC 7.5\nMMDetection CUDA Compiler: 10.1", "seed":None}
line = json.dumps(must_head, ensure_ascii=False)
mAP_fp.write(line + '\n')
line = json.dumps(must_head, ensure_ascii=False)
loss_fp.write(line + '\n')


filename_list = os.listdir(relative_path + root_path)
filename_list.sort()
for i in range(0,len(filename_list)):
    #设置旧文件名（就是路径+文件名）
    json_log_path = relative_path + root_path + filename_list[i]
    val_json_list = []
    train_json_list = []
    if filename_list[i].split(".")[0] == 'loss' or filename_list[i].split(".")[0] == 'mAP':
        # 不处理loss.log.json 和 mAP.log.json文件
        continue
    with open(json_log_path, 'r', encoding='utf_8') as fp:
        multiJson = fp.readlines()
        for oneLineJson in multiJson:
            data = json.loads(oneLineJson)
            if 'mode' in data.keys():
                if data['mode'] == 'val':
                    line = json.dumps(data, ensure_ascii=False)
                    mAP_fp.write(line + '\n')
                elif data['mode'] == 'train':
                    line = json.dumps(data, ensure_ascii=False)
                    loss_fp.write(line + '\n')



