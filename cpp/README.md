# two_stage_demo (C++)

`demos/two_stage_demo.py` 的 C++ 移植:两级 RKNN 推理(受电弓区域 → 部件/异物)。

依赖:
- `librknnrt`(板子已装,`/usr/lib/librknnrt.so`),头文件已 vendor 在 `3rdparty/rknn/include/`
- OpenCV:默认取 `-DOpenCV_DIR` 指向的静态库(见 CMakeLists),装了系统 OpenCV 就用 `-DOpenCV_DIR=/usr/lib/aarch64-linux-gnu/cmake/opencv4` 之类覆盖
- 配置文件是扁平 `key: value` YAML,内置简易解析,无 yaml 依赖

构建(在板子上):

```bash
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

运行(在仓库根目录,默认参数与 Python 版一致):

```bash
./cpp/build/two_stage_demo --image 3C_Test.jpg
# 可选: --stage1-model / --stage2-model / --stage2-config / --output / --conf / --overlap
```
