# YOLOv5 (C++/Python) Inference

**This repository contains the code for yolo inference part of project SME**.

## Dependencies
```
pip install -r requirements.txt #python version
for C++, we used opencv 4.5.4, Ubuntu 20.04
```

## Execution
### Python
```Python
python yolov5.py
```
### CMake C++ Linux
note that the image size is 640x480, if you want to change it, you need to change the code.
```C++ Linux
mkdir build
cd build
cmake ..
cmake --build .
./main
```
