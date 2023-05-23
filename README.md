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
```C++ Linux
mkdir build
cd build
cmake ..
cmake --build .
./main
```
### CMake C++ Windows
```C++ Windows
rmdir /s /q build
cmake -S . -B build
cmake --build build --config Release
.\build\Release\main.exe
```
