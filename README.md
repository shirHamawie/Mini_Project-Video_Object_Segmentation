# Video Object Segmentation

This project is an implementation of a semi-supervised Video Object Segmentation method, building upon the ScStream codebase. 
The goal of this project is to enable the segmentation of a specified object in a video sequence with minimal manual annotation, making it well-suited for scenarios where full supervision is impractical or costly.

## Overview
Video Object Segmentation (VOS) is the task of separating a specific object or objects from the background in a video sequence. In this project, we focus on semi-supervised VOS, which means that we have the ground-truth mask of the object in the first frame of the video sequence. From that initial frame onward, the objective is to perform unsupervised segmentation for the object of interest.

## Key Features
Semi-Supervised Start: The algorithm starts with a provided ground-truth mask in the first frame, which serves as the initial guidance for segmentation.

Unsupervised Segmentation: After the initial frame, the method continues to segment the object without any additional supervision, adapting to the changing appearance and position of the object throughout the video.

Scalability: The approach is designed to be efficient and scalable, making it suitable for real-time or batch processing of video data.

# Watch the video!
https://github.com/shirHamawie/Mini_Project-Video_Object_Segmentation/assets/93772012/eec82b5f-f2fb-4086-b5f5-fcdba5305e9f


## License
MIT License

Copyright (c) [2023] [Shir Hamawie]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
