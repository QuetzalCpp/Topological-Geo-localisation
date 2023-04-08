# Topological-Geo-localisation

Geo-localisation from a single aerial image for Uncrewed Aerial Vehicles (UAVs) is an alternative to other vision-based methods, such as visual Simultaneous Localisation and Mapping (SLAM), seeking robustness under GPS failure. Due to the success of deep learning and the fact that UAVs can carry a low-cost camera, we can train a Convolutional Neural Network (CNN) to predict position from a single aerial image. However, conventional CNN-based methods adapted to this problem require off-board training that involves high computational processing time and where the model can not be used in the same flight mission. In this work, we explore the use of continual learning via latent replay to achieve online training with a CNN model that learns during the flight mission GPS coordinates associated with single aerial images. Thus, the learning process repeats the old data with the new ones using fewer images. Furthermore, inspired by the sub-mapping concept in visual SLAM, we propose a multi-model approach to assess the advantages of using compact models learned continuously with promising results. On average, our method achieved a processing speed of 150 fps with an accuracy of 0.71 to 0.85, demonstrating the effectiveness of our methodology for geo-localisation applications.

## Overview of our approach

![alt text](images/figure1.png)

The continual learning approach consists of 3 steps: 1) Mini-batches generation during the UAV's flight; 2) Continual training using previous mini-batches while new ones are created; 3) Model and multi-model evaluation to classify the current aerial image and get a flight coordinate in a sub-mapping fashion. We use a keyframe search based on a colour histogram to identify the corresponding model in a multi-model approach.

## Video
A video of this approach can be watched at [Youtube](https://youtu.be/xfsU_cCLpFw)

## Recommended system
- Ubuntu 20.04
- ROS Noetic
- Python 3.8.10
- PyTorch 1.9.0
- TorchVision 0.10.0
- Pytorchcv 0.0.67
- Cuda 11.2
- Cudnn 8.0

### Additional Resources
- [Dataset](https://mnemosyne.inaoep.mx/index.php/s/6w3zgta5iXn2ioi)

## Reference
If you use any of data or code, please cite the following reference:

Cabrera-Ponce, A.A., Marin-Ortiz, Manuel & Martinez-Carranza, J. (2023). Continual Learning for Topological Geo-localisation. Journal of Intelligent Fuzzy Systems.


## Related References

- Cabrera-Ponce, A. A., & Martinez-Carranza, J. (2022). Convolutional neural networks for geo-localisation with a single aerial image. Journal of Real-Time Image Processing, 19(3), 565-575. https://doi.org/10.1007/s11554-022-01207-1

```
@article{cabrera2022convolutional,
  title={Convolutional Neural Networks for Geo-Localisation with a Single Aerial Image},
  author={Cabrera-Ponce, Aldrich A and Martinez-Carranza, Jose},
  journal={Journal of Real-Time Image Processing},
  volume={19},
  number={3},
  pages={565--575},
  year={2022},
  publisher={Springer}
}
```

- A. A. Cabrera-Ponce and J. Martinez-Carranza, "erial geo-localisation for MAVs using PoseNet," 2019 Workshop on Research, Education and Development of Unmanned Aerial Systems (RED UAS), Cranfield, United Kingdom, 2019, pp. 192-198, doi: 10.1109/REDUAS47371.2019.8999713

```
@inproceedings{cabrera2019aerial,
  title={Aerial geo-localisation for MAVs using PoseNet},
  author={Cabrera-Ponce, Aldrich A and Martinez-Carranza, J},
  booktitle={2019 Workshop on Research, Education and Development of Unmanned Aerial Systems (RED UAS)},
  pages={192--198},
  year={2019},
  organization={IEEE}
}
```

 ## Acknowledgements
We are thankful for the processing time granted by the National Laboratory of Supercomputing (LNS) under the project 201902063C. The first author is thankful for his scholarship funded by Consejo Nacional de Ciencia y Tecnologia (CONACYT) under grant 802791.
