# 🚦DriveScenify: Boosting Driving Scene Understanding with Advanced Vision-Language Models

## Introduction 📚
DSify is a tailored version of MiniGPT-4 that focuses on understanding and generating responses based on driving scene videos. It aligns a frozen visual encoder from InternVideo with a frozen LLM, Vicuna, using the `PerceiverResampler` from `OpenFlamingo`, specifically for driving scenarios.

## Features 🌟
- Driving Scene Understanding: DriveScenify is designed to accurately comprehend various driving situations, including traffic patterns, vehicle types, and road conditions.
- Contextual Response Generation: The model can generate context-aware responses and suggestions based on the driving scene, providing valuable insights to users.
- Although our primary focus is on training with driving scenario videos, DSify also possess a certain level of understanding for general videos.

## Example 💬
![demo](https://user-images.githubusercontent.com/46072190/236392674-928bb5b4-2308-4061-a20c-b380c63fedd4.gif#pic_center)

## Usage 💻
DriveScenify was initially designed to comprehend corner cases and potentially hazardous situations within driving scenes. Our aim was to leverage the capabilities of Large Language Models (LLMs) to enhance the reasoning process for video understanding, providing a more comprehensive analysis of complex and challenging driving scenarios.

If you want to try the demo of this repo, you only need to refer to the installation process of [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), prepare the environment and Vicuna weights.

Then change the ckpt path in `eval_configs/minigpt4_eval.yaml`. You can download our weight here. [Checkpoint Aligned with Vicuna 13B](https://drive.google.com/file/d/1zFUOvdMo-OTkekz7pt81W_e-zy3X3I54/view?usp=sharing).

### Launching Demo Locally

Try out our demo [demo_video.py](demo_video.py) on your local machine by running

```
python demo_video.py --cfg-path eval_configs/minigpt4_eval.yaml
```

In fact, the demo supports both image and video inputs, so feel free to give it a try, even though the file is named "demo_video". Have fun exploring! 😄🎉📷🎥

## Upcoming Tasks 🤖
- [ ] Strong video foundation model.
- [ ] Training with dialogue datasets.
- [ ] Expanding data generation capabilities.
- [ ] ...

## Contributing 🤝
At present, DriveScenify is in its initial stages, and in many cases, **the performance may not be as ideal as expected**. Data generation is still ongoing, and we are continuously working to improve the model. We highly appreciate and welcome contributions from the community to help enhance DriveScenify's capabilities and performance.

## License 📄
This repository is under [BSD 3-Clause License](LICENSE.md).
Many codes are based on [Lavis](https://github.com/salesforce/LAVIS) with 
BSD 3-Clause License [here](LICENSE_Lavis.md).

## Acknowledgments 🤝
We would like to thank the developers of [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [LLaVA](https://github.com/haotian-liu/LLaVA), [InternVideo](https://github.com/OpenGVLab/InternVideo), [Ask-Anything](https://github.com/OpenGVLab/Ask-Anything), [Image2Paragraph](https://github.com/showlab/Image2Paragraph)and [Vicuna](https://github.com/lm-sys/FastChat) for their incredible work and providing the foundation for DriveScenify.