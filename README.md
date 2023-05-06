# 🚦DriveScenify: Boosting Driving Scene Understanding with Advanced Vision-Language Models

## Introduction 📚
The increasing complexity of traffic situations, coupled with the rapid growth of urban populations, necessitates the development of innovative solutions that can mitigate congestion, reduce traffic-related accidents, and facilitate smoother transportation systems. Recognizing the significant impact of ChatGPT and computer vision technologies on various domains, it is timely to investigate how these advancements can be harnessed to address the critical challenges in urban transportation safety and efficiency. 

With this motivation, we introduce DriveScenify, an approach that aims to boost driving scene understanding by leveraging advanced vision-language models. Our research focuses on developing a tailored version of MiniGPT-4, called DSify, which is specifically designed to process and generate contextually relevant responses based on driving scene videos. DriveScenify's integration of advanced vision-language models into the realm of transportation aims to unlock new possibilities for improving urban mobility, reducing traffic-related accidents, and enhancing the overall driving experience.

Furthermore, our unique combination among various encoders enables DSify to provide accurate and context-aware insights, which can be applied to various transportation applications, especially for traffic management, and road safety analysis.

## Features 🌟
- Spatial-temporal Safe Driving Scene Comprehension: DriveScenify is meticulously developed to accurately interpret diverse driving scenarios, encompassing traffic patterns, vehicle classifications, road conditions and temporal information, with a particular emphasis on promoting driving safety.
- Contextual Response Formulation: The model is capable of generating context-sensitive responses and recommendations derived from the driving scene, offering valuable guidance to users.
- While our central focus lies in training DSify using driving scenario videos, the model also exhibits a degree of competence in understanding and processing general video content. This versatility enhances its potential applications across a broader range of domains while maintaining its primary objective of improving driving safety and scene understanding.

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

## Contributors:
Xiaowei Gao* (SpacetimeLab, University College London, UK)

Pengxiang Li* (IIAU-Lab, Dalian University of Technology, China)

Ming Li (National Heart & Lung Institute, Imperial College London, UK)

Xinke Jiang (Key Lab of High Confidence Software Technologies,Peking Universtiy, China)

(*Equal Contribution)

## Contributing 🤝
At present, DriveScenify is in its initial stages, and in many cases, **the performance may not be as ideal as expected**. Data generation is still ongoing, and we are continuously working to improve the model. We highly appreciate and welcome contributions from the community to help enhance DriveScenify's capabilities and performance.

## License 📄
This repository is under [BSD 3-Clause License](LICENSE.md).
Many codes are based on [Lavis](https://github.com/salesforce/LAVIS) with 
BSD 3-Clause License [here](LICENSE_Lavis.md).

## Acknowledgments 🤝
We would like to thank the developers of [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [LLaVA](https://github.com/haotian-liu/LLaVA), [InternVideo](https://github.com/OpenGVLab/InternVideo), [Ask-Anything](https://github.com/OpenGVLab/Ask-Anything), [Image2Paragraph](https://github.com/showlab/Image2Paragraph) and [Vicuna](https://github.com/lm-sys/FastChat) for their incredible work and providing the foundation for DriveScenify.