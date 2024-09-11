# Temporal Event Stereo via Joint Learning with Stereoscopic Flow (TESNet) (ECCV2024)
<p align="center">
 <img src="https://drive.google.com/file/d/1kBvzaLwlDeJE0CwQvHdqFii7YXL4KB7p/view?usp=sharing" width="80%"/>
</p>
<p align="center">
 <img src="resource/teaser.png" width="80%"/>
</p>


Official code for "Temporal Event Stereo via Joint Learning with Stereoscopic Flow" (ECCV2024)
([Paper](https://arxiv.org/abs/2407.10831))



```bibtex
@Article{tes24eccv,
  author  = {Hoonhee Cho* and Jae-Young Kang* and Kuk-Jin Yoon},
  title   = {Temporal Event Stereo via Joint Learning with Stereoscopic Flow},
  journal = {European Conference on Computer Vision. (ECCV)},
  year    = {2024},
}
```



## Abstract
Event cameras are dynamic vision sensors inspired by the biological retina, characterized by their high dynamic range, high temporal resolution, and low power consumption. These features make them capable of perceiving 3D environments even in extreme conditions. Event data is continuous across the time dimension, which allows a detailed description of each pixel's movements. To fully utilize the temporally dense and continuous nature of event cameras, we propose a novel temporal event stereo, a framework that continuously uses information from previous time steps. This is accomplished through the simultaneous training of an event stereo matching network alongside stereoscopic flow, a new concept that captures all pixel movements from stereo cameras. Since obtaining ground truth for optical flow during training is challenging, we propose a method that uses only disparity maps to train the stereoscopic flow. The performance of event-based stereo matching is enhanced by temporally aggregating information using the flows. We have achieved state-of-the-art performance on the MVSEC and the DSEC datasets. The method is computationally efficient, as it stacks previous information in a cascading manner. 


## Datasets
Please refer to the pre-processing directory ([pre-process](https://github.com/mickeykang16/TemporalEventStereo/tree/main/pre-processing)) for the dataset's format and details.


## Training
Comming Soon
