# Enhancing CT Segmentation Security against Adversarial Attack: Most Activated Filter Approach

**Description**: A novel framework for the detection of adversarial samples in convolutional layer-based CT segmentation models.

## Abstract
In the context of medical imaging, the integration of deep learning with computed tomography (CT) scans has led to improved diagnostics. However, this advancement has simultaneously opened avenues for adversarial attacks, posing risks to the accuracy and integrity of model predictions. Such deceptions, though imperceptible to the human eye, may have profound implications in clinical settings. In this paper, we present a novel framework for the detection of adversarial samples in convolutional layer-based CT segmentation models, with a keen focus on the fast gradient sign method, basic iterative method, and stabilized medical image attack. Our in-depth analysis unveils the superior efficiency of initial layer features over the final layer ones in adversarial detection. We also showcase the role of varying activation levels in differentiating genuine from adversarial inputs. Furthermore, our model robustly differentiates genuine Gaussian noise from adversarial perturbations, thereby minimizing potential false positives. Alongside these technical advancements, we offer visualization analysis designed for physicians, ensuring interpretability and instilling confidence in deep learning-assisted diagnostics. Collectively, our contributions usher in a comprehensive solution to adversarial threats in CT scan analyses, emphasizing not only detection accuracy but also clarity and understanding for medical practitioners.


## Methods

<p align='center'>
<img src='./figures/proposed_method.png" width="400">
</p>

<p align="center">
<img src="./figures/method_diagram.png" width="300">
</p>

1. **Producing adversarial samples using the target model and a clean dataset:** In the beginning, we utilize the provided target model and clean dataset to create adversarial samples through a recognized attack technique. After generating these samples, they are partitioned into training and validation datasets based on subjects.

2. **item Feature extraction from target model filters:** In the next step, we process the training and validation datasets through the target model, extracting features from each filter in its first layer. Essentially, these features will serve as the input to our classifier, capturing unique characteristics of each image. The features are important because they contain patterns that help to differentiate between legitimate and adversarial inputs.

3. **Classifier training for each filter:** Once we have the feature sets, we proceed to train individual classifiers for each filter using the extracted features from the training set. This is performed to learn the mappings from features to labels (adversarial or not) for each filter. The classifiers could be any machine learning models suitable for binary classification, such as Random Forests, Decision Trees, or Support Vector Machine.

4. **Identifying the most discriminative filter:** After training the classifiers, the next step is to evaluate their performance on the validation set. Our primary aim is to identify the filter that is most effective at distinguishing between genuine and adversarial inputs. We are particularly interested in filters that produce \textbf{a high mean and low variance in their classification scores across the validation set}, as these filters are the most reliable and stable for detection.

5. **Building the final adversarial attack detector:** Armed with the most discriminative filter identified in the previous step, we proceed to build the final adversarial attack detector. We train this detector using both the training and validation datasets. This comprehensive training allows the detector to generalize well to unseen data, effectively identifying adversarial attacks while minimizing false positives and negatives.


## Settings
* TODO: PATH SETTINGS
* 
## Usage
* For training attack detector using random forest against FGSM with eps 0.01, 
```
python adv-classifier-optimization-sklearn.py --attack_name 'fgsm' --gpu 0 --eps 0.01 --classifier 0
```

## Dataset
* This work utilizes "Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge" (BTCV) which is publicly available at https://www.synapse.org/#!Synapse:syn3193805/wiki/217789

## Results
We present a side-by-side comparison of PPV and sensitivity among various methods in the table below. As the nature of PPV and sensitivity, higher values for these metrics are preferable. A low PPV suggests that the method is prone to incorrectly labeling genuine samples as adversarial, while low sensitivity implies the method may fail to identify adversarial samples. As the table reveals, our approach outperforms all other methods in both PPV and sensitivity across every configuration. Notably, our method achieves a flawless PPV and sensitivity scores in all scenarios. 

<p align="center">
<img src="./figures/precision_recall.PNG" width="600">
</p>

## Citation
If you find this code useful for your research, please cite the paper:
```
@article{,
  title={},
  author={},
  journal={},
  volume={},
  pages={},
  year={},
  publisher={}
}```
