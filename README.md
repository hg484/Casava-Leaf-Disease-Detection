# Casava Disease detection using leaf images
As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions however  viral diseases have became a major sources of poor crop yield.

Goal of the project is to identify the type of disease present on a Cassava Leaf by blending predictions provided by multiple image models. Techniques like Test Time Augmentation and Noise resistant loss function were used to deal with the noisy dataset.

### Example of input data
<p align="center"> <img src="./Diagrams and Images/image data example .png"> </p>

### Problem of Noisy data
The dataset consist of over 20000 images collected by farmers taking photos of the plants from their gardens which were labelled by experts from a crop research institute.However no lab test were performed to guarentee the accuracy of these labels which led to some discrepencies being present in the given dataset.

Similar looking images had different labels and in an extreme case, **Two identical images had different labels.**
<p align="center"> <img src="./Diagrams and Images/duplicate image different label.png"> </p>

<h4 align="center">Same Image with different label attached to it ( Label 0 and Label 4 )</h4>

### Stratergies used to mitigate effect of noisy data
<ol>
<li>StratifiedKFold data split </li>
This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.
<p align="center"><img src="./Diagrams%20and%20Images/StrarifiedKFold.png"></p>
<li>Bitempered Logistic Loss</li>
 <a href="https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html">Bitempered Logistic Loss</a> is a noise resistant loss function created by the Google Research which can deal with both small and large outliers that may exists in a dataset. <a href="https://paperswithcode.com/method/label-smoothing">Label Smoothing</a> was used along with the loss funtion, it is a regularization technique which improves the ability of your model to generalize and improve the overall performance of the model. 
<br><br>
<li>Test Time Augmentations (TTA)</li>
Test Time augmentations is a technique which boost a model'performance but applying augmentation during inference. Inference is performed on multiple altered versions of the same image.
<p align="center"><img src="./Diagrams and Images/test_time_augmentation_concept.png"></p>
  <p align="center">Image borrowed from <a href="stepup.ai">stepup.ai</a></p>

</ol>

### Training Strategy 
<p align="center"> <img src="./Diagrams and Images/model_training.png"> </p>
The data was divided into 5 stratified splits, and a specific image model is created for each fold. The final prediction generated will be the mean of all predictions prodcuced by each fold model.ResNet101, EfficientNetB0 and EfficientNetB4 were used to generate predictions

### Inference Strategy
#### Single Model Inference
<p align="center"> <img src="./Diagrams and Images/prediction_generation.png"> </p>

#### Prediction Blending
<p align="center"> <img src="./Diagrams and Images/final_prediction_generation.png"> </p>

### Results 
##### Metric used and how it's calculated:
The Metrics used to judge the performance are Accuracy and F1 score.Accuracy is the ratio of the number of labels predicted for the data that exactly match the corresponding set of labels in actual labels of the data. F1 score measures a test's accuracy, it is calculated using precision and recall of the test.


## Add confusion matrix image

<p align="center"> <img src="./Diagrams and Images/Model_performace classification report.png"> </p>


<p align="center"> <img src="./Diagrams and Images/Model_performace confusion matrix.png"> </p>
