# Detection-of-Rare-Diseases-by-observing-Facial-Features
This project has been done under Prof. Mukesh Kumar Rohil at Birla Institute of Technology and Science, Pilani

Proposed Approach
Our goal is to make an end-to-end model which takes an image of the face as input provides result as probabilities against various diseases for the given face.
Architecture for syndrome classiﬁcation: To learn the baseline facial recognition Inception-Resnet-v2 is used as backbone architecture. The whole architecture is illustrated in Figure 3. To train the model for facial recognition task, the model is trained on CASIA web-face dataset [10] which contains 453,453 images over 10,575 identities. After the preprocessing of the image, the image is segmented into diﬀerent facial regions and model is then ﬁne tuned for each facial feature. The syndromic images used to ﬁne-tune the model has been collected from diﬀerent sources.
A softmax classiﬁer is used on each facial region to make separate prediction for each region and then results are averaged out to make a robust multi-class prediction. The evaluation is done based on top-n accuracy. Top-N accuracy basically measures how often the predicted class falls in top N values of softmax distribution. The results are also evaluated based on confusion matrix to visulaize in what ways the model is making an inaccuracy in classifying the image.

The training data consists of almost 20-25 images of each disease and 150 images of face with no genetic condition. I customised the face recognition algorithm to output the boundary of the face identiﬁed in the picture and there are options for both GPU enabled and non-enabled devices. The original architecture has an accuracy of 99.38% on the Labeled Faces in the Wild benchmark.

![alt text](https://github.com/quasayush/Detection-of-Rare-Diseases-by-observing-Facial-Features/blob/main/results/figure4.JPG)

![alt text](https://github.com/quasayush/Detection-of-Rare-Diseases-by-observing-Facial-Features/blob/main/results/figure5.JPG)
