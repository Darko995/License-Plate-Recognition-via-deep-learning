# License-Plate-Recognition-via-deep-learning
License plate detection and recognition using YOLOv3 and CNN

License plate recognition is one of the most important components of modern intelligent transportation systems, and it is widely used.
This is mix of different discipline like image processing, pattern recognition, computer vision and other technologies.
In this project, I propose license plate recognition method and the flowchart of whole system.
It is divided in two parts, first is license plate detection on image and other is character segmentation and recognition.
The license plate was detected using YOLOv3 CNN.
After the license plate has been extracted from the image, it is time to recognize the characters in the image and convert them into text. CNN (Convolutional Neural Network) with four hidden convolutional layers was used to recognize the characters from the image.
Given the issues, data sets with as many different examples as possible were used to train these two deep neural networks. In real problems, there are many aggravating circumstances that can make it difficult to recognize license plates from an image. Some of these circumstances are: the weather conditions in which the photo was taken, the quality of the camera with which the photo was taken, the different size of letters and numbers on the tables, etc. In order to overcome these problems, a set of data from different sources was used ( photographed with different devices, in different weather conditions, images of different quality ).
The accuracy was significantly improved with compared with tradicional recognition methods.

The figure shows the basic architecture of the YOLOv3 network:

![image](https://user-images.githubusercontent.com/53577768/157534703-51760967-19be-4850-bfe8-c054eaeadde0.png)

The DarkNet53 architecture was used to separate the features from the image at the entrance. It consists of 53 convolutional layers including the FC layer.
In figure below you can see the architecture of this convolutional network:

![image](https://user-images.githubusercontent.com/53577768/157535044-2f062ea0-2205-480a-81cc-9c0c42a43142.png)

The network consists of successive convolutional layers in which filters of dimensions 3x3 and 1x1 appear. Finally, the FC layer and Softmax are shown, as this network was originally used to classify images (1000 different classes). However, object detection is needed, not just classification, so a multi-stage detector was used. The picture at the entrance is 416x416. Since objects can be of different sizes, there are types of vectors with features that are sent to the detector: 13x13 for large objects, 26x26 for medium and 52x52 for smaller objects.

In a concrete example, the network distinguishes 80 different classes:

![image](https://user-images.githubusercontent.com/53577768/157535828-7f6e7c99-c6e1-43c1-9278-5f50e880dc85.png)

Early Stopping Method It works in such a way that training stops when the network stops advancing, ie learning and generalizing. Some images from the training set are used as a validation set, ie they are used during training as images that the network has not seen. The error on the training set is constantly decreasing, because the network is learning those examples better, but the goal is not to learn all the examples by heart, but to be able to generalize on examples that it did not see during the training. As long as the error in the validation set decreases, the network is going in the right direction and it is necessary to continue training. When the error at the validation set starts to grow, the training stops

The following libraries were used in the Jupyter notebook development environment: numpy, cv2 and matplotlib.

Two methods are defined: detection(images) and cropped(). The detection (image) method is used to detect the license plate in the image from the input and draw a raightangle around the table. The second cropped() method extracts the part of the image that is bounded by a raightangle.

The figure shows an example of an image that was not used for network training:

![image](https://user-images.githubusercontent.com/53577768/157539809-922e7e02-71ab-47e3-9a6f-c2a6481cd811.png)

The picture shows a cropped part of the picture with a table:

![image](https://user-images.githubusercontent.com/53577768/157539884-757dd590-30a0-4fc9-9e5e-f54b0e324a39.png)


There are characters in the picture, and below it is necessary to recognize these characters and turn them into text. Another convolutional neural network was used for this problem.

The problem of recognizing characters on the license plate does not require segmentation of lines within the text. The built-in EasyOCR module was used for character segmentation in this project. This package within the python programming language is used to convert images to text. The role of this module is to recognize the individual character.

Images (as many as there are characters on the table) are cut based on the boundaries around each character.

After drawing the raightangle around the text, the image was cut along the borders of the raightangle. This eliminates the part of the image that does not contain characters and is not needed. In figure below are shown after cutting off the part of the image that does not contain characters:

![image](https://user-images.githubusercontent.com/53577768/157540028-8a743d98-7f19-4123-9a0a-590c65b6fc58.png)

After the part of the image that is not useful for character detection has been eliminated, it is necessary to detect the boundaries of each individual character and cut out each character in order to be able to recognize or classify numbers and letters. EasyOCR was also used for this part

Based on the list of raightangle coordinates obtained at the output of this algorithm, individual characters were singled out. Raightangles that do not represent individual characters were eliminated using dimensions, after it was experimentally determined which dimensions are frames that detect characters. In figure below the reightangle in which the individual characters are shown are shown:

![image](https://user-images.githubusercontent.com/53577768/157540127-2caa40d4-d3d5-4c32-8ec2-4009c3735f35.png)


After detecting all the individual characters, all the raightangles that contain them were cut out and each character was placed in one picture. On the example of the license plate in the picture below, it is necessary to single out 7 pictures. Pictures of individual characters are shown in the picture next to:

![image](https://user-images.githubusercontent.com/53577768/157540189-20724a7f-c1b8-4d7f-9d07-95b90631589f.png)
![image](https://user-images.githubusercontent.com/53577768/157540219-c8b4a081-39b8-4bd9-88f3-dce27bec2437.png)
![image](https://user-images.githubusercontent.com/53577768/157540236-c29721a5-ddce-4a34-86a5-991df0150a6c.png)
![image](https://user-images.githubusercontent.com/53577768/157540252-8d6aaa34-f37c-4e7d-8481-19d18b5db9e2.png)
![image](https://user-images.githubusercontent.com/53577768/157540256-b31cbfa9-38f1-4754-acf0-eb0b9d0d282b.png)
![image](https://user-images.githubusercontent.com/53577768/157540282-f21fc57a-023a-4035-b930-f41ba24bf865.png)
![image](https://user-images.githubusercontent.com/53577768/157540296-60500bc1-91a7-4d01-a674-e4598cdd7c87.png)


Once the individual characters on the license plate have been selected, they need to be classified. Each individual image is brought to the input of a convolutional network whose task is to show a character in the form of text at its output.

['B', 'G', '4', '5', '5', 'O', 'Y']

Finally, the quality of the model is assessed at the test set. This collection contains about 10% of the total number of images. There are several metrics used to assess the quality of a classification. In figure below the report is shown after the classification:

![image](https://user-images.githubusercontent.com/53577768/157539509-6b957e76-ee59-4bb6-9c8c-48eb18b4dd4d.png)







