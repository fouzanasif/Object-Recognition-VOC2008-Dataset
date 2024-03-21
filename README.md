<h1>PASCAL's Visual Object Classes Challenge</h1>
<p>PASCAL's Visual Object Classes Challenge (VOC) is an annual competition in computer vision that aims to encourage the development of algorithms for object detection and recognition. The challenge involves a set of standard datasets and evaluation metrics that enable researchers to compare their methods against others in a fair and consistent way. The VOC2008 challenge focused on the detection and classification of objects in natural images.</p>

<h2>Brief Summary</h2>

<p>Object Recognition on Pascal's VOC008 competition dataset using Deep Learning Techniques is a project aimed at identifying various objects from the VOC2008 dataset. The project utilizes the power of Convolutional Neural Networks (CNNs) including VGG16, ResNet, DenseNet, MobileNet, and InceptionV3. These CNNs are pre-trained on the ImageNet dataset and employ transfer learning and fine-tuning techniques to accurately recognize 20 different objects.</p>
<p>The VOC2008 dataset contains images from the PASCAL Visual Object Classes (VOC) challenge. It consists of images with annotations for various object categories such as person, car, dog, cat, etc. The dataset is widely used for training and evaluating object detection and recognition models.</p>
<p>This project combines cutting-edge deep learning techniques with a rich dataset to achieve high accuracy in object recognition tasks. By leveraging pre-trained CNNs and fine-tuning them on the VOC2008 dataset, the model is capable of accurately identifying objects in images, enabling applications in fields like computer vision, autonomous driving, and augmented reality.</p>

<h2>DataSet Link</h2>
<a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2008/">Pascal's VOC2008 Dataset</a>

<h2>Problem</h2>
<p>The challenge aimed to address the problem of detecting and recognizing objects in natural images. Object detection is the process of identifying and localizing objects in an image, while object recognition involves recognizing the category of the object. The VOC2008 challenge included 20 object categories, including animals, vehicles, furniture, and people. The goal was to develop algorithms that could accurately detect and classify these objects in real-world images.</p>

<h2>Training and Testing Data</h2>
<p>The VOC2008 dataset consisted of 9963 images, divided into two subsets: the training set and the testing set. The training set contained 5011 images, while the testing set contained 4952 images. Each image in the dataset was annotated with object bounding boxes and object category labels. The annotations were provided by a team of human annotators, who labeled each object with its corresponding category and provided a bounding box around it. The training data was used to train object detection and classification algorithms, while the testing data was used to evaluate the performance of these algorithms.</p>

<h2>Evaluation Criteria</h2>
<p>For the object detection task, the evaluation metric was the mean average precision (mAP) over all 20 object categories. mAP measures the precision of the detection algorithm at various recall levels, where recall is the fraction of objects that are correctly detected. For the object classification task, the evaluation metric was the average precision (AP) over all 20 object categories, where precision is the fraction of correctly classified images.</p>
        
<h2>The Process</h2>
<p>The objective of this project was to apply transfer learning using up to five famous CNN architectures on a classification challenge and compare their results. The dataset used for this challenge was the VOC2008 dataset, which consists of images belonging to different object categories. The main goal was to classify each image into one of the predefined categories.</p>

<h3>Data Preparation</h3>
<p>To achieve this goal, we first prepared the data by dividing it into training, validation, and testing sets. We used the Keras ImageDataGenerator class to load and preprocess the images. The process is defined below in detail:</p>
<ol>
<li>For images info and labels present in VOC2008/ImageSets/Main, we created different folders for train, val, and test, containing subdirectories by the name of objects that were to be identified (e.g., aeroplane, bicycle, tvmonitor, sofa, etc.).</li>
<li>For each image’s name within object_train.txt, test, and val files, the images from VOC2008/JPEGImages were stored in the respective directories (e.g., train/object/image_name.jpg).</li>
<li>These images were then brought directly into our notebook using ImageDataGenerator’s flow_from_directory function.</li>
</ol>

<h3>Transfer Learning with CNN Architectures</h3>
<p>We then applied transfer learning using five famous CNN architectures: VGG16, ResNet50, DenseNet, MobileNet, and InceptionV3. We loaded the pre-trained models and added a few custom layers on top of them. We froze the weights of the pre-trained layers and trained only the added custom layers. The fully connected neural network that we used had 1024 hidden layer nodes (instead of 4096 in order to keep all the used CNN architectures’ NNs identical). Finally, the output layer contained 20 nodes for each object, indexed by certain class label identified by data_generator.class_indices.</p>
<p>After training each model, we evaluated its performance on the testing set using the mean average precision (mAP) evaluation metric. We calculated the mAP by computing the precision and recall for each class and taking their weighted average. Finally, we ranked the top 10 images for each architecture based on their predicted probabilities and used these rankings to compare the performance of each architecture.</p>

<p>However, given the standard accuracy and transfer capacity on Imagenet’s dataset, the performance of all these models on our dataset was quite adverse ranging from 2% accuracy of DenseNet (and 3% ResNet50) to 34% accuracy of VGG, and InceptionV3. This may be due to the fact that we used stored model weights which decreased the accuracy.</p>
<p>Due to lack of training resources (and failures) and ineffective coding practices, the training of each of these architectures may not fruitful, and the work to improve the working will continue onwards.</p>

<h2>Conclusion</h2>
<p>In conclusion, we applied transfer learning using up to five famous CNN architectures on a classification challenge using the VOC2008 dataset. We used the mean average precision (mAP) evaluation metric to compare the performance of each architecture. We ranked the top 10 images for each architecture to further analyze their performance.</p>

<p>Explore the project to delve into the fascinating world of deep learning and object recognition!</p>
