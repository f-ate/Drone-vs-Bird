**DRONE VS BIRD DETECTION MODEL**

**AIM AND OBJECTIVES**

**Aim**

To create a detection model which can differentiate between a Drone and
a Bird with good accuracy whether they are flying at a distance or are
flying close by and then show on the viewfinder whether the object is
Bird or a Drone.

**Objectives**

- The main objective of the project is to create a program which can be
  run on Jetson nano and start detecting using the camera module
  connected to the device.

- Using appropriate data sets for recognizing and interpreting data
  using machine learning.

- To show on the optical viewfinder of the camera module whether a given
  object is a Drone or a Bird.

**ABSTRACT**

- An object is classified based on whether it is a Drone or a Bird and
  then shown on the viewfinder of the camera.

- I have completed this project on jetson nano which is a very small
  computational device.

- A lot of research is being conducted in the field of Computer Vision
  and Machine Learning (ML), where machines are trained to identify
  various objects from one another. Machine Learning provides various
  techniques through which various objects can be detected.

- I have used here CVAT which stands for Computer Vision Annotation Tool
  which is a open source annotation tool made by intel for annotating
  images.

- Small drones are a rising threat due to their possible misuse for
  illegal activities such as smuggling of drugs as well as for terrorism
  attacks using explosives or chemical weapons. Several surveillance and
  detection technologies are under investigation at the moment, with
  different trade-offs in complexity, range, and capabilities..

- Drones and Birds when they are detected by the viewfinder of camera
  their distinction is minute but when used with proper dataset the
  detection model should be able to differentiate with good accuracy
  which results in better security of various places which require it.

**INTRODUCTION**

- This project is based on Drones vs Bird detection model. We are going
  to implement this project with Machine Learning and this project will
  be run on jetson nano.

- This project can also be used to gather information about the location
  of Drones in particular area.

- Objects can be classified into Drone or Bird category based on the
  image annotation we give in roboflow.

- Drone detection becomes difficult sometimes because of various sizes
  of Birds and Drones, also their distance from camera module as well as
  the time of the day which can make them harder for model to detect.
  However, before adding images in CVAT i have already augmented them
  using a software known as XnConvert which is free and has allowed me
  to crop images and change the zoom and contrast of certain images to
  match the time of day, lighting for better recognition by the model.

- Neural networks and machine learning have been used for these tasks
  and have obtained good results.

- Machine learning algorithms have proven to be very useful in pattern
  recognition and classification, and hence can be used for Drone vs
  Bird detection as well.

**LITERATURE REVIEW**

- Adopting effective techniques to automatically detect and identify
  small drones is a very compelling need for a number of different
  stakeholders in both the public and private sectors.Algorithms should
  raise an alarm and provide a position estimate only when a drone is
  present, while not issuing alarms on birds, nor being confused by the
  rest of the scene.

- The use of drones, whose origin is in the military domain, has been
  extended to several application fields including traffic and weather
  monitoring, precision agriculture, and many more. With the COVID-19
  pandemic, there has been a radical increase in the use of drones for
  autonomous delivery of essential grocery and medical supplies, but
  also to enforce social distancing. Nowadays small quadcopters can be
  easily purchased on the Internet at low prices, which brings
  unprecedented opportunities but also poses a number of threats in
  terms of safety, privacy and security.

- Already from the early days of powered flight, birds have been a
  concern to aviation safety. Since 1912, bird strikes have caused 47
  fatal accidents involving commercial air transport. Since aircraft and
  birds share the same airspace at lower altitudes, the hazard of bird
  strikes always exists. The cost of bird strikes to the aviation
  industry is estimated to be more than one billion euros annually.
  Similarly, unmanned aerial vehicles (UAVs) bring an important set of
  challenges to the aviation industry. Despite the restrictions of
  flying drones near airports, and reserved airspaces to ensure that
  they do not infringe each others’ area, there are still several
  challenges involved.

- Small drones are a rising threat due to their possible misuse for
  illegal activities such as smuggling of drugs as well as for terrorism
  attacks using explosives or chemical weapons. Several surveillance and
  detection technologies are under investigation at the moment, with
  different trade-offs in complexity, range, and capabilities.

- Recent incidents like the lethal drone attack on Saudi Arabia's
  largest petroleum company and arms dropping by UAVs in Punjab from
  across the India-Pakistan border has only alerted the agencies to come
  up with a plan to counter the drones.

- A Pakistani drone with a payload attached to it was shot down in Jammu
  and Kashmir’s Kathua district shortly after it crossed into the Indian
  side from across the International Border on Sunday morning, police
  said, adding that magnetic bombs and grenades were recovered from the
  payload.

- “A police search party picked up the movement of the drone at the
  border in Talli Hariya Chak area in early morning and fired at it,”
  said Mukesh Singh, additional director general of police (ADGP),
  Jammu.

- Drones continue to be the biggest security threat with Pakistan
  procuring a large number of them from the Chinese.The Intelligence
  Agencies have time and again warned that Pakistan would look to hit
  India with drones.The agencies say that Pakistan could look to
  replicate a UAE kind of strike in which drones were used.

- These drones are a potential threat and the government had been
  looking for solutions to counter this problem. In this regard the
  agencies conducted a data estimation and learnt that there are over 6
  lakh rogue or unregulated drones of various sizes and capacities.

- Drones can be easily confused with birds, which makes the surveillance
  tasks even more challenging especially in maritime areas where bird
  populations may be massive. The use of video analytics can solve the
  issue, but effective algorithms are needed able to operate also under
  unfavorable conditions, namely weak constraint, long range, reduced
  visibility, etc. Furthermore, practical systems require drones to be
  recognized at far distances, in order to allow time for reaction.
  Thus, very small objects must be recognized and differentiated against
  structured background and other challenging image contents.

- The ever-increasing widespread of unmanned aerial vehicle (UAV)
  technologies in the modern society is opening the door to a number of
  unprecedented opportunities, but poses at the same time considerable
  practical risks. This calls for the need of advanced solutions to
  counteract the possible misuse of UAVs for illegal activities. Indeed,
  due to their accessibility and ease of use, UAVs can be also employed
  for smuggling (illegal transportation at borders or in restricted
  areas), illegal surveillance, privacy violation, interference with
  aircraft operations and terrorist attacks.

- The “International Workshop on Small-Drone Surveillance, Detection and
  Counteraction Techniques” (WOSDETC) is aimed at bringing together
  researchers from both academia and industry, to share recent advances
  in this field. In conjunction, the Drone-vs-Bird Detection Challenge
  is proposed.

**JETSON NANO COMPATIBILITY**

- The power of modern AI is now available for makers, learners, and
  embedded developers everywhere.

- NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that
  lets you run multiple neural networks in parallel for applications
  like image classification, object detection, segmentation, and speech
  processing. All in an easy-to-use platform that runs in as little as 5
  watts.

- Hence due to ease of process as well as reduced cost of implementation
  we have used Jetson nano for model detection and training.

- NVIDIA JetPack SDK is the most comprehensive solution for building
  end-to-end accelerated AI applications. All Jetson modules and
  developer kits are supported by JetPack SDK.

- In our model we have used JetPack version 4.6 which is the latest
  production release and supports all Jetson modules.

**PROPOSED SYSTEM**

1.  Study basics of machine learning and image recognition.

2.  Start with implementation

- Front-end development

- Back-end development

3.  Testing, analyzing and improvising the model. An application using
    CVAT and its machine learning libraries will be using machine
    learning to identify a object and then classify it according to
    whether it’s a Bird or Drone.

4.  Use data sets to interpret the object and show on viewfinder with a
    bounding box annotation whether it is Bird or Drone.

**METHODOLOGY**

The Bird vs Drone detection model is a program that focuses on
implementing real time object detection and classification.

It is a prototype of a new product that comprises of the main module:

Jetson Nano

**1. Object Detection**

- Ability to detect the location of Object in any input image or frame.
  The output is the bounding box coordinates on the detected Object.

- For this task, initially the Data set library Kaggle was considered.
  But integrating it was a complex task so then we just downloaded the
  images from istockphoto.com and made our own data set.

- This Data set identifies Object in a Bitmap graphic object and returns
  the bounding box image with annotation of name present.

**2. Classification and process**

- Classification of the object based on whether it is Bird or Drone on
  the viewfinder.

- Hence CVAT which is a computer vision annotation tool was used along
  with XnConvert which is an image augmentation tool.

- First i downloaded the images required for the dataset from
  istockphoto.com then augmented the images on XnConvert software all in
  all i downloaded close to 250 images of birds and drones and then
  applied augmentation to them which resulted in more than 1000 images.

- In augmentation i applied grayscale to some images, 180 degree
  rotation to some images, crop and white balance changes to some images
  and mirror and solarize to some images.

**INSTALLATION**

> sudo apt-get remove –purge libreoffice\*
>
> sudo apt-get remove –purge thunderbird\*
>
> sudo fallocate -l 4.0G /swapfile1
>
> sudo chmod 600 /swapfile1
>
> sudo mkswap /swapfile1
>
> sudo vim /etc/fstab
>
> \#################add line###########
>
> /swapfile1 swap defaults 0 0
>
> \##########################
>
> \######################################
>
> \# place the voc format folder that we extracted from zip which was
> created by cvat to python/training/detection/ssd/
>
> \########################################
>
> git clone --recursive https://github.com/dusty-nv/jetson-inference
>
> cd jetson-inference
>
> docker/run.sh
>
> cd python/training/detection/ssd
>
> python3 train_ssd.py --dataset-type=voc --data=data/drone_1048
> --model-dir=models/bird2 --batch-size=4 --workers=1 --epochs=65
>
> python3 onnx_export.py --model-dir=models/bird2
>
> detectnet.py --model=models/bird2//ssd-mobilenet.onnx
> --labels=models/bird2/labels.txt --input-blob=input_0
> --output-cvg=scores --output-bbox=boxes data/drone_bird_final.mp4 \##
> To detect from a given saved video
>
> imagenet.py --model=models/my_data_50/resnet18.onnx
> --labels=data/my_data/labels.txt --input_blob=input_0
> --output_blob=output_0 /dev/video0 \## To detect from camera module

**IMAGES**

Images show the various classes of images with annotations

**ADVANTAGES**

- Radars are expensive. And military radars are seriously expensive and
  they can’t even detect the difference between drones vs birds our
  model not only is cheap but also can detect the difference between
  birds and drones easily.

- Drones vs Bird detection system detects an object and detects on
  viewfinder of camera module with good accuracy.

- It can then be used to track the position of drones i.e. where they
  are currently and at which direction they are moving.

- It can work around the clock and therefore can detect all day long.

- When completely automated no user input is required and therefore
  works with absolute efficiency and speed.

**APPLICATION**

- Detects an object and then shows whether the object is Drone or Bird
  in each image frame or viewfinder using a camera module.

- Has a vast use case scenario in Military as well as for places where
  heavy security is of major concern and places where Drones are not
  permitted.

- Can be used as a reference for other ai models based on Drone vs Bird
  Detection.

**FUTURE SCOPE**

- As we know technology is marching towards automation, so this project
  is one of the step towards automation.

- Thus, for more accurate results it needs to be trained for more
  images, and for a greater number of epochs.

- Drone vs Bird Detection model can be very easily implemented in places
  where the security is of great concern.

- Drones vs Bird Detection model can be further improved by adding more
  images of different Drones with newer models to further improve the
  detection and hence be future ready.

**CONCLUSION**

- In this project our model is trying to detect the difference between
  Drones and bird then showing it on viewfinder, live as to whether it
  is Drone or a Bird as we have specified in CVAT.

- This model tries to solve the problem of places where Drones are not
  permitted because of various concerns like security and privacy by
  detecting Drones.

- The model is efficient and highly accurate and hence works without any
  lag and also as the data is already exported to model folder can be
  made to work offline.

**REFERENCE**

1.  Datasets or images used :- <https://www.istockphoto.com/>

**ARTICLES**

1.  <https://www.mdpi.com/1424-8220/21/8/2824/html>

2.  <https://obss.com.tr/en/drone-vs-bird-detection-challenge/>

3.  <https://www.avss2020.org/pages/callForCompetitions.html>

4.  <https://www.oneindia.com/india/the-drone-threat-from-pakistan-that-india-faces-is-immense-3360708.html>

5.  <https://www.hindustantimes.com/india-news/pakistani-drone-carrying-bombs-shot-down-in-jammu-and-kashmir-101653850614293.html>
