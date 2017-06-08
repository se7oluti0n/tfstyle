Fast style-transfer
==================

Download VGG16 pretrained model from:

## Todo

* [x] content loss, style loss, total variabtion...
* [x] train op
* [ ] web app
* [ ] image2image net

## Style transfer by optimization

* Extract content features from `conv2_2` of VGG16
* Extract style features from `conv1_2`, `conv2_2`,
                  `conv3_3`, `conv4_3`,
                  `conv5_3`
* Optimize with Adam optimizer
* Results is depend on following parameters:
    * Selected features layers and model ( Which layers from which models)
    * The loss weights for each layers


### Content

Content | Style | Result
------------ | ------------- | -------------
<img src="images/chipu8.jpg" alt="Smiley face" height="300" width="300"> | <img src="images/composition_vii.jpg" alt="Smiley face" height="300" width="300"> | <img src="images/chipu_compo.jpg" alt="Smiley face" height="300" width="300">
<img src="images/chipu8.jpg" alt="Smiley face" height="300" width="300"> | <img src="images/danh-ghen-1.jpg" alt="Smiley face" height="300" width="300"> | <img src="images/chipu.jpg" alt="Smiley face" height="300" width="300">
<img src="images/chipu8.jpg" alt="Smiley face" height="300" width="300"> | <img src="images/wave.jpg" alt="Smiley face" height="300" width="300"> | <img src="images/chipu_wave.jpg" alt="Smiley face" height="300" width="300">


