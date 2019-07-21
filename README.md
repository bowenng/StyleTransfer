# Style Transfer with improved Color Consistency Loss.

## Color Consistency Loss
Color Consistency Loss essentially tries to minize the Eulidean Distance between the IQ dimension in the YIQ space between the style image and the generated image, therefore allowing the network to selective choose which areas to apply styles and which areas to avoid applying styles. As a result, network with color consistency loss tends to produce less noisy and more visually appealing results.


### Starry Night
![StarryNight](https://github.com/bowenng/StyleTransfer/blob/master/sample_outputs/StarryNight.jpg)

Without Color Consistency Loss:
![UR_StarryNight_Original](https://github.com/bowenng/StyleTransfer/blob/master/sample_outputs/StarryNightOriginal.png)

With Color Consistency Loss:
![UR_StarryNight_Improved](https://github.com/bowenng/StyleTransfer/blob/master/sample_outputs/StarryNightImproved.png)

Without Color Consistency Loss:
![Zheng_StarryNight_Original](https://github.com/bowenng/StyleTransfer/blob/master/sample_outputs/ZhengOriginal.png)

With Color Consistency Loss:
![Zheng_StarryNight_Improved](https://github.com/bowenng/StyleTransfer/blob/master/sample_outputs/ZhengImproved.png)


#### Van Gogh
![Vangogh](https://github.com/bowenng/StyleTransfer/blob/master/sample_outputs/Van.jpg)

Without Color Consistency Loss:
![UR_Vangogh_Original](https://github.com/bowenng/StyleTransfer/blob/master/sample_outputs/VangoghOriginal.png)

With Color Consistency Loss:
![UR_Vangogh_Improved](https://github.com/bowenng/StyleTransfer/blob/master/sample_outputs/VangoghImproved.png)
