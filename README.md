## Neural Network Design At an Atomic Level
![image](https://user-images.githubusercontent.com/38358621/227194530-cb1d3775-6f41-43f0-ba70-efe401fe8243.PNG)

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#file-structure">Files Structure</a>
      <ul>
        <li><a href="#folders">Folders</a></li>
        <li><a href="#entire-files-structure">Entire Files Structure</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

The primary intention of writing this project is to give an overview of how one can have an insight on neural netwrok. 
<br>
In this project a basic human brain unit is created, namely, perceptron. 

* To solve a very simple question, as human, my son doesn't need to think that much, only one perceptron in his brain can be used. 
* As a caring father, I can decide to give him all the tools needed to solve the problem. 
* As a careless father, I can decide to let him learn the problem by himself. 
* As the problem become complex, and my son grows up, he can use multiple perceptron to solve the problems.

1. Simple question `AND Gate`
* Caring father provides right away the weights `10, 10, -15` to his son for solving this basic question
* Careless father let the son waste few seconds to find approximate weights ` 4.66743, 4.66364, -7.08858` for solving this basic question
* Careless father'son can live without his father

2. Complex question `XOR Gate`
* The son need more perceptrons in his brain to solve this logic, `3`
* One perceptron for `NAND gate`, which generates an intermediare solution
* One perceptron for `OR gate`, which generates an intermediare solution
* One perceptron for `AND gate`, which combines  the two intermediare solutions to yield a definitive solution

### Built With
* [c++](https://en.cppreference.com/w/)
* [Visual Studio](https://visualstudio.microsoft.com/)

## Getting Started

This is an sample code of how to implement a neural network and its physiology.
<br>
To get a local copy up and running follow these simple steps.

## File Structure

* `NNUtilities.h(cpp)` utilities functions 
* `Perceptron.h(cpp)` basic neuron unit 
* `MultiLayerPerceptron.h(cpp)` network of perceptrons 
* `NNModels.h(cpp)` examples of network of perceptrons
* `NNDesign.cpp` testing examples of a network of perceptrons


### Prerequisites
Advanced `c++` 


### Installation

Clone the repo
```sh
   git clone https://github.com/zoumson/NNDesign.git
```
 <!-- USAGE EXAMPLES -->
## Usage
* Open the main file, `NNDesign.cpp`
* Choose the neural network for testing<br>
1. AND gate<br>
2. OR gate<br>
3. NAND gate<br>
4. XOR gate<br>
5.  Seven Segment Digit Recognition, 10 outputs<br>
6.  Seven Segment Digit Recognition, 7 outputs<br>
7.  Seven Segment Digit Recognition, 1 output<br>
* Run the main file, `NNDesign.cpp`
* Option 1, outcome
```
 -----------Hardcoded AND Logic Gate-------------

Input 1 Input 2 Output
0       0       3.05902e-07
0       1       0.00669285
1       0       0.00669285
1       1       0.993307


 -----------Trained AND Logic Gate-------------

MSE = 0.233371
MSE = 0.0580662
MSE = 0.0315915
MSE = 0.0209567
MSE = 0.0154221
MSE = 0.0120921
MSE = 0.00989185
MSE = 0.00834008
MSE = 0.00719196
MSE = 0.00631083

Layer 2 Neuron 0: 4.67645  4.6727  -7.10213
Input 1 Input 2 Output
0       0       0.000822669
0       1       0.0809552
1       0       0.0812354
1       1       0.904393
```
  
<!-- ROADMAP -->
## Roadmap

All the variables have meaningful name, use them as reference



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Adama Zouma - <!-- [@your_twitter](https://twitter.com/your_username) -->- stargue49@gmail.com

Project Link: [https://github.com/zoumsonNNDesign](https://github.com/zoumson/NNDesign.git)



<!-- ACKNOWLEDGEMENTS -->

## Acknowledgements
* [Google](https://www.google.com/)
* [Stack Overflow](https://stackoverflow.com/)
* [Github](https://github.com/)
* [Linkedin: Training Neural Network in C++](https://www.linkedin.com/learning/training-neural-networks-in-c-plus-plus/create-a-neural-network-from-scratch-in-c-plus-plus?autoplay=true)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
