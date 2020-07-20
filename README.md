# Revisiting Singing Voice Detection : a quantitative review and the future outlook

This repo contains code for the paper "**Revisiting Singing Voice Detection: a Quantitative Review and the Future Outlook**" by Kyungyun Lee, Keunwoo Choi and Juhan Nam at the 19th International Society for Music Information Retrieval Conference (ISMIR) 2018. [[pdf](http://arxiv.org/abs/1806.01180), [blog post](https://kyungyunlee.github.io/blog/ismir2018)]      

### Requirements  
* specified in requirements.txt

### Public Dataset 
* [Jamendo](http://www.mathieuramona.com/wp/data/jamendo/) with the same labeling, train/valid/test set split as described in the website.  
* [MedleyDB](http://medleydb.weebly.com/)       
We used 61 songs that contain vocals, which can be found in `medleydb_vocal_songs.txt`.    
**Note : MedleyDB does not provide vocal annotations, so we generated labels using the provided instrument activation annotation.**   
Download the songs, change path, and run `python medley_voice_label.py` to generate labels for the 61 songs.   

### Dataset for stress testing (section 5)  
To generate dataset, run 
* `python vibrato_data_gen.py` for vibrato test in section 5.1.  
* `python snr_data_gen.py` for SNR test in section 5.2. (Requires modification for path to MedleyDB vocal containing songs.)

### Reproduction of singing voice detection models (section 3)  
There are 3 reproduced models in the following folders :    
* `lehner_randomforest` [1]  
* `schluter_cnn` [2]
* `leglaive_lstm` [3]  
**Note : Set paths for datasets in each config files within the model folders** 


Commandline arguments are :
* `--model_name` : whatever name you set it during training, and will be saved in `./weights/` folder.
* `--dataset` : one of {`"jamendo", "vibrato", "snr"`}. New dataset can be added with modification in `load_data.py` (might add RWC pop).   

In each model folder, audio processor to preprocess data must be run before playing around with the model. 
* `python audio_processor.py --dataset "jamendo"` in CNN and RNN model with {`"jamendo", "vibrato", "snr"`}  
* `python vocal_var.py --dataset "jamendo"" ` in randomforest model with {`"jamendo", "vibrato", "snr"`}  
**Note : This file for randomforest computes vocal variance and concatenates them with the features extracted from the matlab code provided by the authors of [1]. So, this file only provides functions for computing the vocal variance. Either you can add onto this file to compute other features or you can find the matlab code ;)**  

#### To train models, run the following in each model folder 
* `python main.py --model_name "mynewmodel" `
#### To run pretrained models (models are provided in ./weights/ folder), run the following in each model folder 
* `python test.py --model_name "mynewmodel" --dataset "jamendo" `


### References 
* [1] Bernhard Lehner, Gerhard Widmer, and Reinhard Sonnleitner. "On the reduction of false positives in singing voice detection." [pdf](https://pdfs.semanticscholar.org/ef89/585dfb286b7920ed19a4fb6856876fa180fc.pdf)    
* [2] Jan Schlueter and Thomas Grill. "Exploring data augmentation for improved singing voice detection with neural networks." [pdf](http://www.ofai.at/~jan.schlueter/pubs/2015_ismir.pdf)    
* [3] Simon Leglaive, Romain Hennequin, and Roland Badeau. "Singing voice detection with deep recurrent neural network." [pdf](https://hal.archives-ouvertes.fr/hal-01110035/document)    


### TO DO (2018.06)
* Upload notebook file for model analysis and audacity compatible label generation. 

