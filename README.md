# Voice Conversion using Generative Adversarial Nets conditioned by Phonetic Posterior Grams

##1.Prepare Phase
###1)Set Path
###2)Set HyperParameter
###3)install requirements
Builtin library
tensorflow >= 1.1
numpy >= 1.11.1
librosa == 0.5.1
soundfile
tqdm

External library
libav-tools

##2.Train Phase

###1)Train1 (SR Model)
python3 train1.py 'SR Model Directory Name'

###2)Train2 (Synthesis Model)
python3 train2.py 'SR Model Directory Name' 'Synthesis Model Directory Name'


##3.Evaluate Phase

###1)Eval1 (SR Model)
python3 eval1.py 'SR Model Directory Name'

###2)Eval2 (Synthesis Model)
python3 eval2.py 'SR Model Directory Name' 'Synthesis Model Directory Name'

##4.Convert Phase
python3 convert.py 'Synthesis Model Directory Name'

