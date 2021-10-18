# Cross-patient Seizure Detection with Feature-Mapped Convolutional Neural Networks on Scalp EEG
Modification of ICU EEG Seizure Detector from @danieljkim0118 at the Center of Neuroengineering and Therapeutics at the University of Pennsylvania.

Continuous EEG (cEEG) monitoring has been increasingly demonstrated to be important for diagnosing patients’ conditions within intensive care units (ICUs), 
being linked to lower mortality rates. Most notably, seizures arise from patients who not only suffer from epilepsy or neurological injury, but also from 
those who experience high fever or viral infections including COVID-19. Identifying seizures is therefore a crucial task to monitoring patients’ health status 
in intensive care settings. However, the costly and inconsistent nature of clinicians manually annotating the recordings prevents many hospitals from utilizing 
this technique. This paper demonstrates the usage of deep learning to largely automate the seizure annotation process and allow clinicians to read through a 
highly condensed set of EEG recordings with high seizure probabilities, thereby reducing costs while increasing quality of the annotations. The algorithm first 
preprocesses the data and generates distribution maps of statistical EEG features across the scalp. Convolutional Neural Networks are then utilized to extract 
spatial features, and the model outputs are post-processed to return a highly reduced set of EEG with high seizure probabilities to be further inspected by an 
expert clinician.
