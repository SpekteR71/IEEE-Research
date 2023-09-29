# IEEE-Research
Reinforcement Learning and Ensemble Approach to Malware Detection

This is part of a reseach project pursued over the summer break in 2023. Implemented the Double Deep Q Learning algorithm to effectively classify android malware based on the API calls and permissions requested.
This approach is an improvement over classic Q learning and Deep Q learning as this effectively addresses the overestimation problem of DQN. Achieved 90% accuracy, 94% precision and 85.6% recall.

Here is a brief overview of the DDQN algorithm
![image](https://github.com/SpekteR71/IEEE-Research/assets/106680171/2d011aca-ff1a-4116-8f3e-e1e696246451)

The ensemble model is made by stacking a CNN model for statics features and an RNN model for the dynamic features of Windows Malware. Stacking and gradient boosting, both were compared. Boosting resulted in the best possible ensemble model giving and accuracy of 97%.
