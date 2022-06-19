# federated-learning

## Project steps
1. Get familiar with the literature and the state of the art **read [[1]](#1) [[2]](#2) [[3]](#3)**

2. Implement your baseline
   - centralized experiment + finding best learnning rate
   - divide dataset among client
   - implement FedAvg baseline

3. Ablation studies: study the variations occurring when modifying the clients’ local data distribution and the value of the FL parameters.
4. Normalization layers : study the effect of adding batch normalization layer and group normalization
5. Time for your personal contribution! : extension idea / any contribution

## What have been implemented so far
1. the centralized experiment 
2. data devision among clients 
3. FedAvg + paramater study
4. FedProx algorithm [[4]](#4)
5. FedIN algorithm [[3]](#3)
6. Batch layer experiments
7. an extension idea (includes an autoencoder and a gan )

## still missing or can be improved
1. group normalization layer
2. the experiments tackeled only heteroginity of labels and it asumed all clients have same nb of samples ( client data size imbalance could be tackeled)
3. all the experiments are conducted on a valdation set , the test set is still untouched for a final comparison 
4. didn't calculate the mean accuracy among client accuracy, the metric used untill now was an accuracy on a global centered validation set
5. other ideas can also be exploited

### notes
- when **alpha** `the parameter controlling heteroginity among clients` is not specified, it is equal to 1 (default value) *you can encouter that sometimes in the name of image files or inside the presentation*
- sorry for the image names

## Materials
<a id="1">[1]</a> Google AI Blog, Federated Learning: [Collaborative Machine Learning without Centralized
Training Data](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html).

<a id="2">[2]</a> McMahan, Brendan et al. “[Communication-Efficient Learning of Deep Networks from
Decentralized Data](https://arxiv.org/abs/1602.05629)” Proceedings of the 20th International Conference on Artificial Intelligence
and Statistics, PMLR 54:1273-1282, (2017).

<a id="3">[3]</a> Hsu TM.H. et al. “[Federated Visual Classification with Real-World Data Distribution](https://arxiv.org/abs/2003.08082)” European
Conference on Computer Vision . ECCV 2020. Lecture Notes in Computer Science, vol 12355.
Springer, Cham.

<a id="4">[4]</a> Tian Li, et al. “[Federated optimization in heterogeneous networks.](https://arxiv.org/abs/1812.06127)” In I. Dhillon, D.
Papailiopoulos, and V. Sze, editors, Proceedings of Machine Learning and Systems, volume 2,
pages 429–450, 2020
