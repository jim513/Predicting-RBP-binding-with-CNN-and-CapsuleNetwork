# Predicting-RBP-binding-with-CNN
 딥러닝 기법을 이용한 단백질-핵산 결합 모티프 발굴 프레임워크입니다. CNN과 Capsule Netwokr 기법을 이용하여 RNA 염기서열을 분류할 수 있습니다. 모티프 발굴을 진행하기위하여 MEME 5.2이상의 버전이 요구됩니다. lncRNA 염기서열 데이터셋이 동봉되어 테스트를 진행해 볼 수 있습니다. AUC-ROC 성능 그래프로 성능을 확인할 수 있습니다.
 
# Dependency
Linux

Python = 3.7.9

Keras = 2.4.3

tensorflow = 2.3.1

pandas =1.1.2

numpy =1.19.1

matplotlib = 3.1.0

scikit-learn = 0.19.0

scipy = 1.5.2

biopython = 1.78

meme = 5.2.0

# Execute Step

1. 사용 알고리즘(CNN.py 혹은 Caps.py)에서 사용할 데이터 경로를 수정해주세요.

2. python CNN.py 혹은 python Caps.py 커맨드로 프로그램을 실행해주세요.

3. meme 패키지를 설치하여 streme --oc output_file --rna --p Positive_sequence.txt --n Negative_sequence.txt 명령어를 실행해주세요.

P.s)meme 설치 경로 http://meme-suite.org/doc/download.html

# reference
1.	Taeho Jo, Jie Hou, Jesse Eickholt, Jianlin Cheng. “Improving Protein Fold Recognition by Deep Learning Networks” 2015년 12월 4일. ttps://www.nature.com/articles/srep17573 (액세스: 2020년 9월)
2.	Rafsanjani MuhammodAhmed, Dewan Md Farid, Swakkhar Shatabda, Alok Sharma, Abdollah DehzangiSajid. “PyFeat: a Python-based effective feature generation tool for DNA, RNA and protein sequences.” 2019년 03월 08일. https://academic.oup.com/bioinformatics/article/35/19/3831/5372339 (액세스: 2020년 09월).
3.	Qinhu Zhang, Lin Zhu, De-Shuang Huang .“High-Order Convolutional Neural Network Architecture for Predicting DNA-Protein Binding Sites” 2018년 3월 26일 https://ieeexplore.ieee.org/document/8325519/authors#authors (액세스: 2020년 9월)
4.	Shao-Wu ZhangWang, Xi-Xi Zhang, Jia-Qi WangYa. “Prediction of the RBP binding sites on lncRNAs using the high-order nucleotide encoding convolutional neural network.” 2019년 10월 15일.    https://www.sciencedirect.com/science/article/pii/S0003269719303513 (액세스: 2020년 09월).
5.	Zhen ShenDeng, De-shuang HuangSu-Ping. “Capsule Network for Predicting RNA-Protein Binding Preferences Using Hybrid Feature.” 2019년 09월 24일. https://ieeexplore.ieee.org/document/8847396 (액세스: 2020년 09월).
6.	Zhen Shen, Qinhu Zhang, Kyungsook Han, De-Shuang Huang. “A Deep Learning Model for RNA-Protein Binding Preference Prediction based on Hierarchical LSTM and Attention Network”. 2020년 7월 7일. https://ieeexplore.ieee.org/document/9134909 (액세스:2020년 11월)
7.	Byungkyu Park, Kyungsook Han*.” Discovering protein-binding RNA motifs with a generative model of RNASequences”. 2019년 6월 17일 https://www.sciencedirect.com/science/article/pii/S1476927119305365 (액세스: 2020년 11월)
