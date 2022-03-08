# Skin Cancer AI with CNN
* 영상처리: 이미지 데이터 정제는 contour,bilateralFilter를 이용  
* 라이브러리: tensorflow, matplotlib, pandas, opencv 등 사용  
* 활성함수: relu(은닉층)100개, sigmoid(출력층)2개  
* 손실함수: binary_crossentropy
  
* * *

=======
**트레이닝이미지: 22,000장 검증이미지: 11,000장 테스트이미지: 40,000장**

> 첫번째 학습 결과  

**트레이닝이미지: 10,770장 검증이미지: 2,593장**  
* 모델 구성
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   

 conv2d (Conv2D)             (None, 28, 28, 10)        100       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 14, 14, 10)       0         
 )                                                               
                                                                 
 flatten (Flatten)           (None, 1960)              0         
                                                                 
 dropout (Dropout)           (None, 1960)              0         
                                                                 
 dense (Dense)               (None, 100)               196100    
                                                                 
 dense_1 (Dense)             (None, 1)                 101       
                                                                 
Total params: 196,301
Trainable params: 196,301
Non-trainable params: 0  



* 정확도 그래프  
 ![accuracy](Images\accuracy.png)
* 손실 그래프  
 ![loss](Images\loss.png)


