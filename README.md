# Skin Cancer AI with CNN
* 영상처리: 이미지 데이터 정제는 contour,bilateralFilter를 이용  
* 라이브러리: tensorflow, matplotlib, pandas, opencv 등 사용  
* 활성함수: relu(은닉층), sigmoid(출력층)
* 손실함수: binary_crossentropy  
* 사용한 모델: VGG16, ResNet50
  
* * *


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
 ![accuracy](https://github.com/Oldentomato/Skin_Cancer_AI/blob/main/Images/accuracy.png?raw=true)
* 손실 그래프  
 ![loss](https://github.com/Oldentomato/Skin_Cancer_AI/blob/main/Images/loss.png?raw=true)



> 두번째 학습 결과  

**트레이닝이미지: 2,463장 검증이미지: 1,196장**  
* 모델 구성
Model: "sequential_9"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   

 conv2d_25 (Conv2D)          (None, 28, 28, 16)        160       
                                                                 
 max_pooling2d_25 (MaxPoolin  (None, 14, 14, 16)       0         
 g2D)                                                            
                                                                 
 conv2d_26 (Conv2D)          (None, 14, 14, 16)        2320      
                                                                 
 max_pooling2d_26 (MaxPoolin  (None, 7, 7, 16)         0         
 g2D)                                                            
                                                                 
 conv2d_27 (Conv2D)          (None, 7, 7, 16)          2320      
                                                                 
 max_pooling2d_27 (MaxPoolin  (None, 3, 3, 16)         0         
 g2D)                                                            
                                                                 
 flatten_9 (Flatten)         (None, 144)               0         
                                                                 
 dense_25 (Dense)            (None, 128)               18560     
                                                                 
 dropout_16 (Dropout)        (None, 128)               0         
                                                                 
 dense_26 (Dense)            (None, 32)                4128      
                                                                 
 dropout_17 (Dropout)        (None, 32)                0         
                                                                 
 dense_27 (Dense)            (None, 1)                 33        
                                                                 
Total params: 27,521
Trainable params: 27,521
Non-trainable params: 0
_________________________________________________________________  

* 정확도 그래프  
 ![accuracy](https://github.com/Oldentomato/Skin_Cancer_AI/blob/main/Images/accuracy_2.png?raw=true)
* 손실 그래프  
 ![loss](https://github.com/Oldentomato/Skin_Cancer_AI/blob/main/Images/loss_2.png?raw=true)


> 세번째 학습 결과  
**트레이닝이미지: 940장 검증이미지: 228장 200epoch**  
* 모델 구성  
ImageDataGenerator 사용  
VGG16 사용  
block4_conv1 부터 학습가능으로 설정 (fine_tunning)  
이미지 크기: 512,512,3  


* 정확도 그래프  
 ![accuracy](https://github.com/Oldentomato/Skin_Cancer_AI/blob/main/Images/vgg_acc.png?raw=true)
* 손실 그래프  
 ![loss](https://github.com/Oldentomato/Skin_Cancer_AI/blob/main/Images/vgg_loss.png?raw=true)

 > 네번째 학습 결과  
**트레이닝이미지: 940장 검증이미지: 228장 100epoch**  
* 모델 구성  
ImageDataGenerator 사용  
ResNet50 사용  
conv4_block1_1_conv 부터 학습가능으로 설정 (fine_tunning)  
이미지 크기: 512,512,3  

* 정확도 그래프  
 ![accuracy](https://github.com/Oldentomato/Skin_Cancer_AI/blob/main/Images/res_acc.png?raw=true)
* 손실 그래프  
 ![loss](https://github.com/Oldentomato/Skin_Cancer_AI/blob/main/Images/res_loss.png?raw=true)