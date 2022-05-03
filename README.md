# Skin Cancer AI with CNN
* 영상처리: 이미지 데이터 정제는 contour,bilateralFilter를 이용  
* 라이브러리: tensorflow, matplotlib, pandas, opencv 등 사용  
* 활성함수: relu(은닉층), sigmoid(출력층)
* 손실함수: binary_crossentropy  
* 사용한 모델: VGG16, ResNet50
  
* * *


> 첫번째 학습 결과  

**트레이닝이미지: 10,770장 검증이미지: 2,593장**  


* 정확도 그래프  
 ![accuracy](https://github.com/Oldentomato/Skin_Cancer_AI/blob/main/Images/accuracy.png?raw=true)
* 손실 그래프  
 ![loss](https://github.com/Oldentomato/Skin_Cancer_AI/blob/main/Images/loss.png?raw=true)



> 두번째 학습 결과   

**트레이닝이미지: 2,463장 검증이미지: 1,196장**  
  
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


> 다섯번째 학습 결과   
      
**트레이닝이미지: 10,682장 검증이미지: 3560장 테스트이미지: 3562장 100epoch**  
* 모델 구성  
ImageDataGenerator 사용  
ResNet50 사용  
learning_rate = 0.0001
모든 Conv_Base Freezing 후 재학습 (fine_tunning)  
이미지 크기: 512,512,3  

* 정확도 그래프  
 ![accuracy](https://github.com/Oldentomato/Skin_Cancer_AI/blob/main/Images/res(91)acc.png?raw=true)
* 손실 그래프  
 ![loss](https://github.com/Oldentomato/Skin_Cancer_AI/blob/main/Images/res(91)loss.png?raw=true)  
 Classification Report  

              precision    recall  f1-score   support  

    Melanoma       0.50      1.00      0.67      1780  
 NotMelanoma       0.00      0.00      0.00      1780  

    accuracy                           0.50      3560  
   macro avg       0.25      0.50      0.33      3560  
weighted avg       0.25      0.50      0.33      3560  

> 여섯번째 학습 결과   
      
**트레이닝이미지: 10,682장 검증이미지: 3560장 테스트이미지: 3562장 100epoch**  
* 모델 구성  
ImageDataGenerator 사용  
VGG16 사용  
learning_rate = 0.0001  
모든 Conv_Base Freezing 후 재학습 (fine_tunning)  
이미지 크기: 512,512,3  

Classification Report  

              precision    recall  f1-score   support  

    Melanoma       0.94      0.94      0.94      1780  
 NotMelanoma       0.94      0.94      0.94      1780  
 
    accuracy                           0.94      3560  
   macro avg       0.94      0.94      0.94      3560  
weighted avg       0.94      0.94      0.94      3560  

> 일곱번째 학습 결과   
      
**트레이닝이미지: 10,682장 검증이미지: 3560장 테스트이미지: 3562장 100epoch**  
* 모델 구성  
ImageDataGenerator 사용  
DenseNet121 사용  
learning_rate = 0.0001  
모든 Conv_Base Freezing 후 재학습 (fine_tunning)  
이미지 크기: 512,512,3  

* 정확도 그래프  
 ![accuracy](https://github.com/Oldentomato/Skin_Cancer_AI/blob/main/Images/denseacc.png?raw=true)
* 손실 그래프  
 ![loss](https://github.com/Oldentomato/Skin_Cancer_AI/blob/main/Images/denseloss.png?raw=true)


Classification Report

              precision    recall  f1-score   support

    Melanoma       0.96      0.92      0.94      1780
 NotMelanoma       0.92      0.96      0.94      1780

    accuracy                           0.94      3560
   macro avg       0.94      0.94      0.94      3560
weighted avg       0.94      0.94      0.94      3560