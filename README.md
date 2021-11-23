# object-tracking  

- 주제: 플렌옵틱(다초점 영상) 영상에서 객체(2개 이상) 추적  
- 목적: [DeFocus] 내가 원하는 초점이 잘 잡히지 않았을때, 그때의 시점에서 다른 새로운 초점으로 이동하기 위한 목적 프로젝트 
- 사용: 영화나 드라마에서 주인공의 포커스를 맞춘다거나, 주인공 뒤에 있는 어떤 객체를 포커스하고 싶을 때 사용한다.

# 2021.04.13  
- 모든 focal 폴더의 사진들을 한장씩 객체 추적.    
- 하지만 객체 추적이 안됨.    

# 2021.04.20
- 객체 추적 오류 해결 
  - 기존 : bounding box 를 초기화(임의의 위치 지정) 해줬음.  
    - 이를 주석처리하고, 사용자가 selectROI 를 통해 직접 객체를 선택하도록 바꿈.  
  - bounding box 를 초기화해주는건 goturn 을 사용할때, 이 모델이 객체도 자동으로 추적할 수 있게끔 해줌.  
  
- goturn tracker 사용시 에러 발생(goturn 모델, 가중치 다운받고도 안됨.) 
 
- 2d 상에서 객체 추적

# 2021.04.27
- goturn tracker 오류 : goturn 모델 자체가 문제가 있는거 같아보임.  
  - 이유 : goturn tracker 가 추적하는데 bbox 가 크기가 새로운 이미지의 범위에 벗어나서 오류 발생으로 보임.  
  - 2d 상에서도 돌려봤을때 bbox 크기가 점점 커져 화면 밖으로 나가고 에러가 발생함.  
  
- code name 변경  
  - jun.py -> test_video_tracking.py  
  - test_tracking.py -> NonVideo3_tracking.py  
  
- 각 파일이 돌아가는데 약간의 소스 파일을 올림.  
  - test_video(San_Francisco.mp4)  
  - NonVideo3 : 0번 프레임에서의 2D 이미지 10장, focal 이미지 10장  

# 2021.05.11  
- siam RPN 구조 배움.  


# 2021.05.24  
## [속도기반 알고리즘] https://github.com/jjuun0/object-tracking/issues/1
- siam RPN ++ 에서 아이디어 추가
  - 프레임 하나가 지날수록 물체의 속도를 판단한다. (유클리드 거리를 이용)  
  - 속도가 느려지면 그 객체를 탐지하는 트래커는 프레임 하나를 건너뛰고, 속도가 빨라지면 계속 트래킹한다  
- 다른 아이디어  
  - 현재 트래킹 모델 문제점이 한번 못쫓아가면 계속 못쫓아간다.   
  - 따라서 현재 트래킹에서 best score 라는 점수를 매기는데 가장 높은 스코어로 업데이트 시킨다.  
  -  이를 가장 높은 best score 만 가져오는게 아니라, 두번째 높은 값도 가져와 둘을 비교하자는 의견  

# 2021.06.08  
- 포컬 플레인 사진의 선명도를 계산해서 그중 높은 값을 가지는 사진에서 트래킹을 해보자는 의견 추가  
- SiamRPN++ 에서 하나만 현재 객체 추적이 되어있는데 이를 N개로 다수의 객체추적하는 코딩을 짜보자.  
- MOT 기술 돌려보기  

# 2021.06.15  
- 모델을 20개로 돌려봤는데 노트북 환경에서 성공적으로 돌아간다.  
- 아마 이상도 돌아갈것을 기대함.  
- 하지만 성능이 안좋은지 객체가 잘 트래킹 하지 못하는거 같음.  

# 2021.06.30  
- YOLOv5 + DeepSORT 해당 프로젝트를 돌려봤다.  
- 손흥민 선수가 요새 활약이 많아 해당 영상을 활용하여 저 모델을 실험해 봤다.  
- 꽤나 좋은 성능.  
- YOLOv5 부분만 떼어내어 SelectROI 부분만 사용자가 직접 객체를 Bounding Box를 만들어내서 돌려보자.  

# 2021.07.13  
- demo_focal.py  
- sot 모델로 mot 개발중임 : 사용자가 selectroi 로 객체를 선택하고 트래킹하는 방식  
  - 에러 : 객체가 서로 가까우면 BBox 가 업데이트 되면서 물체가 같아져버림  


# 2021.07.17  
- focal 이미지들을 YOLOv5 + DeepSort 에 적용시킴.  
- 기존 이미지 1920 * 1080 -> 960 * 544 로 사이즈를 변경후 detect & tracking  
- 한 프레임당, focal 이미지들을 덧 그려봄  
- frame : 20 ~ 90   
- focal : 15 ~ 50   
- 한 프레임당 트래킹 연산 시간 : 21초   
- 전체 총 걸린시간 : 1504초 -> 25분  

# 2021.07.18  
- 지난 sot 에러 문제점 : sot 가 anchor 의 Best Score 로 트래킹을 업데이트 하게 되는데 이때 물체가 서로 가까우면 다른 객체를 Best Score 로 채택하는 문제점이였다.  
- 따라서 성능이 안좋다보니 아이디어를 추가해 성능 보완을 해야한다.  

- 첫번째 아이디어  
  - Best Score 을 높은 스코어 두개를 뽑아낸다.  
  - 이때 전 프레임의 BBox 의 중심점과 바로 위에서 뽑아낸 BBox 의 중심점을 유클리드 거리를 이용하여 계산하고 두 거리중 더 짧은 것으로 채택하면 어떨지? 하는것  

- 두번째 아이디어  
  - SelectROI 에서 선택한것을 먼저 DeepSORT 에 넘겨줘 id 를 부여한다  
  - 다음 프레임에서는 이 SelectROI 에서 rpn 기능을 추가한? anchor 를 만들어내주는 모델을 찾고, 이러한 정보를 DeepSORT 에 넘겨준다.  
  - 이때의 confidence 값은 저 anchor 가 만들어준 BBox 가 물체가 있는지 없는지 판단해주는 classfication 을 적용시킨다.

# 2021.07.23
- YOLOv5 + DeepSORT 모델을 포컬 플레인에서 돌린 결과를 프레임별 평균 트래킹 속도를 그래프로 표현
- 트래킹한 결과 동영상으로 바꿈
- https://github.com/jjuun0/object-tracking/blob/master/YOLOv5%20%2B%20DeepSORT/result/YOLOv5%2BDeepSORT%20in%20Focal%20Planes%20Graph.png

# 2021.07.26  
- SiamRPN++  
  - sot 문제점 해결 : 2021.07.18 에서의 문제점을 해결함.  
    - 기존에는 모델을 하나 만들고 트래커를 여러개 만드는 방식에서 트래커가 업데이트 하는데 모델 파라미터에 영향을 끼치는것을 발견.   
    - 따라서 모델을 여러개 만들고 모델의 트래커는 하나만 만드는 방식으로 변경.  

  - 문제점: 물체가 가려지는 object tracking occlusion 현상이 발생.    
    - 해결: 기존에는 물체의 속도가 느려지면 한 프레임을 건너뛰고 다음 프레임을 트래킹 했는데, 이를 수정.  
    - 연속으로 두번 건너뛰었다면 전체 포컬 플레인 영역 이미지를 다시 재 트래킹해서 max index 값을 찾는다.  
 
- YOLOv5 + DeepSORT  
  - 기존 tracking 알고리즘: 모든 프레임마다, 모든 포컬 플레인 영역을 트래킹함.    
  - 수정: 현 프레임에서 가장 많은 트래킹한 물체의 개수가 많은 포컬 플레인 인덱스를 찾고, 다음 프레임에서 포컬 플레인 탐색 영역을 -3 ~ +3 만 트래킹할 수 있도록 함.  
  - 만약 max index 가 없다면 전체 영역을 다시 트래킹함.   

# 2021.08.09  
## [선명도] https://github.com/jjuun0/object-tracking/issues/3
- 선명도 모듈 반영하여 포컬 플레인 영역 범위 조절.  
  - 데이터 셋으로 반영시에 성능이 괜찮아보여 채택

# 2021.08.16  
## [이동방향 예측] https://github.com/jjuun0/object-tracking/issues/5
- 객체의 이동방향을 예측하는 알고리즘 추가 
  - 하지만 매번 tracker 가 예측한 BBox 좌표가 매우 미세한 예민한 값을 가짐으로 이 알고리즘 적용 불가능 판단 

## [Anchor 의 평균좌표로 tracker 이동] https://github.com/jjuun0/object-tracking/issues/6
- 해결: tracker의 anker box 가 5개가 나오게 되는데 이를 평균값으로 tracker 가 예측을 하게끔 한다.
- BBox 가 갑자기 값이 튀는것을 방지하고, tracker 의 BBox 예측값을 한번 의심해보는 방향.

# 2021.08.23  
## [SA-Siam에서의 semantic feature 적용] https://github.com/jjuun0/object-tracking/issues/4
새로운 방식 추가: img classfication task 의 모델 VGG 를 가져와서 correlation 연산을 통해 유사한 focal index 를 찾는 방식 제안  
- VGG 의 pretrained weight 를 불러와 사용한다.  
  - Feature 부분만 떼어내어 target 이미지와 search reagion 영역의 이미지를 이 VGG 네트워크에 인풋으로 넣은후 둘을 DepthwiseXCorr 연산을 통해 어떤 값이 나오는지 확인해봄.  
  - Feature 분석을 어떻게 해야할지 모르겠다.  
- 여태 사용했던 알고리즘 정리하여 ppt 올림.  

# 2021.08.27  
- VGG 네트워크를 통과후 DepthwiseXCorr 연산을 통해 output 의 shape 크기가 5*5*512 이다.  
  - 5 x 5 면 너무 크기가 작아 정보가 부족하다. input 의 크기를 조절하자.  
  
- 분산이 크게 잡힌 부분의 Best Score 값을 한번 측정해보자.  
  - 1. 분산이 크게 잡히면 Best Score 값이 높게 나오는것인지.  
  - 2. 높게 나오는게 아니라 Best Score 이 넓게 퍼져있는것이 아닌지.  

- 새로운 비디오 3개를 받았다. 어떤 비디오를 사용할지, 어느 구간을 트래킹하는데 사용할지를 정하자.    

- VGG feature 분석: 현재 5 x 5 에서 높게 찍히는 위치가 다르므로 해당 인덱스의 위치를 활용하여 focal index 를 측정해보자.
- 결과: focal index 와 gt index 차이가 많이 나서 사용 불가능 판단.  

# 2021.09.17  
- SOLO: Segmenting Objects by Locations 의 Segment model 을 돌려봤다.
- 모델 DECOUPLED_SOLO_R50_3x pretraine  
  ![image](https://user-images.githubusercontent.com/66052461/142971924-3ebac53d-a843-4e8f-af41-12a60bc20a62.png)  
  
- Focal 데이터에 적용: focal 데이터이다 보니 segment된 객체의 개수가 적고, 꽃병을 찾지못함.     
  ![image](https://user-images.githubusercontent.com/66052461/142972025-74810b9f-e224-4444-a18f-be3c93cb2c89.png)  
  
- 2D image 데이터에 적용: image 데이터는 focal 데이터보다는 객체를 많이 탐지함.  
  ![image](https://user-images.githubusercontent.com/66052461/142972106-a62c9699-847a-4bd9-b2af-e8db95f70c3a.png)  



# 2021.09.24
- ground truth 만들고 IoU 측정
  - 기존 tracker 를 이용해서 gt를 만들고 눈으로 확인하면서 값이 맞는지 체크하고, 부족하다면 손으로 수정할 계획이다.
  - 1. csv 형식으로 저장하고, x, y, h, w 형식으로 저장한다.
  - 2. gt 를 읽어오는것을 pandas 를 이용함
  - 3. gt 와 실제 tracker 의 bbox 를 이용하여 IoU 를 측정한다. 

# 2021.10.08  
- 새로운 영상 받아옴  
  - 하지만 잘 트래킹이 안되는 상황 
  - 예시 1   
    ![1007_180305](https://user-images.githubusercontent.com/66052461/136368044-8a866977-ce0e-4f40-9096-e34c71257c17.gif)  
  - 예시 2  
    ![1004_212707](https://user-images.githubusercontent.com/66052461/136368484-96dc24c4-da13-4637-b53e-8ef286b6ac62.gif)  

## [Best score 값에 따른 focal plane 영역 설정] https://github.com/jjuun0/object-tracking/issues/2
- 해결: (기존) 선명도를 반영한 focal plane 영역 설정 -> Naive 한 전 프레임에서의 주변 -7 ~ +7 범위 탐색  

# 2021.10.22  
- 평가 지표: IoU, Distance(gt와 tracking 결과의 BBox 좌상단 좌표와의 유클리디안 거리를 나타냄)  
- xywh 좌표를 매 프레임마다 검출하여 csv 로 저장하고 gt 와 비교를 통해 IoU, Distance 값도 csv 로 저장하여 그래프를 그려봄  
- 추가 영상 받음: 강강수월래 영상을 테스트해봤으나 시간도 부족하고, 영상 자체가 어려워 트래킹하기 어렵다  
  ![ganggang](https://user-images.githubusercontent.com/66052461/142971485-c09d4b6b-b07b-4a70-914f-fb1dbafad2c2.gif)  
  
# 2021.11.12
최종 그래프 산출 및 영상 산출  

## graph
- IoU  
  ![image](https://user-images.githubusercontent.com/66052461/142964829-a80ea36f-058b-401c-9add-c256d9c0803f.png)  

- Distance  
  ![image](https://user-images.githubusercontent.com/66052461/142964890-84363afe-b4ce-4900-ae36-7d6bb464a72e.png)  

- Time (알고리즘별 시간 측정)  
  ![image](https://user-images.githubusercontent.com/66052461/142964945-aa54a671-bf6e-4835-af0e-c0e6101a985b.png)  

## GIF  
- 2D  
  ![2d](https://user-images.githubusercontent.com/66052461/142970090-4cf8e9f2-b18c-47ba-9933-7d365ff83f73.gif)  

- Naïve: 한 프레임에 모든 객체가 트래킹  
  ![naive](https://user-images.githubusercontent.com/66052461/142971096-e2b8e1ec-e9db-4896-88f6-afd2c03bf63e.gif)  

- 1 obj per frame: 한 프레임에 하나의 객체만 트래킹  
  ![1frame1obj](https://user-images.githubusercontent.com/66052461/142971220-d46be84d-010c-4d86-9123-3d49e25886f8.gif)  

- Motion Adaptive: 속도 기반으로 하여 객체의 속도가 빨라지는 경우만 트래킹  
  ![result](https://user-images.githubusercontent.com/66052461/142965765-a2e2a0cf-d3cf-4a73-9d7f-b22701bc9607.gif)  




