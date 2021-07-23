# object-tracking

주제 : 플렌옵틱(다초점 영상) 영상에서 객체(2개 이상) 추적

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
- pytorch 프레임워크로 Custom Dataset 을 구축후에 Dataloader 를 통하여 모델을 학습하고 평가함  
- tensorboard 를 통해 모델, 학습하는데 train loss, acc 와 val loss, acc 도 시각화 하여 사진으로도 저장  
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

