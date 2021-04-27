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
