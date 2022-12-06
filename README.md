# Multiple object-tracking in Plenoptic video-1

- 학부 연구생, 2021.03-2021.11
- 다음 프로젝트 (플렌옵틱 영상의 다수 객체 추적 2 [link](https://github.com/jjuun0/object-tracking-2))

### 과제 요약

  - 주제: 플렌옵틱(다초점 영상) 영상에서 객체(2개 이상) 추적  
  - 목적: [DeFocus] 내가 원하는 초점이 잘 잡히지 않았을때, 그때의 시점에서 다른 새로운 초점으로 이동하기 위한 목적 프로젝트 
  - 사용: 영화나 드라마에서 주인공의 포커스를 맞춘다거나, 주인공 뒤에 있는 어떤 객체를 포커스하고 싶을 때 사용한다.

> 본 연구사업은 과학기술정보통신부의 출연금 등으로 수행하고 있는 한국전자통신연구원의 중대형 공간용 초고해상도 비정형 플렌옵틱 동영상 저작/재생 플랫폼 기술 개발 위탁연구과제 연구결과입니다.
---
### 추가한 알고리즘 및 모델 소개
  1. 단일 객체 추적 모델인 SiamRPN++ 모델(Siamese 신경망 및 유사도 기반) 사용
  2. 플렌옵틱 영상에 대한 데이터 로더 구축
  3. 포컬 스택을 구성하기 위해 Semantic 특징을 부여해 유사도 측정하는 방법을 추가함
  4. 객체 추적의 향상을 위한 RPN Anchor 앙상블 기법을 추가함
  5. 추적 속도의 향상을 위해 속도 기반 알고리즘 추가함.  
  
자세한 소개는 노션을 참고 [링크](https://fortune-scraper-694.notion.site/Plenoptic-Video-1-11b24059f39243c1ae4ea8ba441f8057)  

---
### 폴더 소개
- `PlenOpticVot_SiamRPN++/` : SiamRPN++ 모델로 플렌옵틱에 대한 다수 객체 추적  
- `YOLOv5 + DeepSORT/` : Detection + tracking 모델과 비교함.   
- `opencv_tracker/`: opencv에서 제공하는 tracker로 추적한 코드  
