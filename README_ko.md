<p align="center">
    <img src="https://raw.githubusercontent.com/neuml/txtai/master/logo.png"/>
</p>

<h3 align="center">
    <p>Build AI-powered semantic search applications</p>
</h3>

<p align="center">
    <a href="https://github.com/neuml/txtai/releases">
        <img src="https://img.shields.io/github/release/neuml/txtai.svg?style=flat&color=success" alt="Version"/>
    </a>
    <a href="https://github.com/neuml/txtai/releases">
        <img src="https://img.shields.io/github/release-date/neuml/txtai.svg?style=flat&color=blue" alt="GitHub Release Date"/>
    </a>
    <a href="https://github.com/neuml/txtai/issues">
        <img src="https://img.shields.io/github/issues/neuml/txtai.svg?style=flat&color=success" alt="GitHub issues"/>
    </a>
    <a href="https://github.com/neuml/txtai">
        <img src="https://img.shields.io/github/last-commit/neuml/txtai.svg?style=flat&color=blue" alt="GitHub last commit"/>
    </a>
    <a href="https://github.com/neuml/txtai/actions?query=workflow%3Abuild">
        <img src="https://github.com/neuml/txtai/workflows/build/badge.svg" alt="Build Status"/>
    </a>
    <a href="https://coveralls.io/github/neuml/txtai?branch=master">
        <img src="https://img.shields.io/coveralls/github/neuml/txtai" alt="Coverage Status">
    </a>
</p>

-------------------------------------------------------------------------------------------------------------------------------------------------------

txtai는 데이터를 변환하고 AI를 기반으로 한 의미론적 검색 애플리케이션을 만들기 위해 머신러닝 워크플로우를 실행합니다.

![demo](https://raw.githubusercontent.com/neuml/txtai/master/demo.gif)

기존에 사용하던 검색 시스템은 데이터를 찾기 위해서 키워드를 사용합니다. 의미론적 검색 애플리케이션은 자연어를 이해하고, 단지 동일한 키워드를 가진게 아닌 동일한 의미를 갖는 결과를 식별합니다.

최첨단 머신 러닝 모델의 지원에 의해, 데이터는 검색을 위한 벡터 표현(임베딩(embeddings)이라고 알려진)으로 변환됩니다. 혁신은 빠른 속도로 일어나고 있으며, 이 모델은 문서, 오디오, 이미지 등의 개념을 이해할 수 있습니다.

txtai 특징 요약:

- 🔎 다수의 index backends를 이용해 대규모 유사성 검색 ([Faiss](https://github.com/facebookresearch/faiss), [Annoy](https://github.com/spotify/annoy), [Hnswlib](https://github.com/nmslib/hnswlib))
- 📄 텍스트 조각, 문서, 오디오, 이미지 및 비디오에 대한 embeddings를 만듭니다. 변환기와 단어 벡터를 지원합니다.
- 💡 머신 러닝 파이프라인은 회신, zero-shot 라벨링, 표기, 번역, 요약 및 텍스트 추출과 같은 질문을 추출하기 위해 사용됩니다.
- ↪️️ 워크플로우는 파이프라인을 연결하여 비즈니스 로직을 합칩니다. txtai 프로세스는 마이크로 서비스 또는 완전한 인덱싱 워크플로우일 수 있습니다.
- 🔗 [JavaScript](https://github.com/neuml/txtai.js), [Java](https://github.com/neuml/txtai.java), [Rust](https://github.com/neuml/txtai.rs)와 [Go](https://github.com/neuml/txtai.go)용 API 바인딩
- ☁️ 컨테이너 배포 관리 시스템 (예: Kubernetes)으로 확장되는 Cloud-native 구조

응용 프로그램은 복잡한 NLP기반 데이터 추출을 위한 유사성 검색에서 구조화된 데이터베이스를 생성하기까지의 범위입니다. 다음의 애플리케이션은 txtai를 기반으로 구동됩니다.

| 응용 프로그램  | 설명  |
|:----------|:-------------|
| [paperai](https://github.com/neuml/paperai) | 의료/과학 논문을 위한 AI기반 문헌 발견 및 검토 엔진 |
| [tldrstory](https://github.com/neuml/tldrstory) | 헤드라인과 본문에 대한 AI기반 이해 |
| [neuspo](https://neuspo.com) | 사실 기반의 실시간 스포츠 이벤트 및 뉴스 사이트 |
| [codequestion](https://github.com/neuml/codequestion) | 터미널에서 직접 코딩에 관한 질문을 합니다. |

txtai는 Python 3.6+, [Hugging Face Transformers](https://github.com/huggingface/transformers), [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) 그리고 [FastAPI](https://github.com/tiangolo/fastapi) 로 만들어졌습니다.

## 왜 txtai인가요?

기존 검색 시스템 외에도 점점 더 많은 의미론적 검색 솔루션을 사용할 수 있는데, 왜 txtai를 사용해야 하죠?

- `pip install txtai` 만 있으면 시작할 수 있습니다.
- 소규모 데이터와 빅 데이터에서 모두 잘 작동하며, 프로토타입을 몇 줄의 코드로 제작할 수 있고, 필요에 따라 확장할 수 있습니다.
- 사전 및 사후 처리 데이터를 위한 풍부한 데이터 프로세싱 프레임워크(파이프라인과 워크플로우)
-	API를 통해 선택한 프로그래밍 언어로 작업할 수 있습니다.
-	설치 공간이 적고, 대부분의 종속성이 선택적이며, 필요할 때만 모듈을 요구할 수 있습니다. 
-	예시로 알아봅시다. 설명서는 모든 기능을 다룹니다.

## 설치

가장 쉬운 설치 방법은 via pip 과 PyPI입니다.

    pip install txtai

Python 3.6+ 이 지원됩니다.. 파이썬 [가상 환경](https://docs.python.org/3/library/venv.html) 을 사용하는 걸 추천드립니다.

[소스 설치](https://neuml.github.io/txtai/install/#install-from-source),
[환경별 필수 구성 요소](https://neuml.github.io/txtai/install/#environment-specific-prerequisites) 및
[선택적 종속성](https://neuml.github.io/txtai/install/#optional-dependencies) 에 대한 자세한 정보는 [설치 지침](https://neuml.github.io/txtai/install) 을 참조하십시오.

## 예시

examples 디렉토리에는 txtai의 개요를 제공하는 일련의 설명서(문서) 및 애플리케이션이 있습니다. 아래 섹션을 참조하십시오.

### 의미론적 검색

의미론적/유사성/벡터 검색 애플리케이션을 제작합니다. 

| 문서  | 설명  |       |
|:----------|:-------------|------:|
| [txtai 소개](https://github.com/neuml/txtai/blob/master/examples/01_Introducing_txtai.ipynb) | txtai에서 제공하는 기능 개요 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/01_Introducing_txtai.ipynb) |
| [Hugging Face 데이터 셋으로 임베딩 인덱스 구축](https://github.com/neuml/txtai/blob/master/examples/02_Build_an_Embeddings_index_with_Hugging_Face_Datasets.ipynb) | Hugging Face 데이터 셋 색인 및 검색 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/02_Build_an_Embeddings_index_with_Hugging_Face_Datasets.ipynb) |
| [데이터 소스로 임베딩 인덱스 구축](https://github.com/neuml/txtai/blob/master/examples/03_Build_an_Embeddings_index_from_a_data_source.ipynb)  | 단어 임베딩을 이용한 데이터 소스 색인 및 검색 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/03_Build_an_Embeddings_index_from_a_data_source.ipynb) |
| [Elasticsearch에 의미론적 검색 추가](https://github.com/neuml/txtai/blob/master/examples/04_Add_semantic_search_to_Elasticsearch.ipynb)  | 존재하는 검색 시스템에 의미론적 검색 추가 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/04_Add_semantic_search_to_Elasticsearch.ipynb) |
| [API 갤러리](https://github.com/neuml/txtai/blob/master/examples/08_API_Gallery.ipynb) | JavaScript, Java, Rust와 Go에서 txtai 사용 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/08_API_Gallery.ipynb) |
| [이미지를 이용한 유사성 검색](https://github.com/neuml/txtai/blob/master/examples/13_Similarity_search_with_images.ipynb) | 검색을 위해 포함된 이미지와 텍스트를 동일한 공간에 삽입 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/13_Similarity_search_with_images.ipynb) |
| [배포된 임베딩 클러스터](https://github.com/neuml/txtai/blob/master/examples/15_Distributed_embeddings_cluster.ipynb) | 여러 데이터 노드에 임베딩 인덱스 배포 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/15_Distributed_embeddings_cluster.ipynb) |

### 파이프라인 및 워크플로우

NLP 지원 데이터를 파이프라인과 워크플로우로 변환합니다.

| 문서  | 설명  |       |
|:----------|:-------------|------:|
| [txtai에 관한 질의응답](https://github.com/neuml/txtai/blob/master/examples/05_Extractive_QA_with_txtai.ipynb) | txtai에 관한 질의응답 소개 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/05_Extractive_QA_with_txtai.ipynb) |
| [Elasticsearch에 관한 질의응답](https://github.com/neuml/txtai/blob/master/examples/06_Extractive_QA_with_Elasticsearch.ipynb) | Elasticsearch에 관한 질의응답을 실행 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/06_Extractive_QA_with_Elasticsearch.ipynb) |
| [구조화된 데이터 구축에 관한 질의응답](https://github.com/neuml/txtai/blob/master/examples/20_Extractive_QA_to_build_structured_data.ipynb) | 질의응답을 통해 구조화된 데이터 셋을 구축 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/20_Extractive_QA_to_build_structured_data.ipynb) |
| [zero-shot 분류로 레이블을 적용](https://github.com/neuml/txtai/blob/master/examples/07_Apply_labels_with_zero_shot_classification.ipynb) | 라벨링, 분류 및 주제 모델링을 위해 zero-shot 학습을 사용 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/07_Apply_labels_with_zero_shot_classification.ipynb) |
| [추상적인 텍스트 요약 작성](https://github.com/neuml/txtai/blob/master/examples/09_Building_abstractive_text_summaries.ipynb) | 추상적인 텍스트 요약을 실행 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/09_Building_abstractive_text_summaries.ipynb) |
| [문서로부터 텍스트를 추출](https://github.com/neuml/txtai/blob/master/examples/10_Extract_text_from_documents.ipynb) | PDF, Office, HTML 등으로부터 텍스트를 추출 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/10_Extract_text_from_documents.ipynb) |
| [오디오를 텍스트로 변환](https://github.com/neuml/txtai/blob/master/examples/11_Transcribe_audio_to_text.ipynb) | 오디오 파일을 텍스트로 변환 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/11_Transcribe_audio_to_text.ipynb) |
| [언어 간에 텍스트를 번역](https://github.com/neuml/txtai/blob/master/examples/12_Translate_text_between_languages.ipynb) | 기계 번역 및 언어 감지를 간소화 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/12_Translate_text_between_languages.ipynb) |
| [파이프라인과 워크플로우를 실행](https://github.com/neuml/txtai/blob/master/examples/14_Run_pipeline_workflows.ipynb) | 효율적으로 데이터를 처리하기 위한 간단하면서도 강력한 구조 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/14_Run_pipeline_workflows.ipynb) |
| [구성 가능한 워크플로우로 테이블 형식 데이터 변환](https://github.com/neuml/txtai/blob/master/examples/22_Transform_tabular_data_with_composable_workflows.ipynb) | 워크플로우로 테이블 형식 데이터를 변환, 색인 생성및 검색 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/22_Transform_tabular_data_with_composable_workflows.ipynb) |

### 모델 훈련

NLP 모델을 훈련한다.

| 문서  | 설명  |       |
|:----------|:-------------|------:|
| [텍스트 레이블러를 훈련](https://github.com/neuml/txtai/blob/master/examples/16_Train_a_text_labeler.ipynb) | 텍스트 시퀀스 분류 모델을 구축 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/16_Train_a_text_labeler.ipynb) |
| [레이블을 제외하고 훈련](https://github.com/neuml/txtai/blob/master/examples/17_Train_without_labels.ipynb) | 새로운 모델을 훈련하기위해 zero-shot 분류기를 사용 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/17_Train_without_labels.ipynb) |
| [QA 모델을 훈련](https://github.com/neuml/txtai/blob/master/examples/19_Train_a_QA_model.ipynb) | 질의응답 모델 구축 및 미세 조정 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/19_Train_a_QA_model.ipynb) |
| [ONNX로 모델 내보내기 및 실행](https://github.com/neuml/txtai/blob/master/examples/18_Export_and_run_models_with_ONNX.ipynb) | ONNX로 모델을 내보내고, JavaScript, Java와 Rust에서 기본적으로 실행 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/18_Export_and_run_models_with_ONNX.ipynb) |
| [다른 머신 러닝 모델들을 내보내기 및 실행](https://github.com/neuml/txtai/blob/master/examples/21_Export_and_run_other_machine_learning_models.ipynb) | scikit-learn, PyTorch 등에서 모델을 내보내기 및 실행 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/21_Export_and_run_other_machine_learning_models.ipynb) |

### 응용 프로그램

txtai를 이용한 일련의 애플리케이션 예시입니다. [Hugging Face Spaces](https://hf.co/spaces)에서 호스팅된 버전에 대한 링크도 제공됩니다.

| 응용 프로그램  | 설명  |       |
|:-------------|:-------------|------:|
| [Basic similarity search](https://github.com/neuml/txtai/blob/master/examples/similarity.py) | 기본 유사성 검색 예시입니다. 원래 txtai 데모에서 사용되었습니다. |[🤗](https://hf.co/spaces/NeuML/similarity)|
| [Book search](https://github.com/neuml/txtai/blob/master/examples/books.py) | 도서 유사성 검색 애플리케이션입니다. 자연어 문장을 사용하여 책 설명 및 요청을 색인화합니다. |*오직 로컬에서만 실행*|
| [Image search](https://github.com/neuml/txtai/blob/master/examples/images.py) | 이미지 유사성 검색 애플리케이션입니다. 이미지 디렉토리를 인덱싱하고 검색을 실행하여 입력한 요청과 유사한 이미지를 식별합니다. |[🤗](https://hf.co/spaces/NeuML/imagesearch)|
| [Wiki search](https://github.com/neuml/txtai/blob/master/examples/wiki.py) | 위키백과를 검색하는 애플리케이션입니다. 위키백과 API를 요청하고 상위 결과를 요약합니다. |[🤗](https://hf.co/spaces/NeuML/wikisummary)|
| [Workflow builder](https://github.com/neuml/txtai/blob/master/examples/workflows.py) | txtai 워크플로우를 구축하고 실행합니다. 요약, 텍스트 추출, 사본(transcription), 번역 및 유사성 검색 파이프라인을 함께 연결하여 통합 워크플로우를 실행합니다. |[🤗](https://hf.co/spaces/NeuML/txtai)|

### 문서

파이프라인, 워크플로우, 인덱싱 및 API에 대한 구성 설정을 포함한 [txtai 전체 설명서](https://neuml.github.io/txtai)입니다.

### 추가적인 읽기

- [Introducing txtai, AI-powered semantic search built on Transformers](https://towardsdatascience.com/introducing-txtai-an-ai-powered-search-engine-built-on-transformers-37674be252ec)
- [Run machine-learning workflows to transform data and build AI-powered semantic search applications with txtai](https://towardsdatascience.com/run-machine-learning-workflows-to-transform-data-and-build-ai-powered-text-indices-with-txtai-43d769b566a7)
- [Semantic search on the cheap](https://towardsdatascience.com/semantic-search-on-the-cheap-55940c0fcdab)
- [Tutorial series on dev.to](https://dev.to/neuml/tutorial-series-on-txtai-ibg)

### 기여하기

txtai에 기여하고 싶은 사람은 [해당 가이드](https://github.com/neuml/.github/blob/master/CONTRIBUTING.md)를 참조하십시오.
