# local-sources — 서버 로컬 모델·데이터셋 폴더

이 폴더는 `docker compose`가 `api`와 `worker` 컨테이너의 `/sources/local`에 **읽기 전용**으로 연결하는 로컬 source root입니다. Hugging Face Hub에 올리지 않은 모델이나 데이터셋을 분석할 때 여기에 둡니다.

이 README를 제외한 폴더 내용은 `.gitignore` 대상이라 커밋되지 않습니다. 모델 가중치나 데이터셋 원문을 저장소에 올리지 마세요.

## 사용 방법

1. 모델이나 데이터셋 폴더를 이 디렉터리 아래에 복사하거나 둡니다.

   ```text
   local-sources/
   ├── models/my-model/           # config.json, tokenizer 파일, *.safetensors 등
   └── datasets/my-dataset/       # *.jsonl, *.json, *.parquet, *.csv 등
   ```

2. 웹 화면의 Model 또는 Dataset 입력란에 `local:local/<이 폴더 기준 상대 경로>` 형식으로 입력합니다.

   ```text
   local:local/models/my-model
   local:local/datasets/my-dataset
   ```

   앞의 `local`은 root 이름입니다. 등록된 root 목록은 `GET /api/v1/local-roots`에서 확인할 수 있습니다.

## 알아둘 점

- **경로는 서버 기준입니다.** 브라우저를 실행한 PC가 아니라 Docker가 실행되는 호스트의 이 폴더만 읽습니다 (plan.md §6.1). 원격 Docker 호스트를 쓴다면 그 호스트의 폴더가 연결됩니다.
- **읽기 전용입니다.** 컨테이너는 이 폴더에 쓰거나 파일을 바꾸지 않습니다. 분석 결과와 캐시는 `vfdata` 볼륨(`/data`)에 저장됩니다.
- **가중치를 실행하지 않습니다.** 분석은 config, tokenizer·chat template, safetensors header, 데이터셋 파일만 읽습니다. 모델을 실행하거나 학습하지 않습니다.
- **권한.** 컨테이너는 uid `10001`(non-root)로 실행됩니다. Linux 호스트에서는 파일이 다른 사용자에게도 읽기 가능해야 합니다(예: `chmod -R a+rX local-sources`).
- **심볼릭 링크.** 이 폴더 밖을 가리키는 링크는 컨테이너 안에서 따라갈 수 없고, root 밖으로 나가는 경로는 거부됩니다.
- **다른 폴더 사용.** 이미 모델을 모아 둔 폴더가 있다면 복사하지 말고 `.env`에 `VRAMFORGE_LOCAL_SOURCES_DIR=/절대/경로`를 지정한 뒤 다시 기동합니다. 참조 형식은 `local:local/...` 그대로입니다.

자세한 배포 설정은 [`docs/deployment.md`](../docs/deployment.md)를 참고하세요.
