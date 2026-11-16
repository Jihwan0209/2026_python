## lightGBM 튜닝 / 서빙 예제 필요한 데이터셋
import pandas as pd
url = 'https://raw.githubusercontent.com/rashida048/Datasets/master/USA_cars_datasets.csv'
df = pd.read_csv(url)
df = df[df['price'] > 0]
num_cols = ['year', 'mileage']                                      # 수치형
cat_cols = ['brand', 'model', 'title_status', 'color', 'state']     # 범주형
# *만약에 모델의 성능 향상 하기 위한 도메인 지식의 (계산식)특성 추가 좋다.*

# 범주형 자료들을 원핫인코딩이 아닌 판다스'카테고리' 타입 변경 # lightGBM 내장 원핫인코딩 사용 목적
for col in cat_cols: 
    df[col] = df[col].astype('category')
X = df[num_cols + cat_cols]
Y = df['price']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 1) 튜닝 -> optuna
# 1. optuna 설치 : $env:PYYAML_FORCE_LIBYAML = "0" , pip install optuna,shap
# 2) optuna 로그 숨기기/경고만 출력
import optuna
from lightgbm import LGBMClassifier 
optuna.logging.set_verbosity( optuna.logging.WARNING )
# 4) 목적함수 만들기 : 각 탐색 시도마다 교차 검증 수행 후 평균 성능 반환 , *콜백함수( trial )
def objective( trial ):
    # 4-1 교차검증 KFold
    from sklearn.model_selection import KFold
    cv = KFold( n_splits=5 , shuffle=True , random_state=42 )
    cv_r2_scores= [ ]           # 각 폴더별 검증 점수 저장
    best_iterations = [ ]       # 각 폴더별 최적 트리 번호 저장
    for train_idx, val_idx in cv.split( X_train , Y_train) :
        # 훈련 데이터 세트 내부에서 훈련용/검증용 분할 수행 , K=5 , 5개 폴드 , 각각 한번씩 검증용
        X_tr , X_val = X_train.iloc[ train_idx ] , X_train.iloc[val_idx]
        Y_tr , Y_val = Y_train.iloc[ train_idx ] , Y_train.iloc[val_idx]
        # k폴드에서 분할한 훈련용 모델 만들기
        model = LGBMClassifier(
            n_estimators=1000,
            # trial.suggest_XXX( '파라미터명' , 시작값, 끝값 )
            learning_rate=trial.suggest_float('learning_rate' , 0.01 , 0.1 ),
            num_leaves=trial.suggest_int( 'num_leaves' , 10 , 30),
            max_depth=trial.suggest_int('max_depth' , 3, 10 ),
            min_child_samples=n

            
        )
    pass
# 3) optuna.create_study( direction='maximize/minimize')
# 현재 실습은 회귀모델이며 r2결정계수 평가지표 사용할 예정
sampler = optuna.samplers.TPESampler( seed=42 )
study = optuna.create_study( direction='maximize' , sampler= sampler )
# study.optimize( 목적함수 , n_trials=반복수 )
study.optimize( objective , n_trials=20 ) # -> 반복 수 만큼 목적함수 실행