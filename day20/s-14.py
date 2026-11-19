# '텍스트 마이닝' 실습에 필요한 데이터셋
import pandas as pd
# 리뷰 데이터셋
url = "https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt"
df = pd.read_table(url, names=['rating', 'review']) # raing:평점 , review:리뷰
df = df[df['rating'] != 3] # 평점3 제외
df['label'] = (df['rating'] >= 4).astype(int) # 평점4 이상은 긍정 , 평점2 이하 부정
# 전체데이터셋에서 특정 n개 샘플링 
df_sample = df.sample(n=5000, random_state=42).reset_index(drop=True)





