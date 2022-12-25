from google.colab import files
from bs4 import BeautifulSoup
from google.colab.patches import cv2_imshow

import cv2
import torch
import requests
import numpy as np

#画像をアップロードする
uploaded = files.upload()

key = list(uploaded.keys())[0]
input_img = uploaded[key]

#バイト列→numpy配列変換，デコード，BGR→RGB色空間変換
input_img = np.frombuffer(input_img, dtype=np.uint8)
input_img = cv2.imdecode(input_img, cv2.IMREAD_COLOR)
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

#画像を表示
cv2_imshow(input_img)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

#入力画像を推論
results = model(input_img)

#結果の出力
results.show()

#結果を文字列形式で出力する
results_str = results.__str__()

#文字列で表示された結果の改行以降を削除する
first_line = results_str.split('\n')[0]
print(first_line)

#文字列をスペースで分割してリストに変換する
results_list = first_line.split(" ")

#リスト内から物体の数を抽出する
num_of_objects = int(results_list[3])

#結果から物体が存在するかどうかを判定する
if results_list[3] == '(no':
  #物体が存在しない場合は、物体の数を0とする
  num_of_objects = 0
else:
  #物体が存在する場合は、物体の数を数値として取得する
  num_of_objects = int(results_list[3])

#リスト内から物体のクラス名だけを抽出する
class_name_list = []
for i in range(num_of_objects):
  #インデックスが負の値になるような式を書かないようにする
  #物体の数だけ繰り返し、要素を追加していく
  if i + 4 < len(results_list):
    class_name_list.append(results_list[i+4])

#物体のクラス名を表示する
print(class_name_list)

#Cookpad 検索にリクエストを送信し、レシピの一覧を取得する
#画像認識で推定された物体の名前をそのまま使用して検索する
response = requests.get(f"https://cookpad.com/search/{'+'.join(class_name_list)}")

#取得したレスポンスからレシピの一覧を抽出する
#レシピの一覧を簡単に抽出するために BeautifulSoup を使用
soup = BeautifulSoup(response.text, 'html.parser')

recipe_list = []
for recipe_tag in soup.find_all('a', class_='recipe-title'):
    recipe_title = recipe_tag.text
    recipe_url = recipe_tag['href']
    recipe_list.append({'title': recipe_title, 'url': recipe_url})

#レシピの一覧を表示する
for recipe in recipe_list:
    print(f"タイトル: {recipe['title']}")
    print(f"URL: https://cookpad.com{recipe['url']}")
