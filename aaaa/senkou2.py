import cv2

cascade_path = r'C:\Users\lamp\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml'


# 使用ファイルと入出力ディレクトリ
image_file = "human2.jpg"
image_path = "images/" + image_file
output_path = "images/aa/" + image_file

#ファイル読み込み
image = cv2.imread(image_path)

#グレースケール変換
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)

facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

#print(facerect)
color = (255, 255, 255) #白

# 検出した場合
if len(facerect) > 0:

    #検出した顔を囲む矩形の作成
    for rect in facerect:
      image = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]

    #認識結果の保存
    cv2.imwrite(output_path, image)

    print("true")

else :
    print("false")
