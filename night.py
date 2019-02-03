import cv2
import numpy as np


def resize(src):
    height = src.shape[0]
    width = src.shape[1]

    src2 = cv2.resize(src, (int(width*0.3), int(height*0.3)))
    return src2


def main(file_name):
    #cap = cv2.VideoCapture(0)
    base_image_path = file_name

    frame = resize(cv2.imread(base_image_path, 1))

    # フレームを取得
    #ret, frame = cap.read()

    # フレームをHSVに変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 取得する色の範囲を指定する
    lower_yellow = np.array([0, 220, 150])
    upper_yellow = np.array([80, 255, 255])

    # 指定した色に基づいたマスク画像の生成
    img_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 取得する色の範囲を指定する
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([0, 0, 255])

    # 指定した色に基づいたマスク画像の生成
    img_mask2 = cv2.inRange(hsv, lower_white, upper_white)

    img_mask += img_mask2

    # フレーム画像とマスク画像の共通の領域を抽出する。
    img_color = cv2.bitwise_and(frame, frame, mask=img_mask)

    #cv2.imshow("SHOW COLOR IMAGE", img_color)
    cv2.imwrite("gray_" + file_name, img_color)

    #cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #main('sample_falled1.jpeg')
    #main('falled.jpeg')
    main('base.jpeg')
    main('IMG_1282.jpeg')
