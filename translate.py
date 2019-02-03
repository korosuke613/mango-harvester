# -*- coding: utf-8 -*-

import cv2


def resize(src):
    height = src.shape[0]
    width = src.shape[1]

    src2 = cv2.resize(src, (int(width*0.3), int(height*0.3)))
    return src2


if __name__ == '__main__':

    # 対象画像を指定
    base_image_path = 'base.jpeg'
    temp_image_path = 'IMG_1282.jpeg'

    # 画像をグレースケールで読み込み
    gray_base_src = resize(cv2.imread("gray_"+base_image_path, 0))
    gray_temp_src = resize(cv2.imread("gray_"+temp_image_path, 0))

    # マッチング結果書き出し準備
    # 画像をBGRカラーで読み込み
    color_base_src = resize(cv2.imread("gray_" + base_image_path, 1))
    color_temp_src = resize(cv2.imread("gray_" + temp_image_path, 1))


    # 特徴点の検出
    type = cv2.AKAZE_create()
    kp_01, des_01 = type.detectAndCompute(gray_base_src, None)
    kp_02, des_02 = type.detectAndCompute(gray_temp_src, None)

    # マッチング処理
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des_01, des_02)
    matches = sorted(matches, key = lambda x:x.distance)
    mutch_image_src = cv2.drawMatches(color_base_src, kp_01, color_temp_src, kp_02, matches[:10], None, flags=2)

    # 結果の表示
    #cv2.imshow("mutch_image_src", mutch_image_src)
    cv2.imwrite("mutch_image_src.jpeg", mutch_image_src)

    cv2.waitKey(0)
    cv2.destroyAllWindows()