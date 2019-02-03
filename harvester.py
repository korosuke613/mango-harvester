import cv2
import numpy as np
from sklearn.cluster import KMeans


class Harvester:
    def __init__(self, target_path):
        self.dir_path = "./img/"
        self.gray = "gray_"
        self.base_image = "base.jpeg"
        self.target_image = target_path
        self.size = 0.3
        self.lower_yellow = np.array([0, 220, 150])
        self.upper_yellow = np.array([80, 255, 255])

    def run(self):
        self.change_gray(self.base_image)
        self.change_gray(self.target_image)

        self.matching()

    def resize(self, src):
        height = src.shape[0]
        width = src.shape[1]

        src2 = cv2.resize(src, (int(width * self.size), int(height * self.size)))
        return src2

    def change_gray(self, file_name):
        frame = self.resize(cv2.imread(self.dir_path + file_name, 1))
        # フレームをHSVに変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 取得する色の範囲を指定する
        # 指定した色に基づいたマスク画像の生成
        img_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        # 取得する色の範囲を指定する
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([0, 0, 255])

        # 指定した色に基づいたマスク画像の生成
        img_mask2 = cv2.inRange(hsv, lower_white, upper_white)

        img_mask += img_mask2
        # フレーム画像とマスク画像の共通の領域を抽出する。
        img_color = cv2.bitwise_and(frame, frame, mask=img_mask)
        cv2.imwrite(self.dir_path + "gray_" + file_name, img_color)

    def matching(self):
        # 画像をグレースケールで読み込み
        gray_base_src = cv2.imread(self.dir_path + self.gray + self.base_image, 0)
        gray_temp_src = cv2.imread(self.dir_path + self.gray + self.target_image, 0)

        # マッチング結果書き出し準備
        # 画像をBGRカラーで読み込み
        color_base_src = cv2.imread(self.dir_path + self.gray + self.base_image, 1)
        color_temp_src = cv2.imread(self.dir_path + self.gray + self.target_image, 1)

        # 特徴点の検出
        type_cv = cv2.AKAZE_create()
        kp_01, des_01 = type_cv.detectAndCompute(gray_base_src, None)
        kp_02, des_02 = type_cv.detectAndCompute(gray_temp_src, None)

        # マッチング処理
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des_01, des_02)
        matches = sorted(matches, key=lambda x: x.distance)

        result_img = self.resize(cv2.imread(self.dir_path + self.target_image, 1))
        points = []
        for m in matches[:10]:
            points.append(kp_02[m.trainIdx].pt)
        n = 2
        pred = KMeans(n_clusters=n).fit_predict(points)
        shift = 4
        cluster = [[0, 0, 0, [0, 0], [9999, 9999]],
                   [0, 0, 0, [0, 0], [9999, 9999]]]
        for i, p in enumerate(points):
            if pred[i] == 1:
                c = 0
            else:
                c = 255
            cluster[pred[i]][0] += 1
            cluster[pred[i]][1] += p[0]
            cluster[pred[i]][2] += p[1]
            if cluster[pred[i]][3][0] < p[0]:
                cluster[pred[i]][3][0] = p[0]
            elif cluster[pred[i]][4][0] > p[0]:
                cluster[pred[i]][4][0] = p[0]
            if cluster[pred[i]][3][1] < p[1]:
                cluster[pred[i]][3][1] = p[1]
            elif cluster[pred[i]][4][1] > p[1]:
                cluster[pred[i]][4][1] = p[1]

            cv2.circle(result_img,
                       (int(p[0]*pow(2.0, shift)), int(p[1]*pow(2.0, shift))),
                       100,
                       (0, 0, c),
                       10,
                       shift=shift)
        dst = [[],[]]
        for i in range(n):
            if i == 1:
                c = 0
            else:
                c = 255
            cluster[i][1] /= cluster[i][0]
            cluster[i][2] /= cluster[i][0]
            # cv2.circle(result_img,
            #            (int(cluster[i][1]*pow(2.0, shift)),
            #             int(cluster[i][2]*pow(2.0, shift))),
            #            100,
            #            (0, 0, c),
            #            cluster[i][0]*20,
            #            shift=shift)
            cv2.rectangle(result_img,
                          (int(cluster[i][3][0]*pow(2.0, shift)),
                           int(cluster[i][3][1]*pow(2.0, shift))),
                          (int(cluster[i][4][0]*pow(2.0, shift)),
                           int(cluster[i][4][1]*pow(2.0, shift))),
                          (0, 0, c),
                          shift=shift)
            dst[i] = result_img[
                  int(cluster[i][4][1]):int(cluster[i][3][1])+10,
                  int(cluster[i][4][0]):int(cluster[i][3][0])+10]
        print(dst[0].mean())
        print(dst[1].mean())
        cv2.imwrite(self.dir_path + "result.jpeg", result_img)
        cv2.imshow("result_image_src", result_img)
        cv2.imshow("a", dst[0])
        cv2.imshow("b", dst[1])
        match_image_src = cv2.drawMatches(color_base_src, kp_01, color_temp_src, kp_02, matches[:10], None, flags=2)
        cv2.imwrite(self.dir_path + "match_image_src.jpeg", match_image_src)
        cv2.waitKey(0)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    harvester = Harvester("IMG_1283.jpeg")
    harvester.run()
