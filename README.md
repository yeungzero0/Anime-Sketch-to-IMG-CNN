# Anime-Sketch-to-IMG-CNN

一個使用 **Convolutional Neural Network (CNN)** 將動漫素描（Sketch）轉換成彩色圖像的個人學習專案。

---

## ⚠️ 免責聲明 (Disclaimer)

**本專案純粹用於學習與個人娛樂用途**。

- 此專案是 **POLYU CS 學生** 的學習練習，主要目的是練習 CNN、影像轉換（Image-to-Image Translation）等深度學習技術。
- **嚴禁商業使用**：本專案**不得**用於任何商業活動、盈利目的、產品開發或生產環境。
- **僅供參考與教育**：所有產出的圖像僅供個人學習、測試與玩樂之用。
- **無任何保證**：作者不對使用本專案所產生的任何輸出、結果、版權問題或後果承擔任何責任。
- 生成的圖像可能包含訓練資料的風格偏好，使用者須自行確保不侵犯任何第三方版權（例如動漫角色、美術作品等）。
- 如有任何疑問，請勿用於正式用途。

> **總之：這只是我用來練習 CNN 的練習專案，不是專業工具，也不是商業解決方案。**

---

## 專案特色

- 使用 CNN 架構實現 Sketch → Colored Anime Image
- 支援 PyTorch 訓練與推論
- 簡單易懂的程式碼，適合初學者參考
- 包含資料前處理、模型訓練、推論範例
---

## Result

---

## 如何使用

Please edit the config.py, if you are the first play at this program, 
Ensure LOAD_MODEL is False, it look like (line 45):     LOAD_MODEL = False
After program created the "Disc.pth.tar" & "Gen.pth.tar", ensure LOAD_MODEL is Ture.

For creating your own dataset, save the image at /dataset/newIMG/original
Then run "part1_imageResizeTo512png.py" & "part2_imageToSketch.py"
*dataset should at less 500 more IMG, if you want it be more specific way, you should just create it specifically)
*Just like you want to recolor who anime character you want, you should just download which character as the original dataset.
*Then the sketch will recolored to your ref.

dataset - Ensure dataset should 60%/20%/20% (train/val/test)
real:copy with the resized original character image (part1_imageResizeTo512png.py, copy from /dataset/newIMG/to512size)
sketch:copy with the sketch image (part2_imageToSketch.py, copy from /dataset/newIMG/toSketch)
ref:N/A

When you want to play the Sketch to IMG, run "main_runThisForTesting(SketchToImage).py"
