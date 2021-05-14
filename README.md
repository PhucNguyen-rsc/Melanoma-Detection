# Melanoma-Detection
## 1. Tóm tắt đề tài
* Đưa ra một sản phẩm nhận vào hình ảnh nốt ruồi, xem xét và dự đoán nốt ruồi có ác tính không
* "Skin cancer is a common disease that affect a big amount of peoples. Some facts about skin cancer:

  + Every year there are more new cases of skin cancer than the combined incidence of cancers of the breast, prostate, lung and colon.
  + An estimated 87,110 new cases of invasive melanoma will be diagnosed in the U.S. in 2017.
  + The estimated 5-year survival rate for patients whose melanoma is detected early is about 98 percent in the U.S. The survival rate falls to 62 percent when the disease reaches the lymph nodes, and 18 percent when the disease metastasizes to distant organs.
  + Early detection is critical!"
## 2. Ý tưởng
* Ý tưởng của project là xây dựng mô hình CNN có khả năng dự đoán ảnh nốt ruồi da đưa vào là lành tính (benign) hay ác tính (malign)
### 2.1 Mô hình TEFPA:
* Task: input là hình ảnh nốt ruồi, output là dự đoán có bị ung thư da hay không
* Experiment: những hình ảnh nốt ruồi cùng với dự đoán 
* Function space: 
* Performance: AUC, ROC, hàm loss là focal crossentropy
* Algorithm: 
### 2.2. Data
* Smaple:
  * benign mole
![image](https://user-images.githubusercontent.com/84164707/118297028-d487cc00-b507-11eb-903b-f185bf93d29d.png)
  * malign mole
![image](https://user-images.githubusercontent.com/84164707/118296814-92f72100-b507-11eb-8578-593fed63c3ef.png)
### 2.3. Lý do chọn mô hình CNN
* Những mô hình sử sụng:
  + VCG16
  + EfficientNetB0
  + InceptionV3
  + ResNet50
  + AlexNet (modified)
* Lý do: dựa vào một số bài báo nghiên cứu để có được kết quả cao
### 2.4. Tiền xử lý dự liệu
* Do dataset bị imbalanced cao nên tụi em đã gom toàn bộ ảnh malign lại
* Dùng ImageDataGenerator thành 256x256x3
* Image resizing: Transform images to 224x224x3
## 3. Pre-trained model
### 3.1. Model 1: VGG16
* Model_name: VCG16 + Dense layer
* Batch_size=32, epochs =1000, verbose=1, earlystopping
* AUC = …
### 3.2. Model 2: EfficientNetB0
* Model_name: EfficientNetB0 + Dense layer
* Epochs =1000, verbose=2, earlystopping
* AUC = …
### 3.3. Model 3: InceptionV3
* Model_name: InceptionV3 + Dense layer
* Epochs =1000, verbose=2, earlystopping
* AUC =...
### 3.4. Model 4: ResNet50
* Model_name: ResNet50 + Dense layer
* Epochs =1000, verbose=2, earlystopping
* AUC =...

### 3.5. Model 5: AlexNet (modified)
* Model_name: ResNet50 + Dense layer
* Epochs =1000, verbose=2, earlystopping
* AUC =...
![image](https://user-images.githubusercontent.com/84164707/118298285-7c51c980-b509-11eb-8e47-dc8860560006.png)

## 4. Tools to use
* Tensorflow
* keras
* Python
* matplotlib
* scikit-learn
* streamlit

![image](https://user-images.githubusercontent.com/84164707/118298422-a4412d00-b509-11eb-8abd-4f0441a00c88.png)
![image](https://user-images.githubusercontent.com/84164707/118298436-a86d4a80-b509-11eb-9b66-792f926e37bd.png)

## 5. Kết quả
## 6. Hướng phát triển:
* Tìm cách nâng cao tiềm năng dò soát độ ác tính của nốt ruồi.
* Xây dựng web bắt mắt, dễ sử dụng cũng như tiếp cận nhiều đối tượng khác nhau.
* Gia tăng tốc độ đồng thời bảo đảm độ chính xác của chương trình.
* Nghiên cứu thực hiện nhận biết nhiều nôt ruồi cùng lúc.



