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
* Algorithm: áp dụng và train lại toàn bộ các pretrained models như AlexNet, VGG16, EfficientNetB0, InceptionV3, ResNet50. Đồng thời nhóm ensemble các models này lại bằng phương pháp weighted average ensemble, và áp dụng Deep Stack để tìm ra weight cụ thể cho từng model
### 2.2. Data
* Sample:
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
* Lý do: dựa vào một số bài báo nghiên cứu để có được kết quả cao, đồng thời đây cũng là những models thường được đưa ra nghiên cứu
### 2.4. Tiền xử lý dữ liệu
* Do dataset bị imbalanced cao nên tụi em đã gom toàn bộ ảnh malignant lại
* Nhóm chúng em áp dụng xáo trộn file csv lại , sau đó chọn 4000 bức ảnh benign đàu tiên + 500 bức ảnh malignant làm tập train ; 500 bức ảnh benign tiếp theo và các ảnh malignant còn lại làm tập validation
* Nhóm chúng em dùng các hàm flow_from_dataframe cùng với Imaga Data Generator để dán nhãn cho ảnh (có được từ file csv ở trên)
* Dùng ImageDataGenerator resize lại ảnh thành 256x256x3, đồng thời áp dụng các phương pháp data augmentation khác như shearing, flipping,...
## 3. Pre-trained model
* Tóm tắt chung: nhìn chung chúng em đều xây dựng các pretrained models với cấu trúc : Base + Global Average Pooling + Dense layer (là lớp Prediction, output chỉ có 1 node). Đồng thời chúng em còn áp dụng những kĩ thuật sau trong training để nâng cao hiệu suất cũng như độ chính xác của các model:
  + Dùng class weights (computed từ sklearn) để áp dụng cho từng class trong traning {0:0.5,
                                                                                      1:4.0)
  + Dùng Data Augmentation trong bước chuẩn bị để nâng cao hiệu suất models
  + Dùng Sigmoid Focal Crossentropy là hàm loss (có trong tensorflow-addons). Hàm loss này đã được chứng minh là rất hiệu quả trong việc xử lý những bộ file dataset ảnh có tính imbalanced cao, cơ bản vì hàm này ép các models phải học những đặc trưng khác trong ảnh, không được chỉ tập trung vào 1 hay 1 vài features đơn lẻ
  + Dùng thang đo ROC-AUC làm metrics, vì đây là thang đo mang tính khách quan và chính xác hơn đối với những file dataset có đô imbalanced cao
  + Ngoài ra, trong lúc training, chúng em còn áp dụng workers, use_multiprocessing để tăng tốc độ trong việc training lên
 ![image](https://user-images.githubusercontent.com/84164707/118348362-f028bc00-b573-11eb-9bf9-7c0e7b02d047.png)

### 3.1. Model 1: VGG16
* Model_name: VCG16 + Gloval Average Pooling 2D+  Dense layer
* AUC-ROC score trên tập validation = 0.8829754330151997
![image](https://user-images.githubusercontent.com/84164707/118350526-6aac0880-b581-11eb-9c14-c56e3bc45c64.png)

### 3.2. Model 2: EfficientNetB0
* Model_name: EfficientNetB0 + Gloval Average Pooling 2D+  Dense layer
* AUC-ROC score trên tập validation = 0.6839872746553552
![image](https://user-images.githubusercontent.com/84164707/118350532-7a2b5180-b581-11eb-81c1-2270eaa66996.png)

### 3.3. Model 3: InceptionV3
* Model_name: InceptionV3 + Gloval Average Pooling 2D+  Dense layer
* AUC-ROC score trên tập validation = 0.8140906680805939
![image](https://user-images.githubusercontent.com/84164707/118350544-8f07e500-b581-11eb-930b-1c1aa2b0e1d2.png)

### 3.4. Model 4: ResNet50
* Model_name: ResNet50 + Gloval Average Pooling 2D+  Dense layer
* AUC-ROC score trên tập validation = 0.7783669141039237
![image](https://user-images.githubusercontent.com/84164707/118350581-c70f2800-b581-11eb-876c-cea9dcbc93ff.png)

### 3.5. Model 5: AlexNet (modified)
* Model_name: ResNet50 ++ Gloval Average Pooling 2D+  Dense layer
* AUC-ROC score trên tập validation = 0.727487628137151
![image](https://user-images.githubusercontent.com/84164707/118350612-e1e19c80-b581-11eb-8a8c-ec35a088ef39.png)


### Tìm weights cụ thể cho từng model:
- Chúng em đã sử dụng library Deep Stack, cụ thể là kết hợp các model với hàm  DirichletEnsemble() để tìm ra weights cụ thể cho từng model 
## 4. Tools to use
* Tensorflow và Keras
* Tensorflow-addons
* Python
* matplotlib
* scikit-learn
* streamlit

![image](https://user-images.githubusercontent.com/84164707/118298422-a4412d00-b509-11eb-8abd-4f0441a00c88.png)
![image](https://user-images.githubusercontent.com/84164707/118298436-a86d4a80-b509-11eb-9b66-792f926e37bd.png)

## 5. Kết quả

## 6. Deploy trên web app bằng Streamlit:

## 7. Hướng phát triển:
* Áp dụng các phương pháp Image Segmentation + Object Detection để nâng cao khả năng xác định được các nốt ruồi (moles) trong ảnh
* Phân tích các thang đo ROC-AUC curve và confusion matrix để xác định được threshold nên dùng trong việc xác định mole này là bening hay malignant
* Xây dựng web bắt mắt, dễ sử dụng cũng như tiếp cận nhiều đối tượng khác nhau.
* Gia tăng tốc độ đồng thời bảo đảm độ chính xác của chương trình.
* Nghiên cứu thực hiện nhận biết nhiều nôt ruồi cùng lúc.



