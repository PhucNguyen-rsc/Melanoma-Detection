# Melanoma-Detection
## 1. Tóm tắt chung 
### 1.1 Tóm tắt đề tài
* Đưa ra một sản phẩm nhận vào hình ảnh nốt ruồi, xem xét và dự đoán nốt ruồi có ác tính không
* "Skin cancer is a common disease that affect a big amount of peoples. Some facts about skin cancer:
  + Every year there are more new cases of skin cancer than the combined incidence of cancers of the breast, prostate, lung and colon.
  + An estimated 87,110 new cases of invasive melanoma will be diagnosed in the U.S. in 2017.
  + The estimated 5-year survival rate for patients whose melanoma is detected early is about 98 percent in the U.S. The survival rate falls to 62 percent when the disease reaches the lymph nodes, and 18 percent when the disease metastasizes to distant organs.
  + Early detection is critical!"
### 1.2 Link các sản phẩm có liên quan:
  * Link các model đã được trained xong dựa trên tập dữ liệu ở dưới: https://drive.google.com/drive/folders/10aTvRsL1wrWjqRwsug9JHSLFOfnnqE-U?usp=sharing
  * Link của bài Presentation: https://drive.google.com/drive/folders/1HqhK64tP170rSeZNuFCr77HMard6Y4GJ?usp=sharing
  * Link dataset sử dụng trên Kaggle: https://www.kaggle.com/c/siim-isic-melanoma-classification/data
  
### 1.3 Có gì thay đổi so với đợt predict trước:
- Code về phần preprocessing input đã được sửa (chỉ đang mỗi VGG16 và AlexNet vì những cái còn lại đều tương tự như vậy). Tuy nhiên toàn bộ kết quả ở dưới là cho đợt trained trước chưa bao gồm những chỉnh sửa mới này
- Đã có kết quả cho từng weight của từng model, đồng thời có cả kết quả đánh giá model trên tập test dataset của cuộc thi SIIM-ISIC Melanoma Classification trên Kaggle
- Đã deploy web bằng streamlit thành công, chi tiết xem ở phía dưới

## 2. Ý tưởng
* Ý tưởng của project là áp dụng transfer learning và thuật toán ensemble blending trong việc xây dựng mô hình có khả năng dự đoán ảnh nốt ruồi da đưa vào là lành tính (benign) hay ác tính (malignant)
### 2.1 Mô hình TEFPA:
* Task: input là hình ảnh nốt ruồi, output là dự đoán có bị ung thư da hay không
* Experiment: những hình ảnh nốt ruồi cùng với dự đoán 
* Function space: các models AlexNet, VGG16, EfficientNetB0, InceptionV3, ResNet50.
* Performance: AUC, ROC, hàm loss là focal crossentropy
* Algorithm: áp dụng và train lại toàn bộ các pretrained models như AlexNet, VGG16, EfficientNetB0, InceptionV3, ResNet50. Đồng thời nhóm ensemble các models này lại bằng phương pháp weighted average ensemble, và áp dụng Deep Stack để tìm ra weight cụ thể cho từng model
### 2.2. Data
* Sample:
  * benign mole
  
![image](https://user-images.githubusercontent.com/84164707/118297028-d487cc00-b507-11eb-903b-f185bf93d29d.png)
  * malignant mole
  
![image](https://user-images.githubusercontent.com/84164707/118296814-92f72100-b507-11eb-8578-593fed63c3ef.png)

- Bộ dữ liệu bọn em sử dụng đến từ cuộc thi nổi tiếng SIIM-ISIC Melanoma Classification trên Kaggle (có thể download qua API sau: kaggle competitions download -c siim-isic-melanoma-classification)
- Về cơ bản, bộ dữ liệu được cho có thể chia thành 4 phần dữ liệu chính: file ảnh dưới dạng DICOM format, file ảnh dưới dạng JPEG, file ảnh và metadata dưới dạng TFRecord, và file metadata và nhãn (labels) dưới dạng file csv. Ở đây, vì sự thuật tiện nên nhóm đã quyết định chọn file ảnh jpeg và file csv (để dán nhãn) trong quá trình train các models.
- Bài toán được đặt ra về cơ bản có 2 loại nhãn : 0 (là benign) và 1 (là malignant). Output của model nên là 1 probability chạy từ 0 đến 1
- Về cơ bản, 0 tức là những hình ảnh tế bào da thông thường và 1 tức là ảnh tế bào da bị ung thư hắc tố. Tuy nhiên, khi lúc sau cả nhóm inspect lại thì thấy có **nhiều loại** benign (tức là sub-predictions). Điều này đã đồng nghĩa với việc là ngay cả các ảnh benign cũng có rất nhiều loại/patterns khác nhau, và nhóm đã thiếu sót khi chia dữ liệu ảnh cho việc training và validation không tính đến khả năng này (xem thêm ở lúc sau

![image](https://user-images.githubusercontent.com/68393604/118479274-059c1280-b73b-11eb-985a-8328ce700e95.png)

![image](https://user-images.githubusercontent.com/68393604/118479419-311efd00-b73b-11eb-8340-773ae8d072d2.png)

- Vì đây cốt lõi là file phát hiện bệnh nên đồng thời tỉ lệ imbalanced giữa 2 loại label lớn rất lớn. 

![image](https://user-images.githubusercontent.com/68393604/118479771-a1c61980-b73b-11eb-8311-0baaf8936805.png)

### 2.3. Lý do chọn mô hình CNN
* Những mô hình sử sụng:
  + VGG16
  + EfficientNetB0
  + InceptionV3
  + ResNet50
  + AlexNet (nhóm tự tái tạo lại dựa trện 1 bài báo trên medium, vì AlexNet là 1 trong những model thời đầu nên không thể import qua Keras như thông lệ)
* Lý do: dựa vào một số bài báo nghiên cứu để có được kết quả cao, đồng thời đây cũng là những models thường được đưa ra nghiên cứu nhiều trong những bài báo liên quan đến việc dùng Transfer Learning/ Fine Tuning trong việc chẩn đoán và phát hiện tế bào ung thư hắc tố da (melanoma)
### 2.4. Tiền xử lý dữ liệu
* Do dataset bị imbalanced cao nên tụi em đã gom toàn bộ ảnh malignant lại
* Nhóm chúng em áp dụng xáo trộn file csv lại , sau đó chọn 4000 bức ảnh benign đàu tiên + 500 bức ảnh malignant làm tập train ; 500 bức ảnh benign tiếp theo và các ảnh malignant còn lại làm tập validation

![image](https://user-images.githubusercontent.com/68393604/118480020-e9e53c00-b73b-11eb-8e44-84c6548c93cd.png)

File train sau xử lý

* Nhóm chúng em dùng các hàm flow_from_dataframe cùng với Imaga Data Generator để dán nhãn cho ảnh (có được từ file csv ở trên)
* Dùng ImageDataGenerator resize lại ảnh thành 256x256x3, đồng thời áp dụng các phương pháp data augmentation khác như shearing, flipping,...

![image](https://user-images.githubusercontent.com/68393604/118490050-ba3c3100-b747-11eb-92de-4dd801873002.png)

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
* AUC score trên tập validation = 0.8829754330151997

![image](https://user-images.githubusercontent.com/84164707/118350526-6aac0880-b581-11eb-9c14-c56e3bc45c64.png)

### 3.2. Model 2: EfficientNetB0
* Model_name: EfficientNetB0 + Gloval Average Pooling 2D+  Dense layer
* AUC score trên tập validation = 0.6839872746553552

![image](https://user-images.githubusercontent.com/84164707/118350532-7a2b5180-b581-11eb-81c1-2270eaa66996.png)

### 3.3. Model 3: InceptionV3
* Model_name: InceptionV3 + Gloval Average Pooling 2D+  Dense layer
* AUC score trên tập validation = 0.8140906680805939

![image](https://user-images.githubusercontent.com/84164707/118350544-8f07e500-b581-11eb-930b-1c1aa2b0e1d2.png)

### 3.4. Model 4: ResNet50
* Model_name: ResNet50 + Gloval Average Pooling 2D+  Dense layer
* AUC score trên tập validation = 0.7783669141039237

![image](https://user-images.githubusercontent.com/84164707/118350581-c70f2800-b581-11eb-876c-cea9dcbc93ff.png)

### 3.5. Model 5: AlexNet (modified)
* Model_name: ResNet50 ++ Gloval Average Pooling 2D+  Dense layer
* AUC score trên tập validation = 0.727487628137151

![image](https://user-images.githubusercontent.com/84164707/118350612-e1e19c80-b581-11eb-8a8c-ec35a088ef39.png)


### Tìm weights cụ thể cho từng model:
- Với mục tiêu áp dụng kĩ thuật Ensemble Blending, chúng em đã sử dụng library Deep Stack, cụ thể là kết hợp các model với hàm  DirichletEnsemble() để tìm ra weights cụ thể cho từng model là: 0.8938 cho model 1 (VGG16), 0.0403 cho model 2 (EfficientNetB0), 0.0126 cho model 3 (InceptionV3), 0.0029 cho model 4 (ResNet50) và 0.0504 cho model 5 (AlexNet)

(![image](https://user-images.githubusercontent.com/68393604/118476868-3fb7e500-b738-11eb-983d-983ede716a3f.png)

--> chú ý: đây là những thông số thu được trước khi bắt đầu thêm vào lớp preprocessing input cho riêng từng model (thay vì dựa vào ImageDataGenerator) và thêm vào các lớp Dense layers sau lớp Global Average Pooling cho từng model (để giảm các feature maps dần dần xuống cho model có performance tốt hơn, lấy cảm hứng từ dự án của Thịnh và Nga).
## 4. Tools to use
* Tensorflow và Keras
* Tensorflow-addons
* Python
* matplotlib
* scikit-learn
* streamlit

![image](https://user-images.githubusercontent.com/84164707/118298422-a4412d00-b509-11eb-8abd-4f0441a00c88.png)
![image](https://user-images.githubusercontent.com/84164707/118298436-a86d4a80-b509-11eb-9b66-792f926e37bd.png)

## 5. Kết quả:
- Sau khi submit để tập test trên Kaggle, chúng em thấy được điểm của chúng em trên tập test là 0.8070, không phải là 1 số điểm quá cao. Nguyên ngân có lẽ là do bọn em lần này chưa thực sự tối ưu hóa các model và dữ liệu (chọn lọc model chưa tốt, như có thể thấy là model VGG16 chiếm tỉ trọng rất lớn trong việc quyết định điểm ROC-AUC score final so với các model khác; chưa thêm các lớp Dense layers phía sau lớp Global Average Pooling; chưa có lớp Preprocessing input cụ thể cho từng model; đồng thời chưa tận dụng được file metadata có sẵn). Đồng thời nhóm cũng chưa train dữ liệu trên toàn bộ tập dataset nên tỉ lệ bị biased là rất cao.

## 6. Deploy trên web app bằng Streamlit:
- Bọn em quyết định dùng streamlit để tạo 1 web app cơ bản classify ảnh bọn em đưa vào, chi tiết về code ở trên file Deploy. Ở đây mặc dù nhóm dùng cả 5 model để ensemble kết hợp lại để cho kết quả cuối cùng, thế nhưng nhóm chúng em nghĩ rằng chỉ cần áp dụng VGG16 vào là đủ (vì VGG16 có tỉ lệ đúng rất cao và là thành phần quan trọng chiếm tỉ lệ cao nhất khi ensemble lại). Điều này có thể giúp chúng ta đảm bảo tốc độ chạy của mô hình thuật toán mà vẫn đảm bảo được khả năng dự đoán của model.
--> xem demo trên Youtube: https://youtu.be/vAQ9V3F0VTA

- Hướng dẫn sử dụng:
  * Vì Streamlit chỉ demo được trên local host nên phải tải code về rồi chạy lại trong máy
  * Warning: nhớ cài đặt pydot, graphviz, pydotplus, streamlit đầy đủ vào enviroment để code chạy được. Đồng thời nhớ chèn đường link model trained của mình vô để code chạy được
  * Sau khi đã chạy được web lên như google chrome rồi, thao tác như trong video phía trên : nhấn nút Browse File để chọn 1 file ảnh lên (nhớ 1 lần chỉ được chọn 1 tấm thôi, và tấm ảnh đó phải ở dạng jpeg hay jpg hay png) --> sau khi đã upload được lên thì nhấn nút Classify rồi để model chạy 1 lúc. Nếu muốn chọn lại bức khác thì nhớ nhấn nút xóa bức cũ đi rồi chọn lại tấm mới
  * Nếu muốn thêm model thì chỉnh lại phần def load_trained_model() và def predict() lại

## 7. Hướng phát triển:
* Train trên toàn bộ dữ liệu ảnh, có thể sẽ chuyển qua file TFRecord ddeere tăng hiệu quả training lên
* Như đã nói ở trên, sẽ áp dụng lớp Preproccessing Input, thêm các lớp Dense layers phía sau
* Áp dụng các phương pháp Image Segmentation + Object Detection để nâng cao khả năng xác định được các nốt ruồi (moles) trong ảnh
* Phân tích các thang đo ROC-AUC curve và confusion matrix để xác định được threshold nên dùng trong việc xác định mole này là bening hay malignant
* Xây dựng web bắt mắt và hiệu quả hơn, dễ sử dụng cũng như tiếp cận nhiều đối tượng khác nhau.
* Nghiên cứu thực hiện nhận biết nhiều nôt ruồi cùng lúc (hợp lí trong tình huống deploy model thực tế hơn)



