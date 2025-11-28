
# **Pfizer Vaccine Sentiment Analysis & Engagement Prediction** 

**Mô tả:** \
Phân tích cảm xúc và tương tác của người dùng trên nền tảng mạng xã hội **Twitter** (X) về chủ đề **Vaccine Pfizer BioNTech**. Điểm khác biệt là việc tự xây dựng lại các thuật toán xử lý ngôn ngữ và mô hình phân loại, chủ yếu sử dụng thư viện `Numpy` để xử lý và dùng `Matplotlib`, `Seaborn` để trực quan hoá dữ liệu. Ngoài ra còn có sự trợ giúp của thư viện `Scikit-learn`, chủ yếu là để kiểm tra và báo cáo tính chính xác của mô hình tự tạo. 

---
**Mục lục**
## **Mục lục** 📑

---
- [**Pfizer Vaccine Sentiment Analysis \& Engagement Prediction**](#pfizer-vaccine-sentiment-analysis--engagement-prediction)
  - [**Mục lục** 📑](#mục-lục-)
  - [**1. Giới thiệu**](#1-giới-thiệu)
  - [**2. Bộ dataset**](#2-bộ-dataset)
  - [**3. Phương pháp**](#3-phương-pháp)
    - [**3.1 Tiền xử lý**](#31-tiền-xử-lý)
    - [**3.2 Feature Engineering**](#32-feature-engineering)
    - [**3.3 Mô hình hóa**](#33-mô-hình-hóa)
      - [**a. Logistic Regression**](#a-logistic-regression)
      - [**b. Naive Bayes (Multinomial)**](#b-naive-bayes-multinomial)
  - [**4. Cài đặt \& Thiết lập**](#4-cài-đặt--thiết-lập)
  - [**5. Hướng dẫn sử dụng**](#5-hướng-dẫn-sử-dụng)
  - [**6. Kết quả phân tích được**](#6-kết-quả-phân-tích-được)
  - [**7. Cấu trúc Dự án**](#7-cấu-trúc-dự-án)
  - [**8. Thách thức**](#8-thách-thức)
  - [**9. Hướng phát triển kế tiếp**](#9-hướng-phát-triển-kế-tiếp)
  - [**10.Contributors**](#10contributors)
---
---

## **1. Giới thiệu**

**Bài toán**

Sự ra đời của vaccine chống lại **COVID-19** đã tạo những làn sóng thảo luận mạnh mẽ trên nền tảng mạng xã hội, một trong số đó có vaccince **Pfizer BioNTech**. Bài toán đặt ra làm thế nào biết được ***công chúng có đón nhận vaccine mới này hay không*** và đồng thời cũng tìm hiểu xem ***điều gì để tạo nên sự viral của bài viết*** (điều này làm ảnh hưởng đến sự lan truyền thông tin của vaccine)

**Ý nghĩa**
- Giúp các tổ chức y tế và chính hiểu hiểu được mối lo ngại của người dân là gì.
- Phát hiện sớm các đợt khủng hoảng truyền thông. 

## **2. Bộ dataset**
- Lấy từ bộ dữ liệu đã thu thập sẵn trên Kaggle, bắt nguồn được thu thập từ Twitter API
- Kích cỡ: $\approx 11,000$ dòng dữ liệu
- Các thuộc tính quan trọng:
    - `text`: Nội dung bài tweet
    - `date`: Thời gian đăng
    - `user_followers`, `user_friends`: Số người theo dõi, và số người mà người dùng theo dõi
    - `retweets`, `favourites`: lần lượt là số lượt đăng lại và số lượng yêu thích
    - `is_verified`: trạng thái xác thực tài khoản


## **3. Phương pháp**
Ưu tiên sử dụng `Numpy` để thao tác tất cả =)), và kèm vào đó là hỗ trợ về mặt trực quan hoá của Matplotlib và Seaborn

### **3.1 Tiền xử lý**
- **Làm sạch:** Sử dụng Regex để loại bỏ URL, HTML tags, Mentions, Emoji,...
- **Stemming:** Tự cài đặt thuật toán Porter Stemmer (cắt hậu tố ing, ed, s...) để đưa từ về dạng gốc thay vì dùng thư viện NLTK.
- **Gán nhãn cảm xúc:** Xây dựng bộ từ điển *(POSITIVE_WORDS, NEGATIVE_WORDS)* và logic xử lý phủ định *(Negation handling)* để tự động gán nhãn:
    - **0:** Tiêu cực *(Negative)*
    - **1:** Trung tính *(Neutral)*
    - **2:** Tích cực *(Positive)*

### **3.2 Feature Engineering**
- **Engagement =  retweets + favourites.** Đo lường chất lượng nội dung
- **Reputation ratio = user_followers / user_friends + 1**. Đo lường tầm ảnh hưởng của user đó
- **Acc class:** Phân loại người dùng Weak, Normal, Strong, Influencer

### **3.3 Mô hình hóa**
Sử dụng kỹ thuật **TF-IDF** để vector hóa văn bản.

#### **a. Logistic Regression**
Sử dụng phương pháp **One vs Rest** cho bài toán đa lớp.
- **Hàm Sigmoid:**
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

- **Hàm mất mát (Log Loss + L2 Regularization):**
$$J(w) = - \frac{1}{m} \sum [y \log(\hat{y}) + (1-y) \log(1-\hat{y})] + \frac{\lambda}{2m} \sum w^2$$
- **Tối ưu hóa:** Gradient Descent.

#### **b. Naive Bayes (Multinomial)**
Sử dụng làm mô hình cơ sở (baseline) để so sánh hiệu quả.


## **4. Cài đặt & Thiết lập**
Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```
*(File requirements.txt bao gồm: numpy, matplotlib, seaborn, scikit-learn...)*

## **5. Hướng dẫn sử dụng**
Chạy các file `Notebook` theo thứ tự sau để tái hiện quy trình:

- `01_data_exploration.ipynb`:
    - Load dữ liệu thô.
    - Thực hiện EDA: Phân tích đơn biến, đa biến, chuỗi thời gian.
    - Tạo các biến mới (engagement, hashtags_count...).

- `02_preprocessing.ipynb`:
    - Làm sạch văn bản (Clean text).
    - Gán nhãn cảm xúc tự động.
    - Phân tích mối quan hệ giữa Sentiment và các yếu tố khác.
    - Xuất file sentiment_data.npz.
- `03_modeling.ipynb`:
    - Load dữ liệu đã xử lý.
    - Vector hóa văn bản (TF-IDF).
    - Huấn luyện và đánh giá mô hình (Logistic Regression vs Naive Bayes).

## **6. Kết quả phân tích được**
**Insights từ EDA:**
- **Verified vs Unverified:** Tài khoản Verified (báo chí, tổ chức) đóng vai trò là người đưa tin (Neutral cao), trong khi người dùng thường (Unverified) là nơi bộc lộ cảm xúc thật (Positive/Negative cao) và có Engagement cao hơn.
- **Số lượng Follower có tương quan rất thấp với Engagement.** Nội dung và cảm xúc mới là yếu tố quyết định độ viral.
- Các ngày có lượng tweet tiêu cực cao nhất thường trùng với các sự kiện gián đoạn nguồn cung vaccine hoặc tranh cãi về chính sách tiêm bắt buộc.

Kết quả Mô hình:
- **Logistic Regression:** Đạt độ chính xác khoảng 74%. Tuy nhiên, do dữ liệu mất cân bằng (lớp Neutral chiếm đa số), mô hình có xu hướng bị bias.

- **Naive Bayes:** Cho tốc độ huấn luyện nhanh hơn và xử lý tốt dữ liệu văn bản thưa (sparse data).

## **7. Cấu trúc Dự án**
```raw
Project/
├── data/
│   ├── raw/                
│   └── processed/         
├── src/
│   ├── __init__.py         
│   ├── config.py           # Từ điển từ vựng, Emoji map, Stopwords
│   ├── data_processing.py  # Hàm xử lý chính
│   └── models.py  
├── notebook/     
│   ├── 01_data_exploration.ipynb         
│   ├── 02_preprocessing.ipynb 
│   └── 03_modeling.ipynb            
├── requirements.txt            
└── README.md                   
```
## **8. Thách thức**

Trong quá trình thực hiện dự án mà không phụ thuộc vào `Pandas`/`Scikit-learn` cho các bước xử lý chính:
- Khi load vào dataframe bằng Numpy, ta phát hiện ra rằng cột text có các kí tự là dấu `,` khiến việc load trở nên khó khăn, ta phải đi xử lý file gốc rồi mới load vào được 
- Thêm cột mới bằng cách sử dụng `NumPy Structured Arrays` -> khó linh hoạt
- Tự định nghĩa từ điển, việc này khá là khó khăn trong lúc tìm từ.

## **9. Hướng phát triển kế tiếp**
- Xây dựng hệ thống cập nhật dữ liệu real-time
- Sử dụng các mô hình Deap learning để xử lý sâu hơn
## **10.Contributors**
<center>VÕ NGỌC TIẾN<center>
<center>HCMUS<center>
<center>tientien04012005@gmail.com<center>











