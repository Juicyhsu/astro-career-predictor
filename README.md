# 🌟 占星職業預測系統

## 如果你是名人，會在哪個領域發光？

基於占星學和深度學習的AI職業預測系統，使用Neural Network算法分析個人星盤，預測最適合的職業領域。

### ✨ 功能特色

- 🔮 **精準星盤計算**：使用Swiss Ephemeris計算精確的行星位置
- 🧠 **深度學習預測**：基於Neural Network模型的15大職業領域分析
- ⚖️ **智能特徵權重**：核心行星1.5x權重，外行星0.7x權重的專業配置
- 📊 **視覺化展示**：互動式圖表展示預測結果和星盤分布
- 🤖 **AI個人化建議**：整合Gemini AI提供純占星學深度建議
- 📱 **響應式設計**：支援桌面和行動裝置

### 🚀 線上體驗

[點擊這裡體驗](https://astro-career-predictor-141913333.streamlit.app/)

### 📋 支援的職業領域

**新15類平衡分組系統**

1. 創意寫作類
2. 表演娛樂類
3. 音樂產業類
4. 體育運動類
5. 政治政府類
6. 商業經營類
7. 教育學術類
8. 軍事國防類
9. 視覺藝術類
10. 法律司法類
11. 醫療健康類
12. 工程技術類
13. 餐旅服務類
14. 宗教精神類
15. 特殊行業類


### 🛠️ 技術架構

- **前端**：Streamlit + Plotly
- **後端**：Python + Swiss Ephemeris
- **深度學習**：Neural Network (Multi-Layer Perceptron)
- **特徵工程**：345維特徵向量 + 權重策略
- **AI整合**：Google Gemini API
- **部署**：Streamlit Cloud

### 🧠 Neural Network 模型特色

#### 🔮 計算方式說明

**1. 星盤特徵提取**
- 10顆行星位置 × 12星座 = 120維
- 上升星座特徵 = 12維  
- 宮位系統特徵 = 213維
- **總計345維特徵向量**

**2. AI模型分析**
- 使用Neural Network深度學習算法
- 基於全世界數萬名人資料訓練
- 15大職業領域分類預測

**3. 特徵權重策略**
- 核心行星 (太陽月亮等)：1.5x權重
- 社會行星 (木土)：1.2x權重
- 外行星 (天海冥)：0.7x權重

**4. 適配度評分**
- 0-100分制評分系統
- 結合行星能量、星座特質、宮位領域等綜合作用力
- 考量行星影響力權重

### ⚠️ 重要說明

本預測基於名人數據，僅供參考娛樂用途。AI個人化建議完全基於占星學原理，不受模型預測結果影響。

### 🔧 本地開發

```bash
# 克隆專案
git clone https://github.com/yourusername/astro-career-prediction.git
cd astro-career-prediction

# 安裝依賴
pip install -r requirements.txt

# 確保模型檔案位置正確
# - NeuralNetwork_Optimized_15_Class_Houses375_OPTIMIZED_15_CLASS.pkl
# - mlb_NeuralNetwork_Optimized_15_Class_Houses375_OPTIMIZED_15_CLASS.pkl
# - feature_weights_NeuralNetwork_Optimized_15_Class_Houses375_OPTIMIZED_15_CLASS.pkl
# - optimized_career_mapping_NeuralNetwork_Optimized_15_Class_Houses375_OPTIMIZED_15_CLASS.pkl

# 運行應用
streamlit run app.py
```

### 📦 依賴套件

```txt
streamlit
pandas
numpy
swisseph
pytz
plotly
joblib
google-generativeai
scikit-learn>=1.0.0
```

### 📊 模型資訊

- **演算法**：Neural Network (Multi-Layer Perceptron)
- **網路架構**：多層神經網路 (512-256-128 隱藏層)
- **特徵維度**：345維 (行星位置 + 星座 + 宮位系統)
- **訓練數據**：全世界數萬名人職業數據集
- **創新技術**：
  - 新15類平衡分組解決類別不平衡問題
  - 動態特徵權重基於占星學專業知識
  - WeightedMultiOutputClassifier支援類別權重
- **評估指標**：部分匹配準確率, Macro F1-Score, 適配度評分

### 🎯 技術創新

#### 1. 新15類平衡分組
- 從原始28類不平衡職業重組為15類平衡分組
- 解決名人數據偏見，提升普羅大眾適用性
- 保持職業語義清晰度

#### 2. 智能特徵權重
- 基於占星學專業知識的權重分配
- 核心行星加權，外行星降權
- 提升預測準確性

#### 3. 純占星學AI建議
- Gemini AI完全基於星盤配置分析
- 不受模型預測結果干擾
- 關注第2、6、10宮的職業指向

### 🌟 適用場景

- **生涯探索**：為學生提供平衡且清晰的職業方向建議
- **轉職規劃**：幫助上班族發現被忽略的適合領域
- **人才發展**：輔助HR發現員工潛在職業適性
- **多元職涯**：支持現代協槓族的多重職業發展
- **職業諮詢**：提供專業且實用的職業建議工具

### 👥 貢獻

歡迎提交Issue和Pull Request！

### 📄 授權

MIT License

---

🌟 **基於Swiss Ephemeris權威占星計算**  
🧠 **Neural Network深度學習結合占星學專業知識**  
Made with ❤️ by [九水]