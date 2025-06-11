# 🌟 占星職業預測系統

## 如果你是名人，會在哪個領域發光？

基於占星學和機器學習的AI職業預測系統，使用CatBoost算法分析個人星盤，預測最適合的職業領域。

### ✨ 功能特色

- 🔮 **精準星盤計算**：使用Swiss Ephemeris計算精確的行星位置
- 🎯 **智能職業預測**：基於CatBoost模型的15大職業領域分析
- 📊 **視覺化展示**：互動式圖表展示預測結果和星盤分布
- 🤖 **AI個人化建議**：整合Gemini AI提供深度職業建議
- 📱 **響應式設計**：支援桌面和行動裝置

### 🚀 線上體驗

[點擊這裡體驗](your-streamlit-app-url)

### 📋 支援的職業領域

1. 創意寫作領域
2. 表演娛樂領域
3. 音樂產業
4. 體育競技
5. 政治政府
6. 商業管理
7. 教育研究
8. 軍事國防
9. 視覺藝術
10. 法律體系
11. 健康醫療
12. 工程技術
13. 餐旅觀光
14. 宗教靈性
15. 特殊產業

### 🎨 界面預覽

![主界面](screenshots/main-interface.png)
![預測結果](screenshots/prediction-results.png)

### 🛠️ 技術架構

- **前端**：Streamlit + Plotly
- **後端**：Python + Swiss Ephemeris
- **機器學習**：CatBoost
- **AI整合**：Google Gemini API
- **部署**：Streamlit Cloud

### ⚠️ 重要說明

本系統基於名人數據訓練，預測結果僅供參考娛樂，不應作為職業選擇的唯一依據。

### 🔧 本地開發

```bash
# 克隆專案
git clone https://github.com/yourusername/astro-career-prediction.git
cd astro-career-prediction

# 安裝依賴
pip install -r requirements.txt

# 運行應用
streamlit run app.py
```

### 📊 模型資訊

- **演算法**：CatBoost Multi-Label Classification
- **特徵維度**：345維 (行星位置 + 星座 + 宮位系統)
- **訓練數據**：名人職業數據集
- **評估指標**：Macro F1-Score, 部分匹配率, 平衡準確率

### 👥 貢獻

歡迎提交Issue和Pull Request！

### 📄 授權

MIT License

---

Made with ❤️ by [Your Name]