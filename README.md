🌟 星空職業預言師 - Streamlit Cloud 部署指南
📋 專案概述
這是一個基於占星學的職業預測網頁應用，使用 CatBoost 機器學習模型和 Gemini AI 提供個人化職業建議。

🚀 快速部署步驟
1. 準備文件結構
your-repo/
├── app.py                    # 主應用文件
├── model_utils.py           # 模型工具
├── gemini_integration.py    # Gemini API 集成
├── requirements.txt         # 依賴包
├── .streamlit/
│   └── config.toml         # Streamlit 配置
├── best_catboost_model.pkl  # 模型文件 (可選)
└── README.md               # 說明文件
2. GitHub 設置
創建新的 GitHub 倉庫
上傳所有文件到倉庫
確保倉庫是公開的 (或者使用 Streamlit Cloud Pro)
3. Streamlit Cloud 部署
前往 share.streamlit.io
使用 GitHub 帳號登入
點擊 "New app"
選擇您的倉庫和分支
設置主文件為 app.py
點擊 "Deploy!"
4. 環境變數設置 (可選)
如果要使用 Gemini AI，在 Streamlit Cloud 的 Advanced settings 中添加：

GEMINI_API_KEY = "your_actual_gemini_api_key"
📦 依賴包說明
核心依賴
streamlit>=1.28.0 - Web 應用框架
pandas>=1.5.0 - 數據處理
numpy>=1.24.0 - 數值計算
pyswisseph>=2.10.0 - 占星計算
plotly>=5.15.0 - 互動圖表
catboost>=1.2.0 - 機器學習模型
可選依賴
requests>=2.31.0 - API 請求 (Gemini)
scikit-learn>=1.3.0 - 機器學習工具
🔧 功能特色
✨ 占星計算
使用 Swiss Ephemeris 進行精確的行星位置計算
支援全球主要城市的時區自動調整
計算行星在星座和宮位的分佈
🤖 AI 預測
CatBoost 多標籤分類模型預測職業適合度
345 維占星特徵向量
Top 5 職業領域推薦
💫 個人化建議
整合 Gemini AI 提供客製化職業建議
基於個人星盤配置的深度分析
溫暖專業的建議語調
🎨 視覺設計
星空主題的深色界面
土黃色配色方案
動態星星背景動畫
響應式設計
🛠️ 自定義設置
修改模型
如果您有自己訓練的模型：

將模型文件命名為 best_catboost_model.pkl
上傳到根目錄
確保模型與 345 維特徵向量相容
添加城市
在 app.py 中的 CITIES_DATA 字典添加新城市：

python
"新城市, 國家": {"lat": 緯度, "lon": 經度, "tz": "時區"},
Gemini API 設置
前往 Google AI Studio
獲取免費的 API 密鑰
在 Streamlit Cloud 環境變數中設置 GEMINI_API_KEY
🐛 常見問題
Q: 模型載入失敗
A: 應用會自動使用備用的模擬模型，功能仍可正常運作

Q: Gemini API 不工作
A: 會使用內建的備用建議系統，基於星座特質提供建議

Q: 部署時間過長
A: 初次部署需要安裝 pyswisseph 等大型包，請耐心等待

Q: 占星計算錯誤
A: 檢查出生時間和地點是否正確，系統會自動處理時區轉換

🎯 使用說明
選擇出生資訊: 在左側邊欄設置出生日期、時間和城市
計算星盤: 點擊"開始算命"按鈕
查看結果: 系統會顯示星盤配置和職業預測
獲取AI建議: 點擊按鈕獲取個人化的職業建議
📈 效能優化建議
使用 @st.cache_resource 快取模型載入
使用 @st.cache_data 快取計算結果
考慮使用 Streamlit Cloud Pro 獲得更好效能
🔒 隱私說明
應用不會儲存用戶的個人資訊
所有計算都在伺服器端進行
Gemini API 呼叫遵循 Google 隱私政策
📞 技術支援
如果遇到問題，請檢查：

GitHub 倉庫是否公開
所有必要文件是否已上傳
requirements.txt 格式是否正確
Streamlit Cloud 日誌中的錯誤訊息
🌟 享受您的星空職業預言師之旅！ 🌟

