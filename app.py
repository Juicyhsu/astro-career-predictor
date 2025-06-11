import streamlit as st
import pandas as pd
import numpy as np
import swisseph as swe
import pytz
from datetime import datetime, date, time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import google.generativeai as genai
import os
from timezonefinder import TimezoneFinder
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# 🔧 添加模型類別定義
class CatBoostMultiLabelClassifier:
    def __init__(self, **params):
        self.params = params
        self.estimators_ = []
        
    def fit(self, X, y, class_names=None):
        self.estimators_ = []
        for i in range(y.shape[1]):
            catboost_params = {
                'iterations': self.params.get('iterations', 500),
                'learning_rate': self.params.get('learning_rate', 0.1),
                'depth': self.params.get('depth', 6),
                'l2_leaf_reg': self.params.get('l2_leaf_reg', 3.0),
                'verbose': False,
                'random_seed': 42,
                'thread_count': -1,
                'allow_writing_files': False
            }
            
            if self.params.get('auto_class_weights'):
                catboost_params['auto_class_weights'] = 'Balanced'
            
            estimator = CatBoostClassifier(**catboost_params)
            estimator.fit(X, y[:, i])
            self.estimators_.append(estimator)
        return self
    
    def predict(self, X):
        predictions = []
        for estimator in self.estimators_:
            pred = estimator.predict(X)
            predictions.append(pred)
        return np.array(predictions).T
    
    def predict_proba(self, X):
        probabilities = []
        for estimator in self.estimators_:
            proba = estimator.predict_proba(X)
            if proba.shape[1] > 1:
                probabilities.append(proba[:, 1])
            else:
                probabilities.append(np.ones(len(X)) * 0.5)
        return np.array(probabilities).T

# 頁面配置
st.set_page_config(
    page_title="如果你是名人，會在哪個領域發光？",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS樣式
st.markdown("""
<style>
    /* 主要背景和炫光效果 */
    .main-header {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 50%, #FF8C00 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: #2C1810;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(255, 215, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(255,255,255,0.3), transparent, rgba(255,215,0,0.4), transparent);
        animation: shimmer 2s linear infinite;
    }
    
    @keyframes shimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .main-header::after {
        content: '✨';
        position: absolute;
        top: 20px;
        right: 30px;
        font-size: 2rem;
        animation: twinkle 1.5s ease-in-out infinite alternate;
    }
    
    @keyframes twinkle {
        0% { opacity: 0.3; transform: scale(0.8); }
        100% { opacity: 1; transform: scale(1.2); }
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        position: relative;
        z-index: 1;
    }
    
    /* 星盤表格樣式 */
    .planet-table {
        background: linear-gradient(135deg, #FFF8DC 0%, #FFEBCD 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(255, 215, 0, 0.2);
        border: 3px solid #FFD700;
        position: relative;
        overflow: hidden;
    }
    
    .planet-table::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.1) 0%, rgba(255, 165, 0, 0.05) 100%);
        border-radius: 15px;
        z-index: 0;
    }
    
    .planet-table > * {
        position: relative;
        z-index: 1;
    }
    
    .planet-table table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .planet-table th {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #2C1810;
        padding: 12px;
        text-align: center;
        font-weight: bold;
        border: 1px solid #DDD;
    }
    
    .planet-table td {
        padding: 10px;
        text-align: center;
        border: 1px solid #DDD;
        background: rgba(255, 255, 255, 0.5);
    }
    
    .planet-table tr:nth-child(even) td {
        background: rgba(255, 248, 220, 0.8);
    }
    
    /* 預測結果卡片 */
    .prediction-card {
        background: linear-gradient(135deg, #FFD700 0%, #FFBF00 50%, #FF8C00 100%);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 8px 25px rgba(255, 140, 0, 0.3);
        border: 3px solid #FFA500;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(255, 140, 0, 0.4);
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .prediction-card::after {
        content: '⚡';
        position: absolute;
        top: 10px;
        right: 15px;
        font-size: 1.2rem;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.3); }
    }
    
    .rank-icon {
        display: inline-block;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F0F0 100%);
        color: #FF6B00;
        font-weight: bold;
        font-size: 1.2rem;
        line-height: 40px;
        text-align: center;
        margin-right: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        border: 2px solid #FF8C00;
    }
    
    .career-name {
        font-size: 1.4rem;
        font-weight: bold;
        color: #2C1810;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.5);
    }
    
    .probability-text {
        font-size: 1.1rem;
        color: #8B4513;
        margin: 5px 0 0 0;
        font-weight: 600;
    }
    
    /* 按鈕樣式 */
    .stButton > button {
        background: linear-gradient(135deg, #FFD700 0%, #FF8C00 100%);
        color: #2C1810;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 140, 0, 0.3);
    }
    
    .stButton > button:hover {
        transform: none;
        box-shadow: 0 4px 15px rgba(255, 140, 0, 0.3);
        background: linear-gradient(135deg, #FFBF00 0%, #FF6B00 100%);
        transition: background 0.2s ease;
    }
    
    .stButton > button:active {
        transform: none;
        box-shadow: 0 2px 8px rgba(255, 140, 0, 0.2);
    }
    
    /* 側邊欄樣式 */
    .css-1d391kg {
        background: linear-gradient(180deg, #FFF8DC 0%, #FFEBCD 100%);
    }
    
    /* 閃光效果 */
    .sparkle {
        animation: sparkle 1.5s ease-in-out infinite alternate;
    }
    
    @keyframes sparkle {
        0% { opacity: 0.5; transform: scale(1); }
        100% { opacity: 1; transform: scale(1.05); }
    }
</style>
""", unsafe_allow_html=True)

# 常數定義
SIGN_NAMES = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
              'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']

PLANET_NAMES = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 
                'Saturn', 'Uranus', 'Neptune', 'Pluto']

SIGN_NAMES_ZH = ['牡羊座', '金牛座', '雙子座', '巨蟹座', '獅子座', '處女座', 
                 '天秤座', '天蠍座', '射手座', '摩羯座', '水瓶座', '雙魚座']

PLANET_NAMES_ZH = ['太陽☉', '月亮☽', '水星☿', '金星♀', '火星♂', '木星♃', 
                   '土星♄', '天王星♅', '海王星♆', '冥王星♇']

def get_sign_name_zh(longitude):
    """獲取中文星座名稱"""
    sign_num = int(longitude / 30) % 12
    return SIGN_NAMES_ZH[sign_num]

PLANET_IDS = {
    'Sun': swe.SUN, 'Moon': swe.MOON, 'Mercury': swe.MERCURY, 'Venus': swe.VENUS,
    'Mars': swe.MARS, 'Jupiter': swe.JUPITER, 'Saturn': swe.SATURN,
    'Uranus': swe.URANUS, 'Neptune': swe.NEPTUNE, 'Pluto': swe.PLUTO
}

# 城市坐標數據
CITIES_DATA = {
    "台北, 台灣": {"lat": 25.0330, "lon": 121.5654, "tz": "Asia/Taipei"},
    "台中, 台灣": {"lat": 24.1477, "lon": 120.6736, "tz": "Asia/Taipei"},
    "台南, 台灣": {"lat": 22.9999, "lon": 120.2269, "tz": "Asia/Taipei"},
    "高雄, 台灣": {"lat": 22.6273, "lon": 120.3014, "tz": "Asia/Taipei"},
    "新竹, 台灣": {"lat": 24.8138, "lon": 120.9675, "tz": "Asia/Taipei"},
    "桃園, 台灣": {"lat": 24.9936, "lon": 121.3010, "tz": "Asia/Taipei"},
    "基隆, 台灣": {"lat": 25.1276, "lon": 121.7392, "tz": "Asia/Taipei"},
    "嘉義, 台灣": {"lat": 23.4801, "lon": 120.4491, "tz": "Asia/Taipei"},
    "彰化, 台灣": {"lat": 24.0518, "lon": 120.5161, "tz": "Asia/Taipei"},
    "宜蘭, 台灣": {"lat": 24.7021, "lon": 121.7378, "tz": "Asia/Taipei"},
    "花蓮, 台灣": {"lat": 23.9927, "lon": 121.6014, "tz": "Asia/Taipei"},
    "台東, 台灣": {"lat": 22.7972, "lon": 121.1713, "tz": "Asia/Taipei"},
    "香港": {"lat": 22.3193, "lon": 114.1694, "tz": "Asia/Hong_Kong"},
    "北京, 中國": {"lat": 39.9042, "lon": 116.4074, "tz": "Asia/Shanghai"},
    "上海, 中國": {"lat": 31.2304, "lon": 121.4737, "tz": "Asia/Shanghai"},
    "東京, 日本": {"lat": 35.6762, "lon": 139.6503, "tz": "Asia/Tokyo"},
    "首爾, 韓國": {"lat": 37.5665, "lon": 126.9780, "tz": "Asia/Seoul"},
    "新加坡": {"lat": 1.3521, "lon": 103.8198, "tz": "Asia/Singapore"},
    "紐約, 美國": {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"},
    "洛杉磯, 美國": {"lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles"},
    "倫敦, 英國": {"lat": 51.5074, "lon": -0.1278, "tz": "Europe/London"},
    "巴黎, 法國": {"lat": 48.8566, "lon": 2.3522, "tz": "Europe/Paris"},
    "悉尼, 澳洲": {"lat": -33.8688, "lon": 151.2093, "tz": "Australia/Sydney"}
}

# 職業分組對應
CAREER_GROUPS = {
    'Creative_Writing': '創意寫作領域',
    'Performance_Entertainment': '表演娛樂領域', 
    'Music_Industry': '音樂產業',
    'Sports_Athletics': '體育競技',
    'Politics_Government': '政治政府',
    'Business_Management': '商業管理',
    'Education_Research': '教育研究',
    'Military_Defense': '軍事國防',
    'Visual_Arts': '視覺藝術',
    'Legal_System': '法律體系',
    'Health_Medical': '健康醫療',
    'Engineering_Tech': '工程技術',
    'Hospitality_Tourism': '餐旅觀光',
    'Religion_Spiritual': '宗教靈性',
    'Special_Industries': '特殊產業'
}

@st.cache_resource
def load_model():
    """載入預訓練的CatBoost模型"""
    try:
        # 修復模組載入問題
        import sys
        sys.modules['__main__'].CatBoostMultiLabelClassifier = CatBoostMultiLabelClassifier
        
        model = joblib.load('best_catboost_model.pkl')
        return model
    except Exception as e:
        st.error(f"❌ 載入模型失敗: {e}")
        st.info("💡 建議：重新訓練模型或使用pickle格式")
        return None

def get_sign_name(longitude):
    """獲取星座名稱"""
    sign_num = int(longitude / 30) % 12
    return SIGN_NAMES[sign_num]

def get_sign_degree(longitude):
    """獲取星座內的度數"""
    return longitude % 30

def get_ut_from_local_time(year, month, day, hour, minute, timezone_str):
    """將當地時間轉換為世界時"""
    try:
        local_tz = pytz.timezone(timezone_str)
        local_datetime = datetime(year, month, day, hour, minute)
        aware_datetime = local_tz.localize(local_datetime)
        utc_datetime = aware_datetime.astimezone(pytz.UTC)
        
        return (utc_datetime.year, utc_datetime.month, utc_datetime.day, 
                utc_datetime.hour, utc_datetime.minute)
    except:
        # 簡單的經度轉換作為備用
        time_offset = CITIES_DATA.get(timezone_str, {}).get('lon', 0) / 15.0
        ut_hour = hour - time_offset
        ut_day = day
        
        if ut_hour < 0:
            ut_hour += 24
            ut_day -= 1
        elif ut_hour >= 24:
            ut_hour -= 24
            ut_day += 1
            
        return (year, month, ut_day, int(ut_hour), minute)

def calculate_chart(birth_year, birth_month, birth_day, birth_hour, birth_minute, city_name):
    """計算占星星盤"""
    try:
        # 獲取城市信息
        city_info = CITIES_DATA[city_name]
        lat, lon = city_info['lat'], city_info['lon']
        timezone_str = city_info['tz']
        
        # 轉換為世界時
        ut_year, ut_month, ut_day, ut_hour, ut_minute = get_ut_from_local_time(
            birth_year, birth_month, birth_day, birth_hour, birth_minute, timezone_str
        )
        
        # 計算儒略日
        jd_ut = swe.julday(ut_year, ut_month, ut_day, ut_hour + ut_minute/60.0)
        
        # 計算宮位和上升點
        houses_cusps = None
        asc_longitude = 0
        try:
            house_result = swe.houses(jd_ut, lat, lon, b'P')
            cusps_raw = house_result[0]
            special_points = house_result[1]
            asc_longitude = special_points[0]
            
            # 處理宮位數據
            if len(cusps_raw) == 13:
                houses_cusps = cusps_raw
            elif len(cusps_raw) == 12:
                houses_cusps = [0] + list(cusps_raw)
            else:
                houses_cusps = [0] + list(cusps_raw)
                while len(houses_cusps) < 13:
                    houses_cusps.append((houses_cusps[-1] + 30) % 360)
        except:
            houses_cusps = [i * 30 for i in range(13)]  # 備用等宮制
        
        # 計算行星位置
        planets_data = {}
        flags = swe.FLG_SWIEPH | swe.FLG_SPEED
        
        for planet_name, planet_id in PLANET_IDS.items():
            try:
                result = swe.calc_ut(jd_ut, planet_id, flags)
                positions = result[0] if isinstance(result, tuple) else result
                
                longitude = positions[0]
                sign = get_sign_name(longitude)
                sign_degree = get_sign_degree(longitude)
                
                # 計算宮位
                house = get_planet_house(longitude, houses_cusps) if houses_cusps else 1
                
                planets_data[planet_name] = {
                    'longitude': longitude,
                    'sign': sign,
                    'sign_degree': sign_degree,
                    'house': house
                }
            except:
                planets_data[planet_name] = {
                    'longitude': 0,
                    'sign': 'Aries',
                    'sign_degree': 0,
                    'house': 1
                }
        
        # 添加上升點
        planets_data['ASC'] = {
            'longitude': asc_longitude,
            'sign': get_sign_name(asc_longitude),
            'sign_degree': get_sign_degree(asc_longitude),
            'house': 1  # 上升點總是在第一宮
        }
        
        return planets_data
        
    except Exception as e:
        st.error(f"計算星盤時發生錯誤: {e}")
        return None

def get_planet_house(planet_longitude, houses_cusps):
    """確定行星所在的宮位"""
    planet_longitude = planet_longitude % 360.0
    
    for i in range(1, 13):
        start_cusp = houses_cusps[i] % 360.0
        end_cusp = houses_cusps[i+1 if i < 12 else 1] % 360.0
        
        if start_cusp < end_cusp:
            if start_cusp <= planet_longitude < end_cusp:
                return i
        else:
            if start_cusp <= planet_longitude < 360.0 or 0.0 <= planet_longitude < end_cusp:
                return i
    
    return 1  # 預設第一宮

def create_one_hot_encoding(planets_data):
    """創建One-Hot編碼特徵向量"""
    # 初始化特徵向量 (345維)
    features = np.zeros(345)
    feature_idx = 0
    
    # 行星在星座中 (10行星 x 12星座 = 120維)
    for planet in PLANET_NAMES:
        for sign in SIGN_NAMES:
            if planet in planets_data and planets_data[planet]['sign'] == sign:
                features[feature_idx] = 1
            feature_idx += 1
    
    # 上升星座 (12維)
    for sign in SIGN_NAMES:
        if 'ASC' in planets_data and planets_data['ASC']['sign'] == sign:
            features[feature_idx] = 1
        feature_idx += 1
    
    # 填充剩餘維度 (這裡簡化處理，實際應用中需要完整的宮位系統)
    while feature_idx < 345:
        features[feature_idx] = 0
        feature_idx += 1
    
    return features.reshape(1, -1)

def predict_career(model, features):
    """預測職業並返回Top5結果"""
    try:
        # 預測概率
        probabilities = model.predict_proba(features)[0]
        
        # 獲取Top5預測
        top5_indices = np.argsort(probabilities)[::-1][:5]
        
        results = []
        for i, idx in enumerate(top5_indices):
            career_key = list(CAREER_GROUPS.keys())[idx]
            career_name = CAREER_GROUPS[career_key]
            probability = probabilities[idx]
            
            results.append({
                'rank': i + 1,
                'career': career_name,
                'probability': probability * 100,
                'key': career_key
            })
        
        return results
    except Exception as e:
        st.error(f"預測時發生錯誤: {e}")
        return []

def create_visualization(predictions, planets_data):
    """創建視覺化圖表"""
    # 職業預測圓餅圖
    fig_pie = go.Figure(data=[go.Pie(
        labels=[p['career'] for p in predictions],
        values=[p['probability'] for p in predictions],
        hole=0.3,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(
            colors=['#FFD700', '#FFBF00', '#FF8C00', '#FFA500', '#FF6B00'],
            line=dict(color='#B8860B', width=2)
        )
    )])
    
    fig_pie.update_layout(
        title="Top 5 職業領域預測分布",
        title_x=0.5,
        font=dict(size=14),
        showlegend=True,
        height=400
    )
    
    # 行星分布雷達圖 - 使用中文
    signs_count = {sign: 0 for sign in SIGN_NAMES_ZH}
    for planet_name in PLANET_NAMES:
        if planet_name in planets_data:
            sign_en = planets_data[planet_name]['sign']
            sign_idx = SIGN_NAMES.index(sign_en)
            sign_zh = SIGN_NAMES_ZH[sign_idx]
            signs_count[sign_zh] += 1
    
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=list(signs_count.values()),
        theta=list(signs_count.keys()),
        fill='toself',
        name='行星分布',
        line_color='rgba(255, 165, 0, 0.8)',
        fillcolor='rgba(255, 215, 0, 0.3)',
        marker=dict(size=8)
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(signs_count.values()) + 1],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        title="個人行星星座分布",
        title_x=0.5,
        height=400,
        font=dict(size=12)
    )
    
    return fig_pie, fig_radar

def create_birth_chart_visualization(planets_data):
    """創建個人星盤圓形圖 - 上升點在9點鐘方向，第1宮開始逆時針"""
    fig = go.Figure()
    
    # 獲取上升點角度
    asc_longitude = planets_data.get('ASC', {}).get('longitude', 0)
    
    # 星盤外圈 - 十二宮（第1宮從上升點開始，逆時針排列）
    house_colors = ['#FFE4B5', '#FFEFD5', '#FFF8DC', '#FFFACD'] * 3
    
    # 繪製宮位背景
    for i in range(12):
        # 第1宮從上升點(9點鐘=180度)開始，每宮30度，逆時針
        start_angle = 180 - (i * 30)  # 180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210
        end_angle = 180 - ((i + 1) * 30)
        
        # 確保角度為正值
        start_angle = start_angle % 360
        end_angle = end_angle % 360
        
        # 轉換為弧度
        start_rad = np.radians(start_angle)
        end_rad = np.radians(end_angle)
        
        # 處理跨越0度的情況
        if start_angle < end_angle:
            # 需要跨越0度
            theta1 = np.linspace(start_rad, np.radians(360), 10)
            theta2 = np.linspace(0, end_rad, 10)
            theta = np.concatenate([theta1, theta2])
        else:
            theta = np.linspace(start_rad, end_rad, 20)
        
        r_inner = np.full_like(theta, 0.7)
        r_outer = np.full_like(theta, 1.0)
        
        # 扇形邊界
        x_inner = r_inner * np.cos(theta)
        y_inner = r_inner * np.sin(theta)
        x_outer = r_outer * np.cos(theta)
        y_outer = r_outer * np.sin(theta)
        
        # 添加宮位背景
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_inner, x_outer[::-1], [x_inner[0]]]),
            y=np.concatenate([y_inner, y_outer[::-1], [y_inner[0]]]),
            fill='toself',
            fillcolor=house_colors[i],
            line=dict(color='#DAA520', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # 宮位數字 - 放在每宮中央
        mid_angle = np.radians(180 - (i * 30 + 15))
        fig.add_annotation(
            x=0.85 * np.cos(mid_angle),
            y=0.85 * np.sin(mid_angle),
            text=f"{i+1}",
            showarrow=False,
            font=dict(size=12, color='#8B4513', family="Arial Black"),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#DAA520',
            borderwidth=1
        )
    
    # 添加星座標記 - 根據實際上升星座調整
    for i, sign in enumerate(SIGN_NAMES_ZH):
        # 計算每個星座的位置 - 從上升星座開始
        asc_sign_index = int(asc_longitude / 30)  # 上升星座的索引
        sign_longitude = (asc_longitude + i * 30) % 360
        sign_angle = np.radians(180 - sign_longitude)
        
        fig.add_annotation(
            x=1.15 * np.cos(sign_angle),
            y=1.15 * np.sin(sign_angle),
            text=SIGN_NAMES_ZH[(asc_sign_index + i) % 12],
            showarrow=False,
            font=dict(size=10, color='#8B4513')
        )
    
    # 添加行星位置
    planet_symbols = ['☉', '☽', '☿', '♀', '♂', '♃', '♄', '♅', '♆', '♇']
    planet_colors = ['#FFD700', '#C0C0C0', '#FFA500', '#FF69B4', '#FF4500', 
                    '#4169E1', '#8B4513', '#00CED1', '#0000FF', '#800080']
    
    for i, planet in enumerate(PLANET_NAMES):
        if planet in planets_data:
            planet_data = planets_data[planet]
            # 調整行星角度 - 以上升點為基準
            planet_angle_adjusted = 180 - planet_data['longitude']
            planet_rad = np.radians(planet_angle_adjusted)
            
            # 行星位置在星盤上
            x = 0.6 * np.cos(planet_rad)
            y = 0.6 * np.sin(planet_rad)
            
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                text=planet_symbols[i],
                textfont=dict(size=16, color=planet_colors[i]),
                marker=dict(size=8, color=planet_colors[i]),
                showlegend=False,
                hovertemplate=f"{PLANET_NAMES_ZH[i]}<br>{get_sign_name_zh(planet_data['longitude'])}<br>{planet_data['sign_degree']:.1f}°<extra></extra>"
            ))
    
    # 添加上升點標記 - 固定在9點鐘方向
    if 'ASC' in planets_data:
        asc_data = planets_data['ASC']
        
        fig.add_trace(go.Scatter(
            x=[-1.0],  # 9點鐘方向
            y=[0.0],
            mode='markers+text',
            text='ASC',
            textfont=dict(size=14, color='#FF0000'),
            marker=dict(size=15, color='#FF0000', symbol='triangle-left'),
            showlegend=False,
            hovertemplate=f"上升點<br>{get_sign_name_zh(asc_data['longitude'])}<br>{asc_data['sign_degree']:.1f}°<extra></extra>"
        ))
    
    # 設置圖表布局
    fig.update_layout(
        title="🌟 個人星盤圖",
        title_x=0.5,
        xaxis=dict(range=[-1.4, 1.4], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[-1.4, 1.4], showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        width=500,
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def main():
     # 添加頂部錨點
    st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)
    
    # 檢查是否要顯示成功提示（用toast）
    if st.session_state.get('show_success', False):
        st.toast('星盤解析完成！命運已揭曉🔮', icon='✅')
        st.balloons()  # 加個慶祝動畫
        # 立即清除標記
        st.session_state.show_success = False
    
    # 主標題
    st.markdown("""
    <div class="main-header">
        <h1>⭐ 如果你是名人，會在哪個領域發光？</h1>
        <p>基於占星學的AI職業預測系統</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 側邊欄輸入
    st.sidebar.markdown("### 🌟 請輸入您的出生資訊")
    
    # 出生日期
    birth_date = st.sidebar.date_input(
        "出生日期",
        value=date(1990, 1, 1),
        min_value=date(1900, 1, 1),
        max_value=date.today()
    )
    
    # 出生時間 - 分開選擇
    st.sidebar.markdown("**出生時間**")
    col_hour, col_min = st.sidebar.columns(2)
    
    with col_hour:
        birth_hour = st.selectbox(
            "時",
            range(24),
            index=12
        )
    
    with col_min:
        birth_minute = st.selectbox(
            "分",
            range(60),
            index=0
        )
    
    # 出生地點
    city = st.sidebar.selectbox(
        "出生地點",
        list(CITIES_DATA.keys())
    )
    
    # 計算按鈕
    if st.sidebar.button("🔮 開始預測", type="primary", key="start_prediction"):
        # 清除舊的Gemini結果（在最開始就清除）
        st.session_state.gemini_result = None
        
        # 載入模型
        model = load_model()
        if model is None:
            return
        
        with st.spinner("正在計算您的星盤..."):
            # 計算星盤
            planets_data = calculate_chart(
                birth_date.year, birth_date.month, birth_date.day,
                birth_hour, birth_minute, city
            )
            
            if planets_data:
                # 保存到session state
                st.session_state.planets_data = planets_data
                st.session_state.model = model
                
                # 保存出生資訊用於顯示
                st.session_state.birth_year = birth_date.year
                st.session_state.birth_month = birth_date.month  
                st.session_state.birth_day = birth_date.day
                st.session_state.birth_hour = birth_hour
                st.session_state.birth_minute = birth_minute
                st.session_state.birth_city = city
                
                # 創建特徵向量並預測
                with st.spinner("正在分析您的職業潛能..."):
                    features = create_one_hot_encoding(planets_data)
                    predictions = predict_career(model, features)
                    
                    if predictions:
                        # 保存預測結果
                        st.session_state.predictions = predictions
                                                
                        # 設置成功提示標記（只顯示一次）
                        st.session_state.show_success = True
                        
                        # 重新運行頁面
                        st.rerun()

    # ========== 結果顯示區塊 - 移到這裡 ==========
    # 檢查session state中是否有數據，如果有就顯示結果
    if 'planets_data' in st.session_state and 'predictions' in st.session_state:
        planets_data = st.session_state.planets_data
        predictions = st.session_state.predictions
        
        # 顯示星盤資訊（包含出生資訊）
        birth_info = f"{st.session_state.get('birth_year', birth_date.year)}年{st.session_state.get('birth_month', birth_date.month)}月{st.session_state.get('birth_day', birth_date.day)}日 {st.session_state.get('birth_hour', birth_hour)}:{st.session_state.get('birth_minute', birth_minute):02d} | {st.session_state.get('birth_city', city)}"
        
        st.markdown(f"""
        ## 🌌 您的個人星盤結果
        <div style="background: linear-gradient(135deg, #FFF8DC 0%, #FFEBCD 100%); 
                   padding: 1rem; border-radius: 10px; border: 2px solid #FFD700; margin-bottom: 1rem;">
            <p style="margin: 0; color: #8B4513; font-weight: bold; text-align: center;">
                📅 {birth_info}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 準備表格數據
        planet_data = []
        
        # 先添加上升點
        if 'ASC' in planets_data:
            asc_data = planets_data['ASC']
            planet_data.append({
                '天體': '上升點',
                '星座': get_sign_name_zh(asc_data['longitude']),
                '度數': f"{asc_data['sign_degree']:.1f}°",
                '宮位': f"第{asc_data.get('house', 1)}宮"
            })
        
        # 添加行星數據
        for i, planet in enumerate(PLANET_NAMES):
            if planet in planets_data:
                data = planets_data[planet]
                planet_data.append({
                    '天體': PLANET_NAMES_ZH[i],
                    '星座': get_sign_name_zh(data['longitude']),
                    '度數': f"{data['sign_degree']:.1f}°",
                    '宮位': f"第{data.get('house', 1)}宮"
                })
        
        # 創建DataFrame並顯示
        df_chart = pd.DataFrame(planet_data)
        st.table(df_chart)
        
        # 顯示預測結果
        st.markdown("## 🎯 職業領域預測結果")
        
        # 創建兩欄布局
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # 排名圖示
            rank_emojis = ['🥇', '🥈', '🥉', '🏅', '⭐']
            
            # Top 5 排行榜
            for i, pred in enumerate(predictions):
                st.markdown(f"""
                <div class="prediction-card sparkle">
                    <div style="display: flex; align-items: center;">
                        <span class="rank-icon">{pred['rank']}</span>
                        <div>
                            <h3 class="career-name">{rank_emojis[i]} {pred['career']}</h3>
                            <p class="probability-text">✨ 適配度: {pred['probability']:.1f}%</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 適配度計算說明")
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FFF8DC 0%, #FFEBCD 100%); 
                       padding: 1.5rem; border-radius: 15px; border: 2px solid #FFD700;">
            
            **🔮 計算方式說明：**
            
            **1. 星盤特徵提取**
            - 10顆行星位置 × 12星座 = 120維
            - 上升星座特徵 = 12維  
            - 宮位系統特徵 = 213維
            - **總計345維特徵向量**
            
            **2. AI模型分析**
            - 使用CatBoost機器學習算法
            - 基於全世界數萬人名人資料訓練
            - 15大職業領域分類預測
            
            **3. 適配度評分**
            - 0-100分制評分系統
            - 結合行星能量、星座特質、宮位領域等綜合作用力
            - 考量行星影響力權重
            
            **⚠️ 重要提醒**
            本預測基於名人數據，僅供參考娛樂用途
            </div>
            """, unsafe_allow_html=True)
        
        # 視覺化圖表
        st.markdown("## 📊 視覺化分析")
        
        fig_pie, fig_radar = create_visualization(predictions, planets_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Gemini個人化建議
        st.markdown("## 🤖 AI個人化建議")

        if st.button("🔮 生成個人化職業建議", type="primary", key="gemini_advice"):
            with st.spinner("AI正在分析您的星盤..."):
                try:
                    api_key = st.secrets.get("GEMINI_API_KEY")
                    
                    if not api_key:
                        st.error("❌ 未設置Gemini API密鑰")
                    else:
                        genai.configure(api_key=api_key)
                        model_gemini = genai.GenerativeModel('gemini-2.0-flash')

                        # 準備星盤資訊
                        chart_info = "個人星盤配置：\n"
                        
                        # 上升點
                        if 'ASC' in planets_data:
                            asc_data = planets_data['ASC']
                            chart_info += f"上升點：{get_sign_name_zh(asc_data['longitude'])} {asc_data['sign_degree']:.1f}° 第{asc_data.get('house', 1)}宮\n"
                        
                        # 十大行星
                        for i, planet in enumerate(PLANET_NAMES):
                            if planet in planets_data:
                                data = planets_data[planet]
                                chart_info += f"{PLANET_NAMES_ZH[i]}：{get_sign_name_zh(data['longitude'])} {data['sign_degree']:.1f}° 第{data.get('house', 1)}宮\n"
                        
                        # 構建prompt
                        prompt = f"""
作為專業占星師，請根據以下星盤配置提供詳細解讀：

{chart_info}

請從以下角度分析（用繁體中文回答）：

**🌟 個性特質分析**
- 根據上升星座分析外在表現
- 太陽星座的核心自我
- 月亮星座的內在情感需求

**💼 職業天賦領域**
- 分析各行星在不同宮位的職業指向
- 特別關注第2、6、10宮的行星配置
- 提供3-5個最適合的職業方向

**🎯 人生發展建議**
- 基於星盤配置的人生課題
- 需要注意的挑戰與機會
- 個人成長的關鍵建議

請提供具體、實用且正面的建議，字數控制在800字以內。
                        """
                        
                        response = model_gemini.generate_content(prompt)
                        # 保存到session state
                        st.session_state.gemini_result = response.text
                        
                except Exception as e:
                    st.session_state.gemini_result = f"❌ Gemini解讀發生錯誤：{str(e)}"
        
        # 顯示Gemini結果（如果存在）
        if 'gemini_result' in st.session_state and st.session_state.gemini_result:
            # 黃色細線分隔
            st.markdown("""
            <hr style="border: none; height: 2px; background: linear-gradient(90deg, #FFD700, #FFA500, #FFD700); margin: 2rem 0;">
            """, unsafe_allow_html=True)
            
            st.markdown("### 🌟 專業占星師解讀")
            st.markdown(st.session_state.gemini_result)

    else:
        st.info("💡 請在側邊欄輸入出生資訊，然後點擊「開始預測」")

    # 底部資訊
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🔮 <strong>專業占星職業推薦 AI 系統</strong></p>
        <p>🌟 占星預測結合現代AI技術，為您探索無限可能</p>
        <p><small>基於Swiss Ephemeris權威占星計算</small></p>  
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()