import streamlit as st
import pandas as pd
import numpy as np
import swisseph as swe
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import pickle
import requests
import json
from typing import Dict, List, Tuple
import re
import os

# 載入環境變數（本地開發用）
try:
    from dotenv import load_dotenv
    load_dotenv()  # 載入 .env 文件中的環境變數
except ImportError:
    # 如果沒有安裝 python-dotenv，忽略（在雲端部署時）
    pass

# 導入自定義模塊
from model_utils import load_or_create_model, predict_top_careers
from gemini_integration import GeminiAdvisor

# 設置頁面配置
st.set_page_config(
    page_title="🌟 星空職業預言師",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS樣式
st.markdown("""
<style>
    /* 主要背景 */
    .main {
        background: linear-gradient(135deg, #2c1810 0%, #3d2817 25%, #4a331f 50%, #3d2817 75%, #2c1810 100%);
        background-attachment: fixed;
    }
    
    /* 星空動畫背景 */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(2px 2px at 20px 30px, #ffd700, transparent),
            radial-gradient(2px 2px at 40px 70px, #fff, transparent),
            radial-gradient(1px 1px at 90px 40px, #ffd700, transparent),
            radial-gradient(1px 1px at 130px 80px, #fff, transparent),
            radial-gradient(2px 2px at 160px 30px, #ffd700, transparent),
            radial-gradient(1px 1px at 200px 60px, #fff, transparent),
            radial-gradient(1px 1px at 240px 90px, #ffd700, transparent),
            radial-gradient(2px 2px at 280px 20px, #fff, transparent),
            radial-gradient(1px 1px at 320px 70px, #ffd700, transparent);
        background-repeat: repeat;
        background-size: 350px 120px;
        animation: sparkle 20s linear infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes sparkle {
        0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 1; }
        50% { transform: translateY(-10px) rotate(180deg); opacity: 0.8; }
    }
    
    /* 標題樣式 */
    .main-title {
        text-align: center;
        color: #ffd700;
        font-size: 3.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px #000000;
        margin-bottom: 0.5rem;
        font-family: 'Georgia', serif;
    }
    
    .subtitle {
        text-align: center;
        color: #f4e4bc;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* 卡片樣式 */
    .card {
        background: rgba(61, 40, 23, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #ffd700;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }
    
    /* 按鈕樣式 */
    .stButton > button {
        background: linear-gradient(45deg, #ffd700, #ffed4e);
        color: #2c1810;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #ffed4e, #ffd700);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
    }
    
    /* 側邊欄樣式 */
    .css-1d391kg {
        background: linear-gradient(135deg, #2c1810, #3d2817);
        border-right: 2px solid #ffd700;
    }
    
    /* 文字樣式 */
    .prediction-text {
        color: #f4e4bc;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .planet-info {
        background: rgba(255, 215, 0, 0.1);
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ffd700;
    }
    
    /* 職業卡片 */
    .career-card {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.2), rgba(255, 237, 78, 0.1));
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ffd700;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .career-card:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# 常數定義
SIGN_NAMES = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
              'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']

PLANET_NAMES = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 
                'Saturn', 'Uranus', 'Neptune', 'Pluto']

PLANET_IDS = {
    'Sun': swe.SUN, 'Moon': swe.MOON, 'Mercury': swe.MERCURY,
    'Venus': swe.VENUS, 'Mars': swe.MARS, 'Jupiter': swe.JUPITER,
    'Saturn': swe.SATURN, 'Uranus': swe.URANUS, 
    'Neptune': swe.NEPTUNE, 'Pluto': swe.PLUTO
}

# 城市坐標數據
CITIES_DATA = {
    "台北, 台灣": {"lat": 25.0330, "lon": 121.5654, "tz": "Asia/Taipei"},
    "紐約, 美國": {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"},
    "倫敦, 英國": {"lat": 51.5074, "lon": -0.1278, "tz": "Europe/London"},
    "東京, 日本": {"lat": 35.6762, "lon": 139.6503, "tz": "Asia/Tokyo"},
    "巴黎, 法國": {"lat": 48.8566, "lon": 2.3522, "tz": "Europe/Paris"},
    "洛杉磯, 美國": {"lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles"},
    "香港": {"lat": 22.3193, "lon": 114.1694, "tz": "Asia/Hong_Kong"},
    "新加坡": {"lat": 1.3521, "lon": 103.8198, "tz": "Asia/Singapore"},
    "悉尼, 澳洲": {"lat": -33.8688, "lon": 151.2093, "tz": "Australia/Sydney"},
    "上海, 中國": {"lat": 31.2304, "lon": 121.4737, "tz": "Asia/Shanghai"}
}

# 職業類別映射
CAREER_CATEGORIES = {
    'Creative_Writing': '📝 創意寫作',
    'Performance_Entertainment': '🎭 表演娛樂',
    'Music_Industry': '🎵 音樂產業',
    'Sports_Athletics': '⚽ 體育運動',
    'Politics_Government': '🏛️ 政治政府',
    'Business_Management': '💼 商業管理',
    'Education_Research': '🎓 教育研究',
    'Military_Defense': '🛡️ 軍事國防',
    'Visual_Arts': '🎨 視覺藝術',
    'Legal_System': '⚖️ 法律系統',
    'Health_Medical': '🏥 健康醫療',
    'Engineering_Tech': '⚙️ 工程技術',
    'Hospitality_Tourism': '🏨 餐旅觀光',
    'Religion_Spiritual': '🙏 宗教靈性',
    'Special_Industries': '🌟 特殊產業'
}

@st.cache_resource
def load_model():
    """載入預訓練的CatBoost模型"""
    try:
        model = load_or_create_model('best_catboost_model.pkl')
        return model
    except Exception as e:
        st.warning(f"⚠️ 模型載入錯誤: {e}，將使用備用模型")
        return None

@st.cache_resource
def get_gemini_advisor():
    """獲取Gemini顧問實例"""
    return GeminiAdvisor()

def get_sign_name(longitude):
    """獲取星座名稱"""
    sign_num = int(longitude / 30) % 12
    return SIGN_NAMES[sign_num]

def get_sign_degree(longitude):
    """獲取星座內的度數"""
    return longitude % 30

def calculate_julian_day(year, month, day, hour, minute, timezone_offset=0):
    """計算儒略日"""
    # 調整為UTC時間
    utc_hour = hour - timezone_offset
    if utc_hour < 0:
        utc_hour += 24
        day -= 1
    elif utc_hour >= 24:
        utc_hour -= 24
        day += 1
    
    return swe.julday(year, month, day, utc_hour + minute/60.0)

def calculate_planets(jd, lat, lon):
    """計算行星位置"""
    planets_data = {}
    
    for planet_name, planet_id in PLANET_IDS.items():
        try:
            result = swe.calc_ut(jd, planet_id, swe.FLG_SWIEPH | swe.FLG_SPEED)
            if isinstance(result, tuple) and len(result) >= 1:
                positions = result[0]
                longitude = positions[0] if len(positions) > 0 else 0
                
                planets_data[planet_name] = {
                    'longitude': longitude,
                    'sign': get_sign_name(longitude),
                    'degree': get_sign_degree(longitude)
                }
        except:
            planets_data[planet_name] = {
                'longitude': 0,
                'sign': 'Aries',
                'degree': 0
            }
    
    return planets_data

def calculate_houses(jd, lat, lon):
    """計算宮位"""
    try:
        house_result = swe.houses(jd, lat, lon, b'P')  # Placidus system
        if isinstance(house_result, tuple) and len(house_result) >= 2:
            cusps = house_result[0]
            special_points = house_result[1]
            
            # 上升點
            asc_lon = special_points[0] if len(special_points) > 0 else 0
            
            return {
                'ASC': {
                    'longitude': asc_lon,
                    'sign': get_sign_name(asc_lon),
                    'degree': get_sign_degree(asc_lon)
                },
                'cusps': cusps
            }
    except:
        pass
    
    return {
        'ASC': {'longitude': 0, 'sign': 'Aries', 'degree': 0},
        'cusps': [0] * 13
    }

def create_one_hot_features(planets_data, houses_data):
    """創建One-Hot編碼特徵"""
    features = {}
    
    # 行星在星座 (10行星 x 12星座 = 120維)
    for planet in PLANET_NAMES:
        for sign in SIGN_NAMES:
            key = f"{planet}_{sign}"
            features[key] = 1 if planets_data.get(planet, {}).get('sign') == sign else 0
    
    # 上升星座 (12維)
    for sign in SIGN_NAMES:
        key = f"ASC_{sign}"
        features[key] = 1 if houses_data.get('ASC', {}).get('sign') == sign else 0
    
    # 其他特徵可以根據需要添加...
    # 這裡簡化處理，你可以根據完整的特徵集擴展
    
    return features

def predict_careers(features, model=None):
    """預測職業 - 使用新的模型系統"""
    try:
        # 這裡暫時保留原邏輯作為備用
        if model is None:
            categories = list(CAREER_CATEGORIES.keys())
            probabilities = np.random.dirichlet(np.ones(len(categories)))
            results = list(zip(categories, probabilities))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:5]
        
        # 使用真實模型預測的邏輯保持不變
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        predictions = model.predict_proba(feature_vector)[0]
        
        categories = list(CAREER_CATEGORIES.keys())
        results = list(zip(categories, predictions))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:5]
    except:
        return predict_careers(features, None)

def generate_gemini_advice(planets_data, houses_data=None):
    """使用Gemini生成個人化建議"""
    try:
        advisor = get_gemini_advisor()
        advice = advisor.get_advice(planets_data, houses_data)
        return advice
    except Exception as e:
        # 備用建議
        return f"""
        ✨ **個人星盤分析** ✨
        
        根據您的星盤配置：
        
        🌞 **太陽星座特質**
        您的太陽在{planets_data.get('Sun', {}).get('sign', 'Unknown')}座，展現出{get_sun_traits(planets_data.get('Sun', {}).get('sign', 'Unknown'))}的核心特質。
        
        🌙 **月亮情感需求**  
        月亮在{planets_data.get('Moon', {}).get('sign', 'Unknown')}座，內在情緒傾向{get_moon_traits(planets_data.get('Moon', {}).get('sign', 'Unknown'))}。
        
        💫 **職業發展建議**
        建議您在需要{get_career_suggestion(planets_data.get('Sun', {}).get('sign', 'Unknown'))}的領域發揮天賦，
        並結合您的月亮特質，在工作中保持情緒平衡。
        
        🎯 **成功關鍵**
        • 善用您的天賦，在適合的時機展現才華
        • 保持學習成長的心態，持續精進專業技能
        • 相信直覺，但也要理性分析
        • 建立良好的人際網絡
        
        記住，星盤是指引，成功還需要您的努力和智慧選擇！
        """

def get_sun_traits(sign):
    """獲取太陽星座特質"""
    traits = {
        'Aries': '積極主動、領導力強',
        'Taurus': '穩重可靠、有耐心',
        'Gemini': '機智靈活、溝通能力佳',
        'Cancer': '情感豐富、保護欲強',
        'Leo': '自信大方、創造力強',
        'Virgo': '細心謹慎、完美主義',
        'Libra': '和諧平衡、審美能力強',
        'Scorpio': '深度洞察、意志堅定',
        'Sagittarius': '樂觀進取、愛好自由',
        'Capricorn': '務實負責、目標明確',
        'Aquarius': '獨立創新、人道主義',
        'Pisces': '直覺敏銳、富有同情心'
    }
    return traits.get(sign, '獨特')

def get_moon_traits(sign):
    """獲取月亮星座特質"""
    traits = {
        'Aries': '衝動直接的情緒反應',
        'Taurus': '安穩踏實的情感需求',
        'Gemini': '多變靈活的心境',
        'Cancer': '敏感細膩的內心',
        'Leo': '需要被認可的情感',
        'Virgo': '追求完美的內在標準',
        'Libra': '渴望和諧的情緒平衡',
        'Scorpio': '深層強烈的情感體驗',
        'Sagittarius': '自由奔放的心靈',
        'Capricorn': '嚴謹理性的情感控制',
        'Aquarius': '獨特疏離的情感表達',
        'Pisces': '夢幻感性的內心世界'
    }
    return traits.get(sign, '複雜')

def get_career_suggestion(sign):
    """根據太陽星座給出職業建議"""
    suggestions = {
        'Aries': '領導管理和創新開拓',
        'Taurus': '穩定發展和實務操作',
        'Gemini': '溝通表達和多元學習',
        'Cancer': '照護服務和情感支持',
        'Leo': '表演創作和公眾展示',
        'Virgo': '分析整理和品質控制',
        'Libra': '協調平衡和美學設計',
        'Scorpio': '深度研究和心理洞察',
        'Sagittarius': '教育傳播和國際視野',
        'Capricorn': '組織管理和長期規劃',
        'Aquarius': '科技創新和社會改革',
        'Pisces': '藝術創作和心靈療癒'
    }
    return suggestions.get(sign, '多元發展')

# 主應用界面
def main():
    # 主標題
    st.markdown('<h1 class="main-title">🌟 星空職業預言師 🌟</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">如果你是名人，會在哪個領域發光？</p>', unsafe_allow_html=True)
    
    # 側邊欄輸入
    with st.sidebar:
        st.markdown("### 📅 出生資訊")
        
        # 出生日期
        birth_date = st.date_input(
            "選擇出生日期",
            value=date(1990, 1, 1),
            min_value=date(1900, 1, 1),
            max_value=date.today()
        )
        
        # 出生時間
        birth_time = st.time_input(
            "選擇出生時間",
            value=datetime.strptime("12:00", "%H:%M").time()
        )
        
        # 出生地點
        birth_city = st.selectbox(
            "選擇出生城市",
            options=list(CITIES_DATA.keys()),
            index=0
        )
        
        # 計算按鈕
        calculate_btn = st.button("🔮 開始算命", type="primary", use_container_width=True)
    
    # 主要內容區域
    if calculate_btn:
        # 獲取城市數據
        city_data = CITIES_DATA[birth_city]
        lat, lon = city_data["lat"], city_data["lon"]
        
        # 計算儒略日
        jd = calculate_julian_day(
            birth_date.year, birth_date.month, birth_date.day,
            birth_time.hour, birth_time.minute
        )
        
        # 計算行星和宮位
        with st.spinner("🌌 正在計算您的星盤..."):
            planets_data = calculate_planets(jd, lat, lon)
            houses_data = calculate_houses(jd, lat, lon)
        
        # 顯示基本星盤信息
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🌞 個人星盤概覽")
            
            for planet, data in planets_data.items():
                if planet in ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars']:
                    st.markdown(f"""
                    <div class="planet-info">
                        <strong>{planet}:</strong> {data['sign']} {data['degree']:.1f}°
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🌙 外行星配置")
            
            for planet, data in planets_data.items():
                if planet in ['Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']:
                    st.markdown(f"""
                    <div class="planet-info">
                        <strong>{planet}:</strong> {data['sign']} {data['degree']:.1f}°
                    </div>
                    """, unsafe_allow_html=True)
            
            # 上升星座
            asc_data = houses_data.get('ASC', {})
            st.markdown(f"""
            <div class="planet-info">
                <strong>上升星座:</strong> {asc_data.get('sign', 'Unknown')} {asc_data.get('degree', 0):.1f}°
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 職業預測
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎯 職業預測分析")
        
        # 創建特徵並預測
        with st.spinner("🔍 正在分析您的職業潛能..."):
            model = load_model()
            # 使用新的預測函數
            career_predictions = predict_top_careers(planets_data, houses_data, model, top_k=5)d_model()
            features = create_one_hot_features(planets_data, houses_data)
            career_predictions = predict_careers(features, model)
        
        # 顯示預測結果
        st.markdown("#### 🏆 Top 5 職業領域推薦")
        
        # 創建可視化圖表
        categories = [CAREER_CATEGORIES[pred[0]] for pred in career_predictions]
        probabilities = [pred[1] * 100 for pred in career_predictions]
        
        fig = go.Figure(data=[
            go.Bar(
                x=probabilities,
                y=categories,
                orientation='h',
                marker=dict(
                    color=['#FFD700', '#FFA500', '#FF8C00', '#FF7F50', '#FF6347'],
                    line=dict(color='#2c1810', width=2)
                )
            )
        ])
        
        fig.update_layout(
            title="職業適合度分析",
            xaxis_title="適合度 (%)",
            yaxis_title="職業領域",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f4e4bc'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 詳細職業建議
        for i, (category, prob) in enumerate(career_predictions):
            rank_emoji = ["🥇", "🥈", "🥉", "🏅", "⭐"][i]
            st.markdown(f"""
            <div class="career-card">
                <h4>{rank_emoji} 第{i+1}名：{CAREER_CATEGORIES[category]}</h4>
                <p><strong>適合度：</strong>{prob*100:.1f}%</p>
                <p><strong>建議：</strong>這個領域非常適合您的星座配置，建議深入發展相關技能。</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Gemini個人化建議
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🤖 AI個人化建議")
        
        if st.button("🎭 獲取專屬職業建議", type="secondary"):
            with st.spinner("💫 AI正在為您量身定制建議..."):
                gemini_advice = generate_gemini_advice(planets_data, houses_data)
                st.markdown(f'<div class="prediction-text">{gemini_advice}</div>', 
                           unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 星盤可視化
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🌌 星座分布圖")
        
        # 創建星座分布餅圖
        sign_counts = {}
        for planet_data in planets_data.values():
            sign = planet_data['sign']
            sign_counts[sign] = sign_counts.get(sign, 0) + 1
        
        if sign_counts:
            fig_pie = px.pie(
                values=list(sign_counts.values()),
                names=list(sign_counts.keys()),
                title="行星星座分布",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f4e4bc')
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # 歡迎頁面
        st.markdown("""
        <div class="card">
            <h3>🔮 歡迎來到星空職業預言師</h3>
            <p class="prediction-text">
                在左側輸入您的出生資訊，讓星星為您指引職業方向！
                <br><br>
                ✨ <strong>我們提供：</strong>
                <br>• 精準的占星計算分析
                <br>• 基於AI的職業預測
                <br>• 個人化的發展建議
                <br>• 專業的星盤解讀
                <br><br>
                準備好發現您的天賦了嗎？點擊左側開始您的星空之旅！
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()