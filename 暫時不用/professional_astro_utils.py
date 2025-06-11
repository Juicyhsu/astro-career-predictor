import pandas as pd
import numpy as np
import swisseph as swe
import re
from datetime import datetime
import pytz
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import logging

# 設定簡化的日誌
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger()

# 常數定義 - 使用你的程式碼
SIGN_NAMES = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
              'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']

PLANET_NAMES = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 
                'Saturn', 'Uranus', 'Neptune', 'Pluto']

PLANET_IDS = {
    'Sun': swe.SUN,
    'Moon': swe.MOON,
    'Mercury': swe.MERCURY,
    'Venus': swe.VENUS,
    'Mars': swe.MARS,
    'Jupiter': swe.JUPITER,
    'Saturn': swe.SATURN,
    'Uranus': swe.URANUS,
    'Neptune': swe.NEPTUNE,
    'Pluto': swe.PLUTO
}

# 台灣城市坐標 - 擴展版
TAIWAN_CITY_COORDS = {
    "台北市": (25.0330, 121.5654),
    "新北市": (25.0092, 121.4589),
    "桃園市": (24.9936, 121.3010),
    "台中市": (24.1477, 120.6736),
    "台南市": (22.9908, 120.2133),
    "高雄市": (22.6273, 120.3014),
    "基隆市": (25.1276, 121.7392),
    "新竹市": (24.8138, 120.9675),
    "新竹縣": (24.7038, 121.0381),
    "苗栗縣": (24.5601, 120.8204),
    "彰化縣": (24.0518, 120.5161),
    "南投縣": (23.9609, 120.9718),
    "雲林縣": (23.7092, 120.4313),
    "嘉義市": (23.4801, 120.4491),
    "嘉義縣": (23.4518, 120.2554),
    "屏東縣": (22.5519, 120.5487),
    "宜蘭縣": (24.7021, 121.7378),
    "花蓮縣": (23.9871, 121.6015),
    "台東縣": (22.7972, 121.1713),
    "澎湖縣": (23.5712, 119.5793),
    "金門縣": (24.4324, 118.3177),
    "連江縣": (26.1609, 119.9300)
}

def get_sign_name(longitude):
    """獲取星座名稱"""
    sign_num = int(longitude / 30) % 12
    return SIGN_NAMES[sign_num]

def get_sign_degree(longitude):
    """獲取星座內的度數"""
    return longitude % 30

def get_coordinates_from_location(city, country=None):
    """
    獲取城市坐標 - 簡化版
    """
    # 優先檢查台灣城市
    if city in TAIWAN_CITY_COORDS:
        return TAIWAN_CITY_COORDS[city]
    
    # 硬編碼的國際主要城市
    international_cities = {
        "tokyo": (35.6762, 139.6503),
        "new york": (40.7128, -74.0060),
        "london": (51.5074, -0.1278),
        "paris": (48.8566, 2.3522),
        "seoul": (37.5665, 126.9780),
        "singapore": (1.3521, 103.8198),
        "sydney": (-33.8688, 151.2093),
        "los angeles": (34.0522, -118.2437),
        "chicago": (41.8781, -87.6298),
        "berlin": (52.5200, 13.4050),
        "madrid": (40.4168, -3.7038),
        "rome": (41.9028, 12.4964),
        "amsterdam": (52.3676, 4.9041),
        "moscow": (55.7558, 37.6176),
        "beijing": (39.9042, 116.4074),
        "shanghai": (31.2304, 121.4737),
        "hong kong": (22.3193, 114.1694),
        "bangkok": (13.7563, 100.5018),
        "manila": (14.5995, 120.9842),
        "kuala lumpur": (3.1390, 101.6869),
        "jakarta": (-6.2088, 106.8456),
        "mumbai": (19.0760, 72.8777),
        "delhi": (28.7041, 77.1025),
        "dubai": (25.2048, 55.2708)
    }
    
    city_lower = city.lower() if city else ""
    
    # 檢查國際城市
    for int_city, coords in international_cities.items():
        if int_city in city_lower or city_lower in int_city:
            return coords
    
    # 如果都找不到，嘗試使用 geopy
    try:
        geolocator = Nominatim(user_agent="astro_app", timeout=5)
        query = f"{city}, {country}" if country else city
        location = geolocator.geocode(query)
        if location:
            return (location.latitude, location.longitude)
    except:
        pass
    
    # 默認返回台北
    return (25.0330, 121.5654)

def get_timezone_by_location(city, country=None):
    """
    根據地點獲取時區
    """
    city_timezones = {
        # 台灣
        "台北": "Asia/Taipei", "台中": "Asia/Taipei", "高雄": "Asia/Taipei",
        "台南": "Asia/Taipei", "新北": "Asia/Taipei", "桃園": "Asia/Taipei",
        
        # 國際主要城市
        "tokyo": "Asia/Tokyo", "seoul": "Asia/Seoul", "beijing": "Asia/Shanghai",
        "shanghai": "Asia/Shanghai", "hong kong": "Asia/Hong_Kong",
        "singapore": "Asia/Singapore", "bangkok": "Asia/Bangkok",
        "new york": "America/New_York", "los angeles": "America/Los_Angeles",
        "chicago": "America/Chicago", "london": "Europe/London",
        "paris": "Europe/Paris", "berlin": "Europe/Berlin",
        "madrid": "Europe/Madrid", "rome": "Europe/Rome",
        "moscow": "Europe/Moscow", "sydney": "Australia/Sydney",
        "melbourne": "Australia/Melbourne"
    }
    
    city_lower = city.lower() if city else ""
    
    # 檢查台灣城市
    for tw_city in TAIWAN_CITY_COORDS.keys():
        if tw_city in city or city in tw_city:
            return "Asia/Taipei"
    
    # 檢查國際城市
    for tz_city, tz in city_timezones.items():
        if tz_city in city_lower or city_lower in tz_city:
            return tz
    
    # 默認時區
    return "Asia/Taipei"

def convert_to_utc(year, month, day, hour, minute, city, country=None):
    """
    將當地時間轉換為 UTC
    """
    try:
        # 創建當地時間
        local_dt = datetime(year, month, day, hour, minute)
        
        # 獲取時區
        tz_name = get_timezone_by_location(city, country)
        local_tz = pytz.timezone(tz_name)
        
        # 本地化時間
        localized_dt = local_tz.localize(local_dt)
        
        # 轉換為 UTC
        utc_dt = localized_dt.astimezone(pytz.UTC)
        
        return (utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour, utc_dt.minute)
    
    except Exception:
        # 備用方法：簡單的時區偏移
        lat, lon = get_coordinates_from_location(city, country)
        offset_hours = round(lon / 15.0)  # 粗略計算
        utc_hour = hour - offset_hours
        
        # 處理日期變更
        utc_day = day
        if utc_hour < 0:
            utc_hour += 24
            utc_day -= 1
        elif utc_hour >= 24:
            utc_hour -= 24
            utc_day += 1
            
        return (year, month, utc_day, utc_hour, minute)

def calculate_professional_chart(birth_date, birth_time, birth_place):
    """
    使用 Swiss Ephemeris 計算專業星盤
    
    Args:
        birth_date: datetime.date 對象
        birth_time: time 對象  
        birth_place: 出生地字符串
    
    Returns:
        dict: 包含行星和宮位資訊的字典
    """
    try:
        # 解析出生資訊
        year = birth_date.year
        month = birth_date.month
        day = birth_date.day
        hour = birth_time.hour
        minute = birth_time.minute
        
        # 獲取坐標
        lat, lon = get_coordinates_from_location(birth_place)
        
        # 轉換為 UTC
        ut_year, ut_month, ut_day, ut_hour, ut_minute = convert_to_utc(
            year, month, day, hour, minute, birth_place
        )
        
        # 計算儒略日
        jd_ut = swe.julday(ut_year, ut_month, ut_day, ut_hour + ut_minute/60.0)
        
        # 計算設定
        flags = swe.FLG_SWIEPH | swe.FLG_SPEED
        
        # 計算行星位置
        planets_data = {}
        for planet_name, planet_id in PLANET_IDS.items():
            try:
                result = swe.calc_ut(jd_ut, planet_id, flags)
                if isinstance(result, tuple) and len(result) >= 2:
                    positions = result[0]
                else:
                    positions = result
                
                longitude = positions[0] if len(positions) > 0 else 0
                speed = positions[3] if len(positions) > 3 else 0
                
                planets_data[planet_name] = {
                    'longitude': longitude,
                    'sign': get_sign_name(longitude),
                    'sign_degree': get_sign_degree(longitude),
                    'retrograde': speed < 0
                }
                
            except Exception as e:
                # 如果計算失敗，使用默認值
                planets_data[planet_name] = {
                    'longitude': 0,
                    'sign': 'Aries',
                    'sign_degree': 0,
                    'retrograde': False,
                    'error': str(e)
                }
        
        # 計算宮位 - 使用 Placidus 宮位系統
        houses_data = {}
        try:
            house_result = swe.houses(jd_ut, lat, lon, b'P')  # Placidus
            
            if isinstance(house_result, tuple) and len(house_result) >= 2:
                cusps = house_result[0]
                ascmc = house_result[1]
                
                # 宮位分界點
                for i in range(1, 13):
                    if i < len(cusps):
                        cusp_lon = cusps[i]
                        houses_data[i] = {
                            'longitude': cusp_lon,
                            'sign': get_sign_name(cusp_lon),
                            'sign_degree': get_sign_degree(cusp_lon)
                        }
                
                # 上升點和天頂
                if len(ascmc) >= 2:
                    asc_lon = ascmc[0]
                    mc_lon = ascmc[1]
                    
                    houses_data['ASC'] = {
                        'longitude': asc_lon,
                        'sign': get_sign_name(asc_lon),
                        'sign_degree': get_sign_degree(asc_lon)
                    }
                    
                    houses_data['MC'] = {
                        'longitude': mc_lon,
                        'sign': get_sign_name(mc_lon),
                        'sign_degree': get_sign_degree(mc_lon)
                    }
                    
        except Exception as e:
            # 如果宮位計算失敗，創建默認宮位
            for i in range(1, 13):
                houses_data[i] = {
                    'longitude': (i-1) * 30,
                    'sign': SIGN_NAMES[(i-1) % 12],
                    'sign_degree': 0,
                    'error': str(e)
                }
        
        # 計算行星所在宮位
        if houses_data and 1 in houses_data:
            for planet_name, planet_data in planets_data.items():
                planet_lon = planet_data['longitude']
                planet_house = calculate_planet_house(planet_lon, houses_data)
                planets_data[planet_name]['house'] = planet_house
        
        return {
            'planets': planets_data,
            'houses': houses_data,
            'coordinates': (lat, lon),
            'utc_time': f"{ut_year}-{ut_month:02d}-{ut_day:02d} {ut_hour:02d}:{ut_minute:02d}",
            'julian_day': jd_ut
        }
        
    except Exception as e:
        # 如果完全失敗，返回錯誤信息
        return {
            'error': f"星盤計算失敗: {str(e)}",
            'planets': {},
            'houses': {}
        }

def calculate_planet_house(planet_longitude, houses_data):
    """
    計算行星所在宮位
    """
    planet_lon = planet_longitude % 360.0
    
    for house_num in range(1, 13):
        if house_num in houses_data and house_num + 1 in houses_data:
            start_cusp = houses_data[house_num]['longitude'] % 360.0
            if house_num == 12:
                end_cusp = houses_data[1]['longitude'] % 360.0
            else:
                end_cusp = houses_data[house_num + 1]['longitude'] % 360.0
            
            # 檢查行星是否在此宮位內
            if start_cusp < end_cusp:
                if start_cusp <= planet_lon < end_cusp:
                    return house_num
            else:  # 宮位跨越0度
                if start_cusp <= planet_lon < 360.0 or 0.0 <= planet_lon < end_cusp:
                    return house_num
    
    return 1  # 默認第一宮

def get_house_rulers(houses_data):
    """
    獲取宮位守護星
    """
    # 傳統守護星對應表
    rulers = {
        'Aries': 'Mars', 'Taurus': 'Venus', 'Gemini': 'Mercury',
        'Cancer': 'Moon', 'Leo': 'Sun', 'Virgo': 'Mercury',
        'Libra': 'Venus', 'Scorpio': 'Mars', 'Sagittarius': 'Jupiter',
        'Capricorn': 'Saturn', 'Aquarius': 'Saturn', 'Pisces': 'Jupiter'
    }
    
    house_rulers = {}
    for house_num in range(1, 13):
        if house_num in houses_data:
            house_sign = houses_data[house_num]['sign']
            house_rulers[house_num] = rulers.get(house_sign, 'Unknown')
    
    return house_rulers

def test_professional_calculation():
    """
    測試專業計算函數
    """
    try:
        from datetime import date, time
        
        # 測試數據
        test_date = date(1995, 6, 15)
        test_time = time(14, 30)
        test_place = "台北市"
        
        print("🧪 測試專業占星計算...")
        chart = calculate_professional_chart(test_date, test_time, test_place)
        
        if 'error' in chart:
            print(f"❌ 測試失敗: {chart['error']}")
            return False
        
        print("✅ 測試成功！")
        print(f"📍 坐標: {chart['coordinates']}")
        print(f"⏰ UTC時間: {chart['utc_time']}")
        
        # 顯示前3個行星
        for i, (planet, data) in enumerate(list(chart['planets'].items())[:3]):
            if 'error' not in data:
                print(f"🪐 {planet}: {data['sign']} {data['sign_degree']:.1f}°")
            if i >= 2:
                break
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

if __name__ == "__main__":
    test_professional_calculation()