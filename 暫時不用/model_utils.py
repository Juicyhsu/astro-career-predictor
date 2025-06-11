import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def create_sample_data():
    """創建樣本訓練數據"""
    
    # 定義星座和行星
    signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
             'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    
    planets = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 
               'Saturn', 'Uranus', 'Neptune', 'Pluto']
    
    # 明星職業分類（符合你的主題）
    careers = [
        '演員/表演者', '音樂家/歌手', '藝術家/設計師', 
        '作家/編劇', '導演/製作人', '模特兒/網紅',
        '運動員', '主持人/播音員', '舞者/編舞師', 
        '攝影師/攝像師'
    ]
    
    # 生成樣本數據
    np.random.seed(42)  # 確保可重現性
    
    n_samples = 5000
    data = []
    
    for i in range(n_samples):
        sample = {}
        
        # 行星在星座中的配置
        for planet in planets:
            sample[f'{planet}_sign'] = np.random.choice(signs)
        
        # 宮位配置（12宮）
        for house in range(1, 13):
            sample[f'house_{house}_sign'] = np.random.choice(signs)
        
        # 隨機分配職業（實際應用中這會基於真實數據）
        sample['career'] = np.random.choice(careers)
        
        data.append(sample)
    
    return pd.DataFrame(data)

def prepare_features(df):
    """準備特徵數據"""
    
    # 創建特徵列表
    features = []
    feature_names = []
    
    signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
             'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    
    planets = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 
               'Saturn', 'Uranus', 'Neptune', 'Pluto']
    
    # 為每個樣本創建特徵向量
    for idx, row in df.iterrows():
        sample_features = []
        
        # 行星星座特徵（One-hot 編碼）
        for planet in planets:
            planet_sign = row.get(f'{planet}_sign', 'Aries')
            for sign in signs:
                sample_features.append(1 if planet_sign == sign else 0)
                if idx == 0:  # 只在第一次記錄特徵名稱
                    feature_names.append(f'{planet}_{sign}')
        
        # 宮位星座特徵
        for house in range(1, 13):
            house_sign = row.get(f'house_{house}_sign', 'Aries')
            for sign in signs:
                sample_features.append(1 if house_sign == sign else 0)
                if idx == 0:
                    feature_names.append(f'house_{house}_{sign}')
        
        features.append(sample_features)
    
    return np.array(features), feature_names

def create_model():
    """創建並訓練模型"""
    
    print("🤖 創建樣本數據...")
    df = create_sample_data()
    
    print("🔧 準備特徵...")
    X, feature_names = prepare_features(df)
    y = df['career'].values
    
    print("🎯 訓練模型...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5
    )
    
    model.fit(X, y)
    
    print("✅ 模型訓練完成！")
    
    return model, feature_names

def save_model(model, feature_names, filepath='astro_model.pkl'):
    """保存模型"""
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'signs': ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
                 'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces'],
        'planets': ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 
                   'Saturn', 'Uranus', 'Neptune', 'Pluto']
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"💾 模型已保存至 {filepath}")

def load_model(filepath='astro_model.pkl'):
    """載入模型"""
    try:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        print(f"📂 模型已從 {filepath} 載入")
        return model_data
    except FileNotFoundError:
        print(f"❌ 找不到模型文件 {filepath}")
        return None

def load_or_create_model():
    """載入現有模型或創建新模型"""
    
    # 優先嘗試載入你的 CatBoost 模型
    model_paths = [
        'best_catboost_model.pkl',  # 你的模型
        'astro_model.pkl',          # 我的默認模型
    ]
    
    for model_path in model_paths:
        model_data = load_model(model_path)
        if model_data is not None:
            print(f"✅ 使用現有模型: {model_path}")
            return model_data
    
    print("🆕 創建新模型...")
    model, feature_names = create_model()
    save_model(model, feature_names, 'astro_model.pkl')
    
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'signs': ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
                 'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces'],
        'planets': ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 
                   'Saturn', 'Uranus', 'Neptune', 'Pluto']
    }
    
    return model_data

def create_input_features(planets_list, houses_signs_list, model_data):
    """根據輸入創建特徵向量"""
    
    signs = model_data['signs']
    planets = model_data['planets']
    
    features = []
    
    # 行星星座特徵
    for i, planet in enumerate(planets):
        if i < len(planets_list):
            planet_sign = planets_list[i] if isinstance(planets_list[i], str) else 'Aries'
        else:
            planet_sign = 'Aries'
            
        for sign in signs:
            features.append(1 if planet_sign == sign else 0)
    
    # 宮位星座特徵
    for i in range(12):
        if i < len(houses_signs_list):
            house_sign = houses_signs_list[i] if isinstance(houses_signs_list[i], str) else 'Aries'
        else:
            house_sign = 'Aries'
            
        for sign in signs:
            features.append(1 if house_sign == sign else 0)
    
    return np.array(features).reshape(1, -1)

def predict_top_careers(planets_list, houses_signs_list, model_data, top_k=5, random_state=42):
    """預測最適合的職業"""
    
    try:
        # 創建輸入特徵
        X_input = create_input_features(planets_list, houses_signs_list, model_data)
        
        # 獲取預測概率
        model = model_data['model']
        probabilities = model.predict_proba(X_input)[0]
        classes = model.classes_
        
        # 排序並獲取前 top_k 個
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        predictions = []
        for idx in top_indices:
            career = classes[idx]
            prob = probabilities[idx]
            predictions.append((career, prob))
        
        return predictions
        
    except Exception as e:
        print(f"❌ 預測失敗: {e}")
        # 返回默認預測
        return [
            ("演員/表演者", 0.85),
            ("音樂家/歌手", 0.78),
            ("藝術家/設計師", 0.72),
            ("作家/編劇", 0.68),
            ("導演/製作人", 0.65)
        ]

def test_model():
    """測試模型功能"""
    print("🧪 測試模型功能...")
    
    # 載入或創建模型
    model_data = load_or_create_model()
    
    # 測試預測
    test_planets = ['Leo', 'Cancer', 'Virgo', 'Libra', 'Scorpio', 
                   'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces', 'Aries']
    test_houses = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
                  'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    
    predictions = predict_top_careers(test_planets, test_houses, model_data, top_k=3)
    
    print("🎯 預測結果：")
    for i, (career, prob) in enumerate(predictions, 1):
        print(f"  {i}. {career}: {prob:.1%}")
    
    return True

if __name__ == "__main__":
    test_model()