import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from catboost import CatBoostClassifier
import pickle
import os

# 職業分組策略
class CareerGroupingStrategy:
    def __init__(self):
        self.grouping_map = {
            'Creative_Writing': ['Writers'],
            'Performance_Entertainment': ['Entertainment'],
            'Music_Industry': ['Entertain/Music'],
            'Sports_Athletics': ['Sports', 'Sports Business'],
            'Politics_Government': ['Politics'],
            'Business_Management': ['Business', 'Entertain/Business', 'Business/Marketing'],
            'Education_Research': ['Education', 'Science', 'Humanities+Social Sciences'],
            'Military_Defense': ['Military'],
            'Visual_Arts': ['Art'],
            'Legal_System': ['Law'],
            'Health_Medical': ['Medical', 'Beauty', 'Healing Fields'],
            'Engineering_Tech': ['Engineer', 'Computer', 'Building Trades'],
            'Hospitality_Tourism': ['Food and Beverage', 'Travel'],
            'Religion_Spiritual': ['Religion', 'Occult Fields'],
            'Special_Industries': ['Sex Business', 'Misc.']
        }

# 簡化版CatBoost多標籤分類器
class SimpleCatBoostMultiLabelClassifier:
    def __init__(self, **params):
        self.params = params
        self.estimators_ = []
        self.class_names = []
        
    def fit(self, X, y, class_names=None):
        self.class_names = class_names or [f'Class_{i}' for i in range(y.shape[1])]
        self.estimators_ = []
        
        for i in range(y.shape[1]):
            # 簡化的CatBoost參數
            catboost_params = {
                'iterations': 300,
                'learning_rate': 0.1,
                'depth': 4,
                'verbose': False,
                'random_seed': 42,
                'allow_writing_files': False,
                'auto_class_weights': 'Balanced'
            }
            
            estimator = CatBoostClassifier(**catboost_params)
            estimator.fit(X, y[:, i])
            self.estimators_.append(estimator)
            
        return self
        
    def predict_proba(self, X):
        probabilities = []
        for estimator in self.estimators_:
            proba = estimator.predict_proba(X)
            if proba.shape[1] > 1:
                probabilities.append(proba[:, 1])
            else:
                probabilities.append(np.ones(len(X)) * 0.5)
        return np.array(probabilities).T

def create_dummy_model():
    """創建一個虛擬模型用於演示"""
    class DummyModel:
        def __init__(self):
            self.feature_names = None
            self.class_names = [
                'Creative_Writing', 'Performance_Entertainment', 'Music_Industry',
                'Sports_Athletics', 'Politics_Government', 'Business_Management',
                'Education_Research', 'Military_Defense', 'Visual_Arts',
                'Legal_System', 'Health_Medical', 'Engineering_Tech',
                'Hospitality_Tourism', 'Religion_Spiritual', 'Special_Industries'
            ]
            
        def predict_proba(self, X):
            # 根據輸入特徵生成合理的概率分佈
            n_samples = X.shape[0]
            n_classes = len(self.class_names)
            
            # 使用特徵來影響預測結果
            probabilities = []
            for i in range(n_samples):
                # 基於一些特徵模式生成概率
                probs = np.random.dirichlet(np.ones(n_classes) * 2)
                
                # 根據特徵調整概率 (這裡是簡化邏輯)
                if X[i].sum() > 50:  # 如果特徵值較高
                    # 增加某些職業的概率
                    probs[0] *= 1.5  # Creative_Writing
                    probs[4] *= 1.3  # Politics_Government
                    probs[6] *= 1.4  # Education_Research
                
                # 重新標準化
                probs = probs / probs.sum()
                probabilities.append(probs)
                
            return np.array(probabilities)
    
    return DummyModel()

def save_dummy_model(filepath='best_catboost_model.pkl'):
    """保存虛擬模型"""
    model = create_dummy_model()
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"虛擬模型已保存至 {filepath}")

def download_model_from_cloud(url, local_path, max_retries=3):
    """從雲端下載模型文件"""
    import requests
    from tqdm import tqdm
    
    for attempt in range(max_retries):
        try:
            print(f"正在下載模型... (嘗試 {attempt + 1}/{max_retries})")
            
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="下載進度") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # 如果無法獲取檔案大小，直接寫入
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            print(f"✅ 模型下載完成: {local_path}")
            return True
            
        except Exception as e:
            print(f"❌ 下載失敗 (嘗試 {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                print("正在重試...")
                time.sleep(2)
            else:
                print("所有下載嘗試都失敗了")
                return False
    
    return False

def load_or_create_model(model_path='best_catboost_model.pkl'):
    """載入模型，支援從雲端下載"""
    
    # 首先嘗試從本地載入
    if os.path.exists(model_path):
        try:
            print("正在載入本地模型...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("✅ 成功載入本地預訓練模型")
            return model
        except Exception as e:
            print(f"❌ 載入本地模型失敗: {e}")
            # 刪除損壞的文件
            try:
                os.remove(model_path)
            except:
                pass
    
    # 嘗試從雲端下載
    model_urls = [
        os.getenv('MODEL_DOWNLOAD_URL'),  # 主要 URL
        os.getenv('MODEL_BACKUP_URL'),    # 備用 URL
        # 可以在這裡添加更多備用連結
    ]
    
    for url in model_urls:
        if url:
            try:
                print(f"嘗試從雲端下載模型: {url[:50]}...")
                if download_model_from_cloud(url, model_path):
                    # 下載成功，嘗試載入
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    print("✅ 成功載入雲端預訓練模型")
                    return model
            except Exception as e:
                print(f"❌ 從雲端載入模型失敗: {e}")
                # 清理下載失敗的文件
                try:
                    if os.path.exists(model_path):
                        os.remove(model_path)
                except:
                    pass
    
    print("⚠️ 所有模型載入方式都失敗，使用虛擬模型進行演示")
    return create_dummy_model()

# 特徵工程函數
def create_astrological_features(planets_data, houses_data):
    """創建占星特徵向量"""
    features = {}
    
    # 行星在星座特徵 (10行星 x 12星座 = 120維)
    planets = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
    signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
             'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    
    for planet in planets:
        for sign in signs:
            key = f"{planet}_{sign}"
            if planet in planets_data and planets_data[planet].get('sign') == sign:
                features[key] = 1
            else:
                features[key] = 0
    
    # 上升星座特徵 (12維)
    asc_sign = houses_data.get('ASC', {}).get('sign', 'Aries')
    for sign in signs:
        key = f"ASC_{sign}"
        features[key] = 1 if asc_sign == sign else 0
    
    # 行星度數特徵 (連續值)
    for planet in planets:
        if planet in planets_data:
            features[f"{planet}_degree"] = planets_data[planet].get('degree', 0)
        else:
            features[f"{planet}_degree"] = 0
    
    # 上升點度數
    features['ASC_degree'] = houses_data.get('ASC', {}).get('degree', 0)
    
    # 補齊特徵到345維 (根據你的訓練數據)
    current_features = len(features)
    target_features = 345
    
    for i in range(current_features, target_features):
        features[f"feature_{i}"] = 0
    
    return features

def predict_top_careers(planets_data, houses_data, model=None, top_k=5):
    """預測職業排名"""
    # 創建特徵
    features = create_astrological_features(planets_data, houses_data)
    
    # 轉換為向量
    feature_vector = np.array(list(features.values())).reshape(1, -1)
    
    # 如果沒有模型，使用虛擬模型
    if model is None:
        model = create_dummy_model()
    
    # 預測
    try:
        probabilities = model.predict_proba(feature_vector)[0]
        class_names = getattr(model, 'class_names', [
            'Creative_Writing', 'Performance_Entertainment', 'Music_Industry',
            'Sports_Athletics', 'Politics_Government', 'Business_Management',
            'Education_Research', 'Military_Defense', 'Visual_Arts',
            'Legal_System', 'Health_Medical', 'Engineering_Tech',
            'Hospitality_Tourism', 'Religion_Spiritual', 'Special_Industries'
        ])
        
        # 創建結果列表
        results = list(zip(class_names, probabilities))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
        
    except Exception as e:
        print(f"預測錯誤: {e}")
        # 返回隨機結果
        class_names = [
            'Creative_Writing', 'Performance_Entertainment', 'Music_Industry',
            'Sports_Athletics', 'Politics_Government', 'Business_Management',
            'Education_Research', 'Military_Defense', 'Visual_Arts',
            'Legal_System', 'Health_Medical', 'Engineering_Tech',
            'Hospitality_Tourism', 'Religion_Spiritual', 'Special_Industries'
        ]
        probs = np.random.dirichlet(np.ones(len(class_names)))
        results = list(zip(class_names, probs))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

if __name__ == "__main__":
    # 創建並保存虛擬模型用於測試
    save_dummy_model()