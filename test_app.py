#!/usr/bin/env python3
"""
測試腳本 - 驗證主要功能是否正常運作
在部署前運行此腳本確保所有依賴都已正確安裝
"""

import sys
import importlib

def test_imports():
    """測試所有必要的包是否能正常導入"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'swisseph',
        'plotly',
        'requests',
        'sklearn'
    ]
    
    print("🔍 測試包導入...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError as e:
            print(f"❌ {package} - 失敗: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️ 以下包需要安裝: {', '.join(failed_imports)}")
        print("請運行: pip install -r requirements.txt")
        return False
    else:
        print("\n🎉 所有包導入成功！")
        return True

def test_swisseph():
    """測試 Swiss Ephemeris 功能"""
    print("\n🌌 測試占星計算...")
    try:
        import swisseph as swe
        
        # 測試基本計算
        jd = swe.julday(2000, 1, 1, 0)
        result = swe.calc_ut(jd, swe.SUN, swe.FLG_SWIEPH)
        
        if isinstance(result, tuple) and len(result) >= 1:
            print(f"✅ Swiss Ephemeris 計算成功")
            print(f"   太陽位置: {result[0][0]:.2f}°")
            return True
        else:
            print("❌ Swiss Ephemeris 計算結果格式不正確")
            return False
            
    except Exception as e:
        print(f"❌ Swiss Ephemeris 測試失敗: {e}")
        return False

def test_model_utils():
    """測試模型工具模塊"""
    print("\n🤖 測試模型功能...")
    try:
        from model_utils import create_dummy_model, create_astrological_features
        
        # 測試虛擬模型
        model = create_dummy_model()
        print("✅ 虛擬模型創建成功")
        
        # 測試特徵創建
        planets_data = {
            'Sun': {'sign': 'Leo', 'degree': 15.5},
            'Moon': {'sign': 'Cancer', 'degree': 22.3}
        }
        houses_data = {
            'ASC': {'sign': 'Scorpio', 'degree': 5.8}
        }
        
        features = create_astrological_features(planets_data, houses_data)
        print(f"✅ 特徵向量創建成功，維度: {len(features)}")
        
        # 測試預測
        import numpy as np
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        predictions = model.predict_proba(feature_vector)
        print(f"✅ 模型預測成功，輸出形狀: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型功能測試失敗: {e}")
        return False

def test_gemini_integration():
    """測試 Gemini 集成"""
    print("\n🌟 測試 Gemini 集成...")
    try:
        from gemini_integration import GeminiAdvisor
        
        advisor = GeminiAdvisor()  # 不提供 API 密鑰，會使用備用方案
        
        planets_data = {
            'Sun': {'sign': 'Leo', 'degree': 15.5},
            'Moon': {'sign': 'Cancer', 'degree': 22.3}
        }
        houses_data = {
            'ASC': {'sign': 'Scorpio', 'degree': 5.8}
        }
        
        advice = advisor.get_advice(planets_data, houses_data)
        
        if advice and len(advice) > 50:  # 確保返回了合理長度的建議
            print("✅ Gemini 集成測試成功 (使用備用建議)")
            print(f"   建議長度: {len(advice)} 字符")
            return True
        else:
            print("❌ Gemini 集成返回的建議過短")
            return False
            
    except Exception as e:
        print(f"❌ Gemini 集成測試失敗: {e}")
        return False

def test_plotly_charts():
    """測試 Plotly 圖表功能"""
    print("\n📊 測試圖表功能...")
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
        # 測試創建簡單圖表
        fig = go.Figure(data=[go.Bar(x=['A', 'B', 'C'], y=[1, 2, 3])])
        fig.update_layout(title="測試圖表")
        
        print("✅ Plotly 圖表創建成功")
        return True
        
    except Exception as e:
        print(f"❌ Plotly 圖表測試失敗: {e}")
        return False

def run_all_tests():
    """運行所有測試"""
    print("🚀 開始運行功能測試...\n")
    
    tests = [
        ("包導入", test_imports),
        ("占星計算", test_swisseph),
        ("模型功能", test_model_utils),
        ("Gemini集成", test_gemini_integration),
        ("圖表功能", test_plotly_charts)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 測試出現異常: {e}")
            results.append((test_name, False))
    
    # 總結
    print("\n" + "="*50)
    print("📋 測試結果總結:")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{test_name:15} - {status}")
        if result:
            passed += 1
    
    print(f"\n總計: {passed}/{total} 項測試通過")
    
    if passed == total:
        print("\n🎉 所有測試通過！應用可以正常部署。")
        return True
    else:
        print(f"\n⚠️ 有 {total - passed} 項測試失敗，請檢查相關配置。")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)