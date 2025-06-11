#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試所有功能的腳本
執行這個腳本來檢查你的應用是否正常工作
"""

import sys
import os
from datetime import date, time

def test_imports():
    """測試所有必要的模組導入"""
    print("🔍 測試模組導入...")
    
    try:
        import streamlit as st
        print("✅ Streamlit 導入成功")
    except ImportError as e:
        print(f"❌ Streamlit 導入失敗: {e}")
        return False
    
    try:
        import swisseph as swe
        print("✅ Swiss Ephemeris 導入成功")
    except ImportError as e:
        print(f"❌ Swiss Ephemeris 導入失敗: {e}")
        print("💡 請執行: pip install pyswisseph")
        return False
    
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI 導入成功")
    except ImportError as e:
        print(f"❌ Google Generative AI 導入失敗: {e}")
        print("💡 請執行: pip install google-generativeai")
        return False
    
    try:
        from professional_astro_utils import calculate_professional_chart
        print("✅ 專業占星計算模組導入成功")
    except ImportError as e:
        print(f"❌ 專業占星計算模組導入失敗: {e}")
        return False
    
    try:
        from model_utils import load_or_create_model, predict_top_careers
        print("✅ 模型工具導入成功")
    except ImportError as e:
        print(f"❌ 模型工具導入失敗: {e}")
        return False
    
    try:
        from gemini_integration import get_career_advice
        print("✅ Gemini 整合模組導入成功")
    except ImportError as e:
        print(f"❌ Gemini 整合模組導入失敗: {e}")
        return False
    
    return True

def test_astro_calculation():
    """測試占星計算功能"""
    print("\n🔮 測試占星計算...")
    
    try:
        from professional_astro_utils import calculate_professional_chart
        
        # 測試數據
        test_date = date(1995, 6, 15)
        test_time = time(14, 30)
        test_place = "台北市"
        
        chart = calculate_professional_chart(test_date, test_time, test_place)
        
        if 'error' in chart:
            print(f"❌ 占星計算失敗: {chart['error']}")
            return False
        
        # 檢查必要的數據
        if 'planets' not in chart or 'houses' not in chart:
            print("❌ 占星計算結果不完整")
            return False
        
        # 檢查行星數據
        planets = chart['planets']
        if len(planets) < 10:
            print(f"⚠️ 只計算了 {len(planets)} 個行星（應該有10個）")
        
        # 顯示一些結果
        print(f"✅ 占星計算成功！")
        print(f"📍 坐標: {chart.get('coordinates', '未知')}")
        print(f"⏰ UTC時間: {chart.get('utc_time', '未知')}")
        
        # 顯示太陽位置
        if 'Sun' in planets:
            sun = planets['Sun']
            print(f"☉ 太陽: {sun.get('sign', '未知')} {sun.get('sign_degree', 0):.1f}°")
        
        return True
        
    except Exception as e:
        print(f"❌ 占星計算測試失敗: {e}")
        return False

def test_model_utils():
    """測試機器學習模型"""
    print("\n🤖 測試機器學習模型...")
    
    try:
        from model_utils import load_or_create_model, predict_top_careers
        
        # 載入模型
        model = load_or_create_model()
        print("✅ 模型載入成功")
        
        # 測試預測
        test_planets = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
        test_houses = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
        
        predictions = predict_top_careers(test_planets, test_houses, model, top_k=3, random_state=42)
        
        if len(predictions) == 3:
            print("✅ 職業預測成功")
            for i, (career, prob) in enumerate(predictions, 1):
                print(f"   {i}. {career}: {prob:.1%}")
            return True
        else:
            print("❌ 職業預測結果不正確")
            return False
            
    except Exception as e:
        print(f"❌ 模型測試失敗: {e}")
        return False

def test_gemini_api():
    """測試 Gemini API 連接"""
    print("\n🤖 測試 Gemini API...")
    
    try:
        from gemini_integration import get_api_status, get_career_advice
        
        # 檢查 API 狀態
        status = get_api_status()
        print(f"API 狀態: {status}")
        
        if "✅" in status:
            print("✅ Gemini API 連接成功")
            
            # 測試簡單的建議生成
            test_chart = {
                'planets': {'太陽 ☉': '獅子座 ♌'},
                'top_careers': ['測試職業'],
                'birth_info': {'date': '1995-06-15', 'time': '14:30', 'place': '台北市'}
            }
            
            advice = get_career_advice(test_chart)
            
            if advice and not advice.startswith('❌'):
                print("✅ AI 建議生成成功")
                print(f"建議長度: {len(advice)} 字元")
                return True
            else:
                print(f"⚠️ AI 建議生成有問題: {advice[:100]}...")
                return False
        else:
            print("⚠️ Gemini API 未正確設置")
            print("💡 請確認 GEMINI_API_KEY 環境變數或 .env 文件")
            return False
            
    except Exception as e:
        print(f"❌ Gemini API 測試失敗: {e}")
        return False

def test_env_setup():
    """檢查環境設置"""
    print("\n⚙️ 檢查環境設置...")
    
    # 檢查 .env 文件
    if os.path.exists('.env'):
        print("✅ 找到 .env 文件")
        
        # 檢查 API Key
        try:
            with open('.env', 'r') as f:
                content = f.read()
                if 'GEMINI_API_KEY' in content:
                    print("✅ .env 文件包含 GEMINI_API_KEY")
                else:
                    print("⚠️ .env 文件缺少 GEMINI_API_KEY")
        except Exception as e:
            print(f"⚠️ 無法讀取 .env 文件: {e}")
    else:
        print("⚠️ 未找到 .env 文件")
    
    # 檢查環境變數
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print(f"✅ 環境變數 GEMINI_API_KEY 已設置 (長度: {len(api_key)})")
    else:
        print("⚠️ 環境變數 GEMINI_API_KEY 未設置")
    
    return True

def main():
    """主測試函數"""
    print("🚀 開始測試星空職業預言師應用...")
    print("=" * 50)
    
    tests = [
        ("環境設置", test_env_setup),
        ("模組導入", test_imports),
        ("占星計算", test_astro_calculation),
        ("機器學習模型", test_model_utils),
        ("Gemini API", test_gemini_api)
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
    print("\n" + "=" * 50)
    print("📊 測試結果總結:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 總體結果: {passed}/{len(results)} 項測試通過")
    
    if passed == len(results):
        print("🎉 恭喜！所有測試都通過了，你的應用應該可以正常運行！")
        print("💡 現在可以執行: streamlit run app.py")
    else:
        print("⚠️ 部分測試失敗，請根據上述信息修復問題")
        
        if passed >= 3:
            print("💡 基本功能可用，可以嘗試運行應用，但某些功能可能不正常")

if __name__ == "__main__":
    main()