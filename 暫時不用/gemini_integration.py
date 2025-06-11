import os
import google.generativeai as genai
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

def get_api_status():
    """檢查 Gemini API 狀態"""
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        return "❌ GEMINI_API_KEY 未設置"
    
    try:
        genai.configure(api_key=api_key)
        
        # 嘗試不同的模型名稱
        model_names = [
            'gemini-1.5-flash',
            'gemini-1.5-pro', 
            'gemini-pro',
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro'
        ]
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("測試")
                return f"✅ Gemini API 連接成功 (模型: {model_name}, API Key 長度: {len(api_key)})"
            except Exception as model_error:
                if "not found" in str(model_error):
                    continue
                else:
                    raise model_error
        
        # 如果所有模型都失敗，列出可用模型
        try:
            models = genai.list_models()
            available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
            return f"❌ 常用模型不可用。可用模型: {available_models[:3]}"
        except:
            return "❌ 無法獲取模型列表"
    
    except Exception as e:
        return f"❌ Gemini API 連接失敗: {str(e)}"

def get_career_advice(chart_info):
    """
    使用 Gemini API 生成職業建議
    
    Args:
        chart_info: 包含星盤資訊的字典
    
    Returns:
        str: AI 生成的職業建議
    """
    
    # 檢查 API Key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return "❌ 請先設置 GEMINI_API_KEY 環境變數"
    
    try:
        # 配置 Gemini API
        genai.configure(api_key=api_key)
        
        # 按優先級嘗試不同模型
        model_names = [
            'gemini-1.5-flash',  # 最新且快速
            'gemini-1.5-pro',    # 最新專業版
            'gemini-pro',        # 舊版
        ]
        
        model = None
        working_model = None
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                # 測試模型是否可用
                test_response = model.generate_content("測試", 
                    generation_config=genai.types.GenerationConfig(
                        candidate_count=1,
                        max_output_tokens=50,
                        temperature=0.1,
                    ))
                working_model = model_name
                break
            except Exception as e:
                if "not found" in str(e):
                    continue
                else:
                    raise e
        
        if not model:
            return "❌ 沒有可用的 Gemini 模型"
        
        # 構建詳細的 prompt
        prompt = f"""
你是一位專業的占星師，擅長結合傳統占星學與現代職業分析。請根據以下詳細的出生星盤資訊，為用戶提供深度的明星潛質分析。

**出生資訊：**
- 出生日期：{chart_info.get('birth_info', {}).get('date', '未知')}
- 出生時間：{chart_info.get('birth_info', {}).get('time', '未知')}
- 出生地點：{chart_info.get('birth_info', {}).get('place', '未知')}

**行星配置：**
"""
        
        # 添加行星資訊
        if 'planets' in chart_info:
            for planet, info in chart_info['planets'].items():
                if isinstance(info, dict):
                    retrograde = " (逆行)" if info.get('retrograde', False) else ""
                    prompt += f"- {planet}：{info.get('sign', '未知')} {info.get('degree', 0):.1f}° 第{info.get('house', '未知')}宮{retrograde}\n"
                else:
                    prompt += f"- {planet}：{info}\n"
        
        # 添加上升星座
        if 'ascendant' in chart_info:
            asc = chart_info['ascendant']
            prompt += f"\n**上升星座：** {asc.get('sign', '未知')} {asc.get('degree', 0):.1f}°\n"
        
        # 添加宮位資訊
        if 'houses' in chart_info:
            prompt += "\n**宮位配置：**\n"
            for house, sign in chart_info['houses'].items():
                prompt += f"- {house}：{sign}\n"
        
        # 添加 AI 預測結果
        if 'top_careers' in chart_info:
            prompt += f"\n**AI 預測的前五個發光領域：**\n"
            for i, career in enumerate(chart_info['top_careers'][:5], 1):
                prompt += f"{i}. {career}\n"
        
        prompt += """

請根據以上星盤資訊，提供一份深度的明星潛質分析報告，包含以下內容：

1. **整體明星潛質評估**：分析這個人的天生明星特質
2. **最適合的發光領域**：結合 AI 預測，詳細解釋為什麼這些領域最適合
3. **個人特色與優勢**：基於行星配置分析獨特魅力
4. **發展建議**：具體的能力培養和發展方向
5. **注意事項**：需要克服的挑戰或弱點

請用溫暖、鼓勵且專業的語調，讓用戶感受到自己的獨特價值。字數控制在 800-1200 字之間。
"""
        
        # 生成回應 - 使用更穩定的配置
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=2000,
                temperature=0.7,
            )
        )
        
        if response and response.text:
            return f"[使用模型: {working_model}]\n\n{response.text}"
        else:
            return "❌ AI 回應為空，請稍後再試"
            
    except Exception as e:
        error_msg = str(e)
        
        # 提供更詳細的錯誤資訊
        if "API_KEY" in error_msg:
            return "❌ API Key 錯誤，請檢查 GEMINI_API_KEY 是否正確"
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            return "❌ API 配額已用完，請稍後再試或檢查計費設定"
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            return "❌ 網路連線問題，請檢查網路設定"
        elif "not found" in error_msg.lower():
            return "❌ 模型不可用，Google 可能已更新 API"
        else:
            return f"❌ Gemini API 錯誤: {error_msg}"

def list_available_models():
    """列出所有可用的模型"""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return "❌ API Key 未設置"
        
        genai.configure(api_key=api_key)
        models = genai.list_models()
        
        print("📋 可用的 Gemini 模型：")
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                print(f"✅ {model.name}")
            else:
                print(f"⚠️ {model.name} (不支援 generateContent)")
                
    except Exception as e:
        print(f"❌ 無法列出模型: {e}")

def test_gemini_connection():
    """測試 Gemini 連接"""
    print("🧪 測試 Gemini API 連接...")
    
    # 檢查環境變數
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY 未設置")
        print("💡 請在 .env 文件中設置: GEMINI_API_KEY=你的API密鑰")
        return False
    
    print(f"✅ API Key 已設置 (長度: {len(api_key)})")
    
    # 列出可用模型
    print("\n🔍 檢查可用模型...")
    list_available_models()
    
    # 測試 API 調用
    print("\n🧪 測試 API 調用...")
    try:
        genai.configure(api_key=api_key)
        
        # 按優先級測試模型
        model_names = [
            'gemini-2.0-flash',
            'gemini-2.0-pro-exp', 
            'gemini-pro'
        ]
        
        for model_name in model_names:
            try:
                print(f"🔄 測試模型: {model_name}")
                model = genai.GenerativeModel(model_name)
                
                response = model.generate_content(
                    "請說：測試成功",
                    generation_config=genai.types.GenerationConfig(
                        candidate_count=1,
                        max_output_tokens=50,
                        temperature=0.1,
                    )
                )
                
                if response and response.text:
                    print(f"✅ 模型 {model_name} 測試成功")
                    print(f"📝 回應: {response.text}")
                    return True
                    
            except Exception as model_error:
                print(f"❌ 模型 {model_name} 失敗: {model_error}")
                continue
        
        print("❌ 所有模型都無法使用")
        return False
            
    except Exception as e:
        print(f"❌ Gemini API 測試失敗: {e}")
        return False

if __name__ == "__main__":
    # 執行測試
    test_gemini_connection()