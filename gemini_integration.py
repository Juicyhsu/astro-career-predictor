import requests
import json
import os
from typing import Dict, Any

class GeminiAdvisor:
    def __init__(self, api_key: str = None):
        """
        初始化Gemini顧問
        api_key: Gemini API密鑰，如果為None會從環境變量讀取
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        
    def format_planets_info(self, planets_data: Dict[str, Any]) -> str:
        """格式化行星信息為文字描述"""
        planet_descriptions = []
        
        # 重要行星的描述
        important_planets = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn']
        
        for planet in important_planets:
            if planet in planets_data:
                data = planets_data[planet]
                sign = data.get('sign', 'Unknown')
                degree = data.get('degree', 0)
                planet_descriptions.append(f"{planet}在{sign}座{degree:.1f}度")
        
        # 外行星
        outer_planets = ['Uranus', 'Neptune', 'Pluto']
        for planet in outer_planets:
            if planet in planets_data:
                data = planets_data[planet]
                sign = data.get('sign', 'Unknown')
                planet_descriptions.append(f"{planet}在{sign}座")
        
        return "，".join(planet_descriptions)
    
    def create_prompt(self, planets_data: Dict[str, Any], houses_data: Dict[str, Any]) -> str:
        """創建給Gemini的提示詞"""
        planets_info = self.format_planets_info(planets_data)
        asc_info = houses_data.get('ASC', {})
        asc_sign = asc_info.get('sign', 'Unknown')
        
        prompt = f"""
請根據以下占星資訊，以專業占星師的角度，為這個人提供職業發展建議。請用繁體中文回答，語調要專業但易懂。

星盤配置：
{planets_info}
上升星座：{asc_sign}座

請提供以下內容：
1. 個人特質分析（基於太陽、月亮、上升星座）
2. 職業天賦洞察（基於水星、金星、火星的配置）
3. 成功策略建議（基於木星、土星的位置）
4. 需要注意的挑戰和成長方向

請以溫暖、鼓勵的語調撰寫，長度約300-400字，並使用適當的emoji增加可讀性。
"""
        return prompt
    
    def get_advice(self, planets_data: Dict[str, Any], houses_data: Dict[str, Any] = None) -> str:
        """
        獲取Gemini的職業建議
        
        Args:
            planets_data: 行星數據字典
            houses_data: 宮位數據字典
            
        Returns:
            str: AI生成的建議文字
        """
        if not self.api_key:
            return self._get_fallback_advice(planets_data, houses_data)
        
        try:
            # 創建請求內容
            prompt = self.create_prompt(planets_data, houses_data or {})
            
            headers = {
                'Content-Type': 'application/json',
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024,
                }
            }
            
            # 發送請求
            url = f"{self.base_url}?key={self.api_key}"
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    text = result['candidates'][0]['content']['parts'][0]['text']
                    return text
                else:
                    return self._get_fallback_advice(planets_data, houses_data)
            else:
                print(f"Gemini API 錯誤: {response.status_code}")
                return self._get_fallback_advice(planets_data, houses_data)
                
        except Exception as e:
            print(f"呼叫Gemini API時發生錯誤: {e}")
            return self._get_fallback_advice(planets_data, houses_data)
    
    def _get_fallback_advice(self, planets_data: Dict[str, Any], houses_data: Dict[str, Any] = None) -> str:
        """當API不可用時的備用建議"""
        sun_sign = planets_data.get('Sun', {}).get('sign', 'Unknown')
        moon_sign = planets_data.get('Moon', {}).get('sign', 'Unknown')
        asc_sign = houses_data.get('ASC', {}).get('sign', 'Unknown') if houses_data else 'Unknown'
        
        # 基本的星座特質映射
        sign_traits = {
            'Aries': {'trait': '積極主動', 'career': '領導管理', 'challenge': '學習耐心'},
            'Taurus': {'trait': '穩重踏實', 'career': '金融理財', 'challenge': '保持彈性'},
            'Gemini': {'trait': '機智靈活', 'career': '溝通媒體', 'challenge': '專注深度'},
            'Cancer': {'trait': '情感豐富', 'career': '照護服務', 'challenge': '建立界限'},
            'Leo': {'trait': '自信創意', 'career': '表演藝術', 'challenge': '團隊合作'},
            'Virgo': {'trait': '細心完美', 'career': '分析研究', 'challenge': '接受不完美'},
            'Libra': {'trait': '和諧平衡', 'career': '設計美學', 'challenge': '果斷決策'},
            'Scorpio': {'trait': '深度洞察', 'career': '心理研究', 'challenge': '信任他人'},
            'Sagittarius': {'trait': '樂觀探索', 'career': '教育旅遊', 'challenge': '注重細節'},
            'Capricorn': {'trait': '務實目標', 'career': '企業管理', 'challenge': '工作平衡'},
            'Aquarius': {'trait': '創新獨立', 'career': '科技創新', 'challenge': '情感表達'},
            'Pisces': {'trait': '直覺敏感', 'career': '藝術療癒', 'challenge': '現實適應'}
        }
        
        sun_info = sign_traits.get(sun_sign, {'trait': '獨特', 'career': '多元發展', 'challenge': '自我探索'})
        moon_info = sign_traits.get(moon_sign, {'trait': '複雜', 'career': '情感相關', 'challenge': '情緒管理'})
        
        advice = f"""
🌟 **個人特質分析**

您的太陽在{sun_sign}座，展現出{sun_info['trait']}的核心特質。月亮在{moon_sign}座，讓您在情感層面更加豐富細膩。上升{asc_sign}座則影響著他人對您的第一印象。

💼 **職業天賦洞察**

基於您的星盤配置，建議考慮{sun_info['career']}相關的領域發展。您天生具備在此領域發光發熱的潛質，特別是能夠發揮您{sun_info['trait']}的特長。

🎯 **成功策略建議**

1. 善用您的天賦優勢，在適合的環境中展現才華
2. 保持學習成長的心態，持續精進專業技能  
3. 建立良好的人際網絡，機會往往來自於人脈
4. 相信自己的直覺，但也要理性分析決策

⚠️ **成長挑戰**

需要特別注意{sun_info['challenge']}這個成長課題。透過conscious effort和實際練習，您可以將這個挑戰轉化為更大的優勢。

記住，星盤只是指引方向的羅盤，最終的成功還是要靠您的努力和選擇。祝您在人生道路上發光發熱！✨
"""
        return advice

# 使用範例
def demo_gemini_advice():
    """示範如何使用GeminiAdvisor"""
    # 模擬行星數據
    planets_data = {
        'Sun': {'sign': 'Leo', 'degree': 15.5},
        'Moon': {'sign': 'Cancer', 'degree': 22.3},
        'Mercury': {'sign': 'Virgo', 'degree': 8.7},
        'Venus': {'sign': 'Leo', 'degree': 25.1},
        'Mars': {'sign': 'Gemini', 'degree': 12.4}
    }
    
    houses_data = {
        'ASC': {'sign': 'Scorpio', 'degree': 5.8}
    }
    
    # 創建顧問實例
    advisor = GeminiAdvisor()  # API密鑰會從環境變量讀取
    
    # 獲取建議
    advice = advisor.get_advice(planets_data, houses_data)
    print(advice)

if __name__ == "__main__":
    demo_gemini_advice()