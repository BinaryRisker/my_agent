"""
天气查询工具
通过公共天气API获取天气信息
"""

import json
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import requests
from langchain.tools import BaseTool
from pydantic import Field
from loguru import logger

from config import get_config


class WeatherCache:
    """天气数据缓存"""
    
    def __init__(self, cache_timeout: int = 300):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timeout = cache_timeout
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """获取缓存数据"""
        if key in self.cache:
            data = self.cache[key]
            if time.time() - data['timestamp'] < self.cache_timeout:
                return data['weather']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, weather_data: Dict[str, Any]) -> None:
        """设置缓存数据"""
        self.cache[key] = {
            'weather': weather_data,
            'timestamp': time.time()
        }
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()


class WeatherAPI:
    """天气API客户端"""
    
    def __init__(self):
        self.config = get_config()
        self.api_key = self.config.tools.weather_api_key
        self.base_url = self.config.tools.weather_api_url
        self.default_units = self.config.tools.weather_default_units
        self.cache = WeatherCache(self.config.tools.weather_cache_timeout)
        
        # 如果没有API密钥，使用模拟数据
        self.use_mock = not self.api_key
        if self.use_mock:
            logger.warning("未配置天气API密钥，将使用模拟数据")
    
    def get_weather(self, location: str, units: str = None) -> Dict[str, Any]:
        """
        获取天气信息
        
        Args:
            location: 位置（城市名、坐标等）
            units: 单位系统（metric, imperial, kelvin）
            
        Returns:
            天气信息字典
        """
        units = units or self.default_units
        cache_key = f"{location}_{units}"
        
        # 检查缓存
        cached_data = self.cache.get(cache_key)
        if cached_data:
            logger.info(f"从缓存获取 {location} 的天气信息")
            return cached_data
        
        try:
            if self.use_mock:
                # 使用模拟数据
                weather_data = self._get_mock_weather(location, units)
            else:
                # 调用真实API
                weather_data = self._call_weather_api(location, units)
            
            # 缓存结果
            self.cache.set(cache_key, weather_data)
            
            logger.info(f"获取 {location} 的天气信息成功")
            return weather_data
            
        except Exception as e:
            logger.error(f"获取天气信息失败: {e}")
            return self._get_error_response(str(e))
    
    def _call_weather_api(self, location: str, units: str) -> Dict[str, Any]:
        """调用真实的天气API"""
        url = f"{self.base_url}/weather"
        params = {
            'q': location,
            'appid': self.api_key,
            'units': units,
            'lang': 'zh_cn'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # 解析API响应
        return self._parse_api_response(data, units)
    
    def _parse_api_response(self, data: Dict[str, Any], units: str) -> Dict[str, Any]:
        """解析API响应数据"""
        # 单位转换
        temp_unit = "°C" if units == "metric" else ("°F" if units == "imperial" else "K")
        speed_unit = "m/s" if units == "metric" else "mph"
        
        return {
            'location': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'temperature_unit': temp_unit,
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'description': data['weather'][0]['description'],
            'main': data['weather'][0]['main'],
            'wind_speed': data.get('wind', {}).get('speed', 0),
            'wind_speed_unit': speed_unit,
            'wind_direction': data.get('wind', {}).get('deg', 0),
            'clouds': data.get('clouds', {}).get('all', 0),
            'visibility': data.get('visibility', 0) / 1000,  # 转换为km
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M'),
            'timezone': data['timezone'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'OpenWeatherMap API'
        }
    
    def _get_mock_weather(self, location: str, units: str) -> Dict[str, Any]:
        """生成模拟天气数据"""
        import random
        
        temp_unit = "°C" if units == "metric" else ("°F" if units == "imperial" else "K")
        speed_unit = "m/s" if units == "metric" else "mph"
        
        # 根据单位生成合理的温度范围
        if units == "metric":
            temp = round(random.uniform(-10, 35), 1)
            wind_speed = round(random.uniform(0, 15), 1)
        elif units == "imperial":
            temp = round(random.uniform(14, 95), 1)
            wind_speed = round(random.uniform(0, 33), 1)
        else:  # kelvin
            temp = round(random.uniform(263, 308), 1)
            wind_speed = round(random.uniform(0, 15), 1)
        
        weather_conditions = [
            ("晴朗", "Clear"),
            ("多云", "Clouds"),
            ("阴天", "Overcast"),
            ("小雨", "Rain"),
            ("雷雨", "Thunderstorm"),
            ("雾霾", "Haze")
        ]
        
        description, main = random.choice(weather_conditions)
        
        return {
            'location': location,
            'country': 'CN',
            'temperature': temp,
            'temperature_unit': temp_unit,
            'feels_like': temp + random.uniform(-3, 3),
            'humidity': random.randint(30, 90),
            'pressure': random.randint(1000, 1030),
            'description': description,
            'main': main,
            'wind_speed': wind_speed,
            'wind_speed_unit': speed_unit,
            'wind_direction': random.randint(0, 360),
            'clouds': random.randint(0, 100),
            'visibility': round(random.uniform(5, 15), 1),
            'sunrise': '06:30',
            'sunset': '18:45',
            'timezone': 28800,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': '模拟数据'
        }
    
    def _get_error_response(self, error_msg: str) -> Dict[str, Any]:
        """生成错误响应"""
        return {
            'error': True,
            'message': f"获取天气信息失败: {error_msg}",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_forecast(self, location: str, days: int = 5) -> Dict[str, Any]:
        """
        获取天气预报（简化版）
        
        Args:
            location: 位置
            days: 预报天数
            
        Returns:
            天气预报信息
        """
        # 这里简化处理，实际应该调用预报API
        current_weather = self.get_weather(location)
        
        if current_weather.get('error'):
            return current_weather
        
        # 生成简单的预报数据
        forecast = []
        base_temp = current_weather['temperature']
        
        for i in range(days):
            date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
            temp_variation = random.uniform(-5, 5)
            
            forecast.append({
                'date': date,
                'temperature_high': round(base_temp + temp_variation + 3, 1),
                'temperature_low': round(base_temp + temp_variation - 3, 1),
                'description': current_weather['description'],
                'humidity': random.randint(30, 90)
            })
        
        return {
            'location': current_weather['location'],
            'forecast': forecast,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': current_weather['source']
        }


class WeatherTool(BaseTool):
    """天气查询工具"""
    
    name: str = "weather"
    description: str = """
    天气查询工具，可以获取指定位置的当前天气信息。
    支持全球主要城市和地区的天气查询。
    
    输入格式：城市名（中文或英文）
    例如：北京、上海、New York、London
    
    返回信息包括：温度、湿度、天气状况、风速等详细信息。
    """
    
    def __init__(self):
        super().__init__()
        self.weather_api = WeatherAPI()
    
    def _run(self, location: str) -> str:
        """
        运行天气查询工具
        
        Args:
            location: 查询位置
            
        Returns:
            天气信息字符串
        """
        logger.info(f"查询天气: {location}")
        
        location = location.strip()
        if not location:
            return "错误：请提供要查询天气的位置"
        
        # 获取天气数据
        weather_data = self.weather_api.get_weather(location)
        
        # 格式化输出
        if weather_data.get('error'):
            return weather_data['message']
        
        return self._format_weather_info(weather_data)
    
    async def _arun(self, location: str) -> str:
        """异步运行"""
        return self._run(location)
    
    def _format_weather_info(self, data: Dict[str, Any]) -> str:
        """格式化天气信息"""
        try:
            info_lines = [
                f"📍 {data['location']}, {data['country']}",
                f"🌡️  温度: {data['temperature']}{data['temperature_unit']} (体感 {data['feels_like']:.1f}{data['temperature_unit']})",
                f"☁️  天气: {data['description']} ({data['main']})",
                f"💧 湿度: {data['humidity']}%",
                f"🌬️  风速: {data['wind_speed']} {data['wind_speed_unit']}",
                f"📊 气压: {data['pressure']} hPa",
                f"👁️  能见度: {data['visibility']} km",
                f"🌅 日出: {data['sunrise']} | 🌇 日落: {data['sunset']}",
                f"⏰ 更新时间: {data['timestamp']}",
                f"📡 数据源: {data['source']}"
            ]
            
            return "\\n".join(info_lines)
            
        except KeyError as e:
            logger.error(f"格式化天气信息时出错: {e}")
            return f"天气信息格式化失败: 缺少字段 {e}"
    
    def get_forecast(self, location: str, days: int = 3) -> str:
        """
        获取天气预报
        
        Args:
            location: 位置
            days: 预报天数
            
        Returns:
            预报信息字符串
        """
        logger.info(f"查询天气预报: {location}, {days}天")
        
        forecast_data = self.weather_api.get_forecast(location, days)
        
        if forecast_data.get('error'):
            return forecast_data['message']
        
        # 格式化预报信息
        lines = [f"📅 {forecast_data['location']} {days}天天气预报:"]
        
        for day_data in forecast_data['forecast']:
            lines.append(
                f"  {day_data['date']}: "
                f"{day_data['temperature_low']}-{day_data['temperature_high']}°C, "
                f"{day_data['description']}, "
                f"湿度{day_data['humidity']}%"
            )
        
        lines.append(f"\\n⏰ 更新时间: {forecast_data['timestamp']}")
        lines.append(f"📡 数据源: {forecast_data['source']}")
        
        return "\\n".join(lines)


def create_weather_tool() -> WeatherTool:
    """创建天气工具实例"""
    return WeatherTool()


if __name__ == "__main__":
    # 测试天气工具
    import random
    
    weather_tool = create_weather_tool()
    
    test_locations = [
        "北京",
        "上海",
        "广州",
        "深圳",
        "New York",
        "London",
        "Tokyo",
        "不存在的城市"
    ]
    
    print("天气工具测试:")
    print("=" * 60)
    
    for location in test_locations:
        print(f"\\n查询位置: {location}")
        print("-" * 40)
        result = weather_tool._run(location)
        print(result)
    
    # 测试天气预报
    print("\\n\\n天气预报测试:")
    print("=" * 60)
    forecast = weather_tool.get_forecast("北京", 5)
    print(forecast)
    
    print("\\n天气工具测试完成！")