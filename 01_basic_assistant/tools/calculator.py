"""
计算器工具
提供安全的数学计算功能
"""

import ast
import math
import operator
from typing import Union, Dict, Any
from langchain.tools import BaseTool
from pydantic import Field
from loguru import logger

from config import get_config


class SafeMathEvaluator:
    """安全的数学表达式评估器"""
    
    # 支持的操作符
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    
    # 支持的数学函数
    functions = {
        'abs': abs,
        'round': round,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'sinh': math.sinh,
        'cosh': math.cosh,
        'tanh': math.tanh,
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,
        'exp': math.exp,
        'sqrt': math.sqrt,
        'ceil': math.ceil,
        'floor': math.floor,
        'degrees': math.degrees,
        'radians': math.radians,
        'factorial': math.factorial,
    }
    
    # 支持的常量
    constants = {
        'pi': math.pi,
        'e': math.e,
        'tau': math.tau,
        'inf': math.inf,
        'nan': math.nan,
    }
    
    def __init__(self):
        self.config = get_config()
        # 根据配置过滤允许的函数
        allowed_funcs = self.config.tools.calc_allowed_functions
        if allowed_funcs:
            self.functions = {
                name: func for name, func in self.functions.items() 
                if name in allowed_funcs
            }
    
    def eval_expression(self, expression: str) -> Union[float, int, str]:
        """
        安全评估数学表达式
        
        Args:
            expression: 数学表达式字符串
            
        Returns:
            计算结果或错误信息
        """
        try:
            # 检查表达式长度
            if len(expression) > self.config.tools.calc_max_expression_length:
                return f"表达式过长，最大长度为 {self.config.tools.calc_max_expression_length} 字符"
            
            # 解析表达式
            parsed = ast.parse(expression, mode='eval')
            result = self._eval_node(parsed.body)
            
            # 处理结果
            if isinstance(result, float):
                if result.is_infinite():
                    return "结果为无穷大"
                if math.isnan(result):
                    return "结果为 NaN (非数字)"
                # 保留6位小数
                if result == int(result):
                    return int(result)
                return round(result, 6)
            
            return result
            
        except ZeroDivisionError:
            return "错误：除数不能为零"
        except OverflowError:
            return "错误：数值溢出"
        except ValueError as e:
            return f"错误：数值错误 - {str(e)}"
        except SyntaxError:
            return "错误：表达式语法错误"
        except Exception as e:
            return f"错误：{str(e)}"
    
    def _eval_node(self, node):
        """递归评估AST节点"""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Name):
            # 检查是否为支持的常量
            if node.id in self.constants:
                return self.constants[node.id]
            else:
                raise ValueError(f"不支持的标识符: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.operators.get(type(node.op))
            if op:
                return op(left, right)
            else:
                raise ValueError(f"不支持的操作符: {type(node.op)}")
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.operators.get(type(node.op))
            if op:
                return op(operand)
            else:
                raise ValueError(f"不支持的一元操作符: {type(node.op)}")
        elif isinstance(node, ast.Call):
            # 函数调用
            func_name = node.func.id
            if func_name not in self.functions:
                raise ValueError(f"不支持的函数: {func_name}")
            
            func = self.functions[func_name]
            args = [self._eval_node(arg) for arg in node.args]
            
            return func(*args)
        else:
            raise ValueError(f"不支持的节点类型: {type(node)}")


class CalculatorTool(BaseTool):
    """计算器工具"""
    
    name: str = "calculator"
    description: str = """
    一个安全的数学计算器工具。可以执行基本的算术运算和数学函数。
    支持的操作：加法(+)、减法(-)、乘法(*)、除法(/)、幂运算(**)、取模(%)
    支持的函数：sin, cos, tan, log, sqrt, abs, round 等
    支持的常量：pi, e, tau
    
    示例：
    - 2 + 3 * 4
    - sqrt(16)
    - sin(pi/2)
    - log(e)
    """
    
    def __init__(self):
        super().__init__()
        self.evaluator = SafeMathEvaluator()
    
    def _run(self, expression: str) -> str:
        """
        运行计算器工具
        
        Args:
            expression: 要计算的数学表达式
            
        Returns:
            计算结果字符串
        """
        logger.info(f"计算表达式: {expression}")
        
        # 清理表达式
        expression = expression.strip()
        if not expression:
            return "错误：表达式不能为空"
        
        # 执行计算
        result = self.evaluator.eval_expression(expression)
        
        logger.info(f"计算结果: {result}")
        return str(result)
    
    async def _arun(self, expression: str) -> str:
        """异步运行（暂时与同步版本相同）"""
        return self._run(expression)
    
    def get_supported_functions(self) -> Dict[str, str]:
        """获取支持的函数列表"""
        return {
            name: func.__doc__ or f"{name} 函数"
            for name, func in self.evaluator.functions.items()
        }
    
    def get_examples(self) -> Dict[str, str]:
        """获取使用示例"""
        return {
            "基本运算": "2 + 3 * 4 - 1",
            "幂运算": "2 ** 3",
            "平方根": "sqrt(16)",
            "三角函数": "sin(pi/2)",
            "对数": "log(e)",
            "绝对值": "abs(-5)",
            "四舍五入": "round(3.14159, 2)",
            "复杂表达式": "sqrt(sin(pi/4)**2 + cos(pi/4)**2)",
        }


def create_calculator_tool() -> CalculatorTool:
    """创建计算器工具实例"""
    return CalculatorTool()


if __name__ == "__main__":
    # 测试计算器工具
    calc = create_calculator_tool()
    
    test_expressions = [
        "2 + 3",
        "10 / 3",
        "2 ** 3",
        "sqrt(16)",
        "sin(pi/2)",
        "log(e)",
        "abs(-5)",
        "round(3.14159, 2)",
        "2 + 3 * 4",
        "sqrt(sin(pi/4)**2 + cos(pi/4)**2)",
        "factorial(5)",
        "1 / 0",  # 测试除零
        "sqrt(-1)",  # 测试错误值
    ]
    
    print("计算器工具测试:")
    print("=" * 50)
    
    for expr in test_expressions:
        result = calc._run(expr)
        print(f"{expr:30} = {result}")
    
    print("\\n支持的函数:")
    functions = calc.get_supported_functions()
    for name in sorted(functions.keys()):
        print(f"  {name}")
    
    print("\\n使用示例:")
    examples = calc.get_examples()
    for desc, expr in examples.items():
        result = calc._run(expr)
        print(f"  {desc}: {expr} = {result}")
    
    print("\\n计算器工具测试完成！")