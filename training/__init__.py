# training/__init__.py - Make training package importable
from .trainer import FinancialTrainer
from .evaluator import ModelEvaluator

__all__ = ['FinancialTrainer', 'ModelEvaluator']
