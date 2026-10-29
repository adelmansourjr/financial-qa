"""Bounded Decimal arithmetic over cited document variables; never calls eval."""

from __future__ import annotations

import ast
from decimal import Decimal, DecimalException, localcontext

from pydantic import BaseModel, ConfigDict, Field

from .data import Evidence


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(default="", max_length=300)
    citations: list[str] = Field(default_factory=list, max_length=12)
    scale: str = Field(default="", pattern=r"^(|percent|thousand|million|billion)$")
    abstain: bool = False
    reason: str = Field(default="", max_length=500)


class ExecutionError(ValueError):
    """The proposed calculation is unsupported, ungrounded, or numerically invalid."""


def execute(plan: Plan, evidence: list[Evidence]) -> Decimal:
    if plan.abstain or not plan.expression:
        raise ExecutionError("No answer was proposed")
    if not plan.citations or len(set(plan.citations)) != len(plan.citations):
        raise ExecutionError("Provide unique evidence citations")
    by_id = {item.id: item for item in evidence}
    if set(plan.citations) - by_id.keys():
        raise ExecutionError("Citation does not refer to retrieved evidence")
    variables = {
        key: value
        for identifier in plan.citations
        for key, value in by_id[identifier].variables.items()
    }
    try:
        tree = ast.parse(plan.expression, mode="eval")
    except (SyntaxError, RecursionError) as error:
        raise ExecutionError("Invalid expression") from error
    if len(list(ast.walk(tree))) > 60:
        raise ExecutionError("Expression is too complex")
    used = set()

    def visit(node: ast.AST, depth: int = 0) -> Decimal:
        if depth > 12:
            raise ExecutionError("Expression is too deeply nested")
        if isinstance(node, ast.Expression):
            return visit(node.body, depth + 1)
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ExecutionError(f"Variable {node.id} has no cited source")
            used.add(node.id)
            return variables[node.id]
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value in {0, 1, 100}:
            return Decimal(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand, depth + 1)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                result = left / right
            else:
                raise ExecutionError("Only +, -, *, and / are supported")
            if not result.is_finite() or abs(result) > Decimal("1e18"):
                raise ExecutionError("Calculation exceeds numeric bounds")
            return result
        raise ExecutionError("Unsupported expression syntax or uncited constant")

    try:
        with localcontext() as context:
            context.prec = 28
            result = visit(tree)
    except DecimalException as error:
        raise ExecutionError("Calculation is undefined, including division by zero") from error
    if not used:
        raise ExecutionError("An answer must use at least one document variable")
    return result
