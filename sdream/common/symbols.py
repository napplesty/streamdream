import z3
from enum import Enum
from typing import Dict, Any, Self, List, Optional
from operator import add, sub, mul, truediv
from functools import reduce
from copy import deepcopy

class ScalarSymType(Enum):
    INT = 0
    BOOLEAN = 1
    INFINITY = 2

class ScalarExprType(Enum):
    REAL_VALUE = 0
    SYMBOL_VALUE = 1
    ADD = 2
    SUB = 3
    MUL = 4
    DIV = 5
    EQUAL = 6
    NOT_EQUAL = 7
    GREATER_THAN = 8
    LESS_THAN = 9
    GREATER_THAN_OR_EQUAL = 10
    LESS_THAN_OR_EQUAL = 11

class ScalarBinaryExpr:
    def __init__(self, type: ScalarExprType, left, right = None):
        self.type = type
        self.left = left    
        self.right = right

class ScalarSym:
    def __init__(self, name: str, type: ScalarSymType, expr: Optional[ScalarBinaryExpr] = None):
        self.name = name
        self.type = type
        if expr is not None:
            self.expr = expr
        else:
            self.expr = ScalarBinaryExpr(ScalarExprType.SYMBOL_VALUE, self)

    @staticmethod
    def from_real(value: Any):
        if isinstance(value, int):
            return ScalarSym(f"%({value})", ScalarSymType.INT, ScalarBinaryExpr(ScalarExprType.REAL_VALUE, value))
        elif isinstance(value, bool):
            return ScalarSym(f"%({value})", ScalarSymType.BOOLEAN, ScalarBinaryExpr(ScalarExprType.REAL_VALUE, value))
        else:
            raise ValueError(f"Invalid value: {value}")

    def get_expr_name(self, left, right, expr_type):
        assert expr_type != ScalarExprType.SYMBOL_VALUE and expr_type != ScalarExprType.REAL_VALUE, "Invalid expression type"
        assert isinstance(left, ScalarSym) and isinstance(right, ScalarSym), "Invalid operands"
        
        op_str = None
        if expr_type == ScalarExprType.ADD:
            op_str = "+"
        elif expr_type == ScalarExprType.SUB:
            op_str = "-"
        elif expr_type == ScalarExprType.MUL:
            op_str = "*"
        elif expr_type == ScalarExprType.DIV:
            op_str = "/"
        elif expr_type == ScalarExprType.EQUAL:
            op_str = "=="
        elif expr_type == ScalarExprType.NOT_EQUAL:
            op_str = "!="
        elif expr_type == ScalarExprType.GREATER_THAN:
            op_str = ">"
        elif expr_type == ScalarExprType.LESS_THAN:
            op_str = "<"
        elif expr_type == ScalarExprType.GREATER_THAN_OR_EQUAL:
            op_str = ">="
        elif expr_type == ScalarExprType.LESS_THAN_OR_EQUAL:
            op_str = "<="
        else:
            raise ValueError(f"Invalid expression type: {expr_type}")
        return f"%({left.name}){op_str}%({right.name})"
    
    def get_elements(self) -> List[Self]:
        elements: List[Self] = []
        def recurse(sym: Self):
            nonlocal elements
            if sym is None:
                return
            if sym.type == ScalarSymType.SYMBOL_VALUE:
                elements.append(sym)
            else:
                recurse(sym.expr.left)
                recurse(sym.expr.right)
        recurse(self)
        return elements
    
    def __neg__(self):
        return ScalarSym(f"%(0)-%({self.name})", self.type, ScalarBinaryExpr(ScalarExprType.SUB, ScalarSym("0", ScalarSymType.INT, ScalarBinaryExpr(ScalarExprType.REAL_VALUE, 0)), self))
    
    def __add__(self, other: Self):
        return ScalarSym(self.get_expr_name(self, other, ScalarExprType.ADD), self.type, ScalarBinaryExpr(ScalarExprType.ADD, self, other))
    
    def __sub__(self, other: Self):
        return ScalarSym(self.get_expr_name(self, other, ScalarExprType.SUB), self.type, ScalarBinaryExpr(ScalarExprType.SUB, self, other))
    
    def __mul__(self, other: Self):
        return ScalarSym(self.get_expr_name(self, other, ScalarExprType.MUL), self.type, ScalarBinaryExpr(ScalarExprType.MUL, self, other))
    
    def __div__(self, other: Self):
        return ScalarSym(self.get_expr_name(self, other, ScalarExprType.DIV), self.type, ScalarBinaryExpr(ScalarExprType.DIV, self, other))
    
    def __floordiv__(self, other: Self):
        return ScalarSym(self.get_expr_name(self, other, ScalarExprType.FLOOR_DIV), self.type, ScalarBinaryExpr(ScalarExprType.FLOOR_DIV, self, other))
    
    def __eq__(self, other: Self):
        return ScalarSym(self.get_expr_name(self, other, ScalarExprType.EQUAL), ScalarSymType.BOOLEAN, ScalarBinaryExpr(ScalarExprType.EQUAL, self, other))
    
    def __ne__(self, other: Self):
        return ScalarSym(self.get_expr_name(self, other, ScalarExprType.NOT_EQUAL), ScalarSymType.BOOLEAN, ScalarBinaryExpr(ScalarExprType.NOT_EQUAL, self, other))
    
    def __gt__(self, other: Self):
        return ScalarSym(self.get_expr_name(self, other, ScalarExprType.GREATER_THAN), ScalarSymType.BOOLEAN, ScalarBinaryExpr(ScalarExprType.GREATER_THAN, self, other))
    
    def __ge__(self, other: Self):
        return ScalarSym(self.get_expr_name(self, other, ScalarExprType.GREATER_THAN_OR_EQUAL), ScalarSymType.BOOLEAN, ScalarBinaryExpr(ScalarExprType.GREATER_THAN_OR_EQUAL, self, other))
    
    def __lt__(self, other: Self):
        return ScalarSym(self.get_expr_name(self, other, ScalarExprType.LESS_THAN), ScalarSymType.BOOLEAN, ScalarBinaryExpr(ScalarExprType.LESS_THAN, self, other))
    
    def __le__(self, other: Self):
        return ScalarSym(self.get_expr_name(self, other, ScalarExprType.LESS_THAN_OR_EQUAL), ScalarSymType.BOOLEAN, ScalarBinaryExpr(ScalarExprType.LESS_THAN_OR_EQUAL, self, other))
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"ScalarSym({self.name})"
    
    def __hash__(self):
        return hash(self.name)

class SymContext:
    def __init__(self):
        self.syms: Dict[str, ScalarSym] = {}
        self.asserts = []

        self.alias_cache = {}
        self.predicate_cache = {}

        self.parent_context = None

    def symbol(self, name: str, type: ScalarSymType, value: Any= None):
        if value is not None:
            assert type == ScalarSymType.INT or type == ScalarSymType.BOOLEAN, "Invalid symbol type"
            assert isinstance(value, (int, float, bool)), "Invalid symbol value"
            self.syms[name] = ScalarSym(name, type, ScalarBinaryExpr(ScalarExprType.REAL_VALUE, value))
        else:
            assert type == ScalarSymType.INT or type == ScalarSymType.FLOAT, "Invalid symbol type"
            self.syms[name] = ScalarSym(name, type)
        return self.syms[name]
    
    def get_z3_expr(self, sym: ScalarSym):
        if sym.expr.type == ScalarExprType.SYMBOL_VALUE:
            if sym.type == ScalarSymType.INT:
                return z3.Int(sym.name)
            elif sym.type == ScalarSymType.BOOLEAN:
                return z3.Bool(sym.name)
            elif sym.type == ScalarSymType.INFINITY:
                return z3.oo
            else:
                raise ValueError(f"Invalid symbol type: {sym.type}")
        elif sym.expr.type == ScalarExprType.REAL_VALUE:
            return sym.expr.left
        else:
            raise ValueError(f"Invalid expression type: {sym.expr.type}")
    
    def verify_sat(self, predicate: ScalarSym, value: bool):
        solver = z3.Solver()
        def get_z3_expr_recurse(sym: ScalarSym, env: Dict[str, z3.ExprRef]):
            if sym.expr.type == ScalarExprType.SYMBOL_VALUE:
                return env[sym.name]
            elif sym.expr.type == ScalarExprType.REAL_VALUE:
                return sym.expr.left
            elif sym.expr.type == ScalarExprType.ADD:
                return get_z3_expr_recurse(sym.expr.left, env) + get_z3_expr_recurse(sym.expr.right, env)
            elif sym.expr.type == ScalarExprType.SUB:
                return get_z3_expr_recurse(sym.expr.left, env) - get_z3_expr_recurse(sym.expr.right, env)
            elif sym.expr.type == ScalarExprType.MUL:
                return get_z3_expr_recurse(sym.expr.left, env) * get_z3_expr_recurse(sym.expr.right, env)
            elif sym.expr.type == ScalarExprType.DIV:
                return get_z3_expr_recurse(sym.expr.left, env) / get_z3_expr_recurse(sym.expr.right, env)
            elif sym.expr.type == ScalarExprType.EQUAL:
                return get_z3_expr_recurse(sym.expr.left, env) == get_z3_expr_recurse(sym.expr.right, env)
            elif sym.expr.type == ScalarExprType.NOT_EQUAL:
                return get_z3_expr_recurse(sym.expr.left, env) != get_z3_expr_recurse(sym.expr.right, env)
            elif sym.expr.type == ScalarExprType.GREATER_THAN:
                return get_z3_expr_recurse(sym.expr.left, env) > get_z3_expr_recurse(sym.expr.right, env)
            elif sym.expr.type == ScalarExprType.GREATER_THAN_OR_EQUAL:
                return get_z3_expr_recurse(sym.expr.left, env) >= get_z3_expr_recurse(sym.expr.right, env)
            elif sym.expr.type == ScalarExprType.LESS_THAN:
                return get_z3_expr_recurse(sym.expr.left, env) < get_z3_expr_recurse(sym.expr.right, env)
            elif sym.expr.type == ScalarExprType.LESS_THAN_OR_EQUAL:
                return get_z3_expr_recurse(sym.expr.left, env) <= get_z3_expr_recurse(sym.expr.right, env)
            else:
                raise ValueError(f"Invalid expression type: {sym.expr.type}")                
        z3_symbols = {}
        for name, sym in self.syms.items():
            z3_symbols[name] = self.get_z3_expr(sym)
        for statement in self.asserts:
            solver.add(get_z3_expr_recurse(statement, z3_symbols))
        solver.add(get_z3_expr_recurse(predicate, z3_symbols) == value)
        result = solver.check()
        return result == z3.sat
    
    def verify_forall_sat(self, predicate: ScalarSym, value: bool):
        return not self.verify_sat(predicate, not value)

    def assume(self, expr: ScalarSym):
        self.asserts.append(expr)

    def sym_range(self, start: ScalarSym, end: ScalarSym, step: ScalarSym = None):
        assert isinstance(start, ScalarSym) and isinstance(end, ScalarSym), "Invalid operands"
        assert step is None or isinstance(step, ScalarSym), "Invalid step"
        assert start.type == ScalarSymType.INT and end.type == ScalarSymType.INT, "Invalid operands"
        assert step is None or step.type == ScalarSymType.INT, "Invalid step"
        self.assume(start < end)
        self.assume(step > 0)
        self.assume(start > 0)

        start_sym = self.symbol(f"%({start.name}_range_{end.name})_%(0)", ScalarSymType.INT)
        next_sym = self.symbol(f"%({start.name}_range_{end.name})_%({step.name})", ScalarSymType.INT)
        self.assume(start_sym == start)
        self.assume(next_sym == start_sym + step)
        return [start_sym, next_sym]
    
    def sym_if(self, condition: ScalarSym) -> 'SymContext':
        if condition.type != ScalarSymType.BOOLEAN:
            raise ValueError(f"Invalid condition: {condition}")
        if_context = SymContext()
        if_context.syms = self.syms.copy()
        if_context.asserts = self.asserts.copy()
        if_context.assume(deepcopy(condition))
        else_context = SymContext()
        else_context.syms = self.syms.copy()
        else_context.asserts = self.asserts.copy()
        if condition.expr.type == ScalarExprType.NOT_EQUAL:
            condition.expr.type = ScalarExprType.EQUAL
        elif condition.expr.type == ScalarExprType.EQUAL:
            condition.expr.type = ScalarExprType.NOT_EQUAL
        elif condition.expr.type == ScalarExprType.GREATER_THAN:
            condition.expr.type = ScalarExprType.LESS_THAN_OR_EQUAL
        elif condition.expr.type == ScalarExprType.GREATER_THAN_OR_EQUAL:
            condition.expr.type = ScalarExprType.LESS_THAN
        elif condition.expr.type == ScalarExprType.LESS_THAN:
            condition.expr.type = ScalarExprType.GREATER_THAN_OR_EQUAL
        elif condition.expr.type == ScalarExprType.LESS_THAN_OR_EQUAL:
            condition.expr.type = ScalarExprType.GREATER_THAN
        else:
            raise ValueError(f"Invalid condition: {condition}")
        else_context.assume(deepcopy(condition))
        if_context.parent_context = self
        else_context.parent_context = self
        return (if_context, else_context)
    
class TensorElemSym:
    def __init__(self, tensor_sym: 'TensorSym', name: str, shape: List[ScalarSym], index: List[ScalarSym], scalar_sym: ScalarSym):
        self.tensor_sym = tensor_sym
        self.name = name
        self.shape = shape
        self.index = index
        self.version_infos: List[ScalarSym] = [scalar_sym]

    def assign(self, scalar_sym: ScalarSym):
        self.version_infos.append(scalar_sym)

    def get_version(self, version: int):
        return self.version_infos[version]
    
    def get_latest_version(self):
        return self.version_infos[-1]

class TensorSym:
    def __init__(self, context: SymContext, name: str, shape: List[ScalarSym]):
        self.context = context
        self.name = name
        self.shape = shape
        self.active_elems: List[TensorElemSym] = []

    def __getitem__(self, index):
        if hasattr(index, "__iter__"):
            assert len(index) == len(self.shape), "Invalid access"
            self.active_elems.append(TensorElemSym(self, f"%({self.name})_%({'_'.join(map(lambda x: f"%({x.name})", index))})", self.shape, index, self.context.symbol(f"%({self.name})_%({'_'.join(map(str, index))})", ScalarSymType.INT)))
            return self.active_elems[-1]
        elif isinstance(index, ScalarSym):
            assert len(self.shape) == 1, "Invalid access"
            self.active_elems.append(TensorElemSym(self, f"%({self.name})_%({index.name})", self.shape, [index], self.context.symbol(f"%({self.name})_%({index.name})", ScalarSymType.INT)))
            return self.active_elems[-1]
        else:
            raise ValueError(f"Invalid index: {index}")
    
class TensorContext:
    def __init__(self):
        self.contexts = [SymContext()]
        self.active_contexts = {0}

    def refresh(self, context: SymContext):
        pass


