from forgec.lexer import Lexer
from forgec.parser import Parser
from forgec.semantic import TypeChecker
from forgec.ir_gen import IRGenerator
from forgec.diagnostics import DiagnosticEngine

def test_ir_gen_basic():
    source = "fn main() { let x = 10; let y = x + 5; }"
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    ir_gen = IRGenerator(diag)
    ir_code = ir_gen.generate(program)
    print(f"\nGenerated IR:\n{ir_code}")
    
    assert 'define void @"main"' in ir_code
    assert "alloca i32" in ir_code
    assert "store i32 10" in ir_code
    assert "add i32" in ir_code

def test_ir_gen_control_flow():
    source = "fn main() { let x = 10; let y = if x > 5 { 1; } else { 0; }; }"
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    ir_gen = IRGenerator(diag)
    ir_code = ir_gen.generate(program)
    
    print(f"\nGenerated IR (Control Flow):\n{ir_code}")
    
    assert "br i1" in ir_code
    assert 'label %"then"' in ir_code
    assert 'label %"else"' in ir_code
    assert "phi" in ir_code
    assert "i32 [1, %\"then\"], [0, %\"else\"]" in ir_code
