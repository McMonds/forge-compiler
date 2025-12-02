from forgec.lexer import Lexer
from forgec.parser import Parser
from forgec.semantic import TypeChecker
from forgec.ir_gen import IRGenerator
from forgec.diagnostics import DiagnosticEngine

def test_ir_gen_struct_basic():
    source = """
    struct Point { x: int, y: int }
    fn main() {
        let p = Point { x: 10, y: 20 };
    }
    """
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    ir_gen = IRGenerator(diag)
    ir_code = ir_gen.generate(program)
    
    print(f"\nGenerated IR (Struct):\n{ir_code}")
    
    # Verify struct type is defined
    assert "{i32, i32}" in ir_code
    # Verify struct instantiation
    assert "insertvalue" in ir_code

def test_ir_gen_struct_field_access():
    source = """
    struct Point { x: int, y: int }
    fn main() {
        let p = Point { x: 10, y: 20 };
        let x_val = p.x;
        let sum = p.x + p.y;
    }
    """
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    ir_gen = IRGenerator(diag)
    ir_code = ir_gen.generate(program)
    
    print(f"\nGenerated IR (Field Access):\n{ir_code}")
    
    # Verify field extraction
    assert "extractvalue" in ir_code
    # Verify arithmetic on extracted values
    assert "add i32" in ir_code
