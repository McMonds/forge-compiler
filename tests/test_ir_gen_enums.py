from forgec.lexer import Lexer
from forgec.parser import Parser
from forgec.semantic import TypeChecker
from forgec.ir_gen import IRGenerator
from forgec.diagnostics import DiagnosticEngine

def test_ir_gen_enum_basic():
    source = """
    enum Option { Some(int), None }
    fn main() {
        let x = Option::Some(42);
        let y = Option::None;
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
    
    print(f"\nGenerated IR (Enum):\n{ir_code}")
    
    # Verify enum type is defined (tagged union)
    assert "{i8, i32}" in ir_code
    # Verify enum instantiation
    assert "insertvalue" in ir_code
    # Verify tag is set
    assert "Option.tag" in ir_code
