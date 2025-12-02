from forgec.lexer import Lexer
from forgec.parser import Parser
from forgec.semantic import TypeChecker
from forgec.ir_gen import IRGenerator
from forgec.diagnostics import DiagnosticEngine

def test_ir_gen_match_basic():
    source = """
    enum Option { Some(int), None }
    fn main() {
        let x = Option::Some(42);
        let result = match x {
            Option::Some(val) => { val + 1 },
            Option::None => { 0 }
        };
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
    
    print(f"\nGenerated IR (Match):\n{ir_code}")
    
    # Verify switch instruction
    assert "switch i8" in ir_code
    # Verify extraction of payload
    assert "extractvalue" in ir_code
    # Verify phi node
    assert "phi" in ir_code and "i32" in ir_code
