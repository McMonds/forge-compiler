import unittest
from forgec.lexer import Lexer, TokenType
from forgec.parser import Parser
from forgec.semantic import TypeChecker
from forgec.diagnostics import DiagnosticEngine
from forgec.ast_nodes import ModDecl, UseDecl

class TestModules(unittest.TestCase):
    def setUp(self):
        self.diagnostics = DiagnosticEngine()

    def parse(self, source):
        lexer = Lexer(source, self.diagnostics)
        tokens = lexer.tokenize()
        parser = Parser(tokens, self.diagnostics)
        return parser.parse()

    def test_mod_decl(self):
        source = "mod my_module;"
        program = self.parse(source)
        self.assertEqual(len(program.modules), 1)
        self.assertIsInstance(program.modules[0], ModDecl)
        self.assertEqual(program.modules[0].name, "my_module")
        self.assertFalse(program.modules[0].is_public)

    def test_pub_mod_decl(self):
        source = "pub mod my_public_module;"
        program = self.parse(source)
        self.assertEqual(len(program.modules), 1)
        self.assertTrue(program.modules[0].is_public)

    def test_use_decl(self):
        source = "use std::io::print;"
        program = self.parse(source)
        self.assertEqual(len(program.imports), 1)
        self.assertIsInstance(program.imports[0], UseDecl)
        self.assertEqual(program.imports[0].path, "std::io::print")

    def test_pub_use_decl(self):
        source = "pub use std::vec::Vec;"
        program = self.parse(source)
        self.assertEqual(len(program.imports), 1)
        self.assertTrue(program.imports[0].is_public)

    def test_pub_function(self):
        source = "pub fn foo() {}"
        program = self.parse(source)
        self.assertEqual(len(program.functions), 1)
        self.assertTrue(program.functions[0].is_public)

    def test_pub_struct(self):
        source = "pub struct Point { x: int, y: int }"
        program = self.parse(source)
        self.assertEqual(len(program.structs), 1)
        self.assertTrue(program.structs[0].is_public)

    def test_pub_enum(self):
        source = "pub enum Color { Red, Blue }"
        program = self.parse(source)
        self.assertEqual(len(program.enums), 1)
        self.assertTrue(program.enums[0].is_public)

    def test_semantic_mod_registration(self):
        source = """
        mod math;
        use std::io;
        """
        program = self.parse(source)
        checker = TypeChecker(self.diagnostics)
        checker.check(program)
        
        self.assertIn("math", checker.modules)
        self.assertFalse(self.diagnostics.has_errors)

    def test_duplicate_mod_error(self):
        source = """
        mod math;
        mod math;
        """
        program = self.parse(source)
        checker = TypeChecker(self.diagnostics)
        checker.check(program)
        
        self.assertTrue(self.diagnostics.has_errors)

if __name__ == '__main__':
    unittest.main()
